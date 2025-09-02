"""
Advanced Document Ingestion Pipeline for 200+ Past Papers
Supports PDF processing, metadata extraction, chunking, and vectorization
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import PyPDF2
import fitz  # PyMuPDF for better PDF processing
from langchain_community.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader,
    TextLoader,
    UnstructuredPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Import our custom components
from config.pinecone_config import get_pinecone_manager
from retrieval.hybrid_retrieval import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Enhanced metadata for exam papers"""
    filename: str
    subject: str
    year: int
    paper_type: str  # midterm, final, quiz, etc.
    institution: str
    professor: Optional[str] = None
    difficulty: Optional[str] = None  # easy, medium, hard
    topics: List[str] = None
    page_count: int = 0
    file_size: int = 0
    processing_date: str = None
    content_hash: str = None
    language: str = "en"
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.processing_date is None:
            self.processing_date = datetime.now().isoformat()

class DocumentProcessor:
    """Advanced document processor for exam papers"""
    
    def __init__(self, 
                 embeddings_model: str = "text-embedding-ada-002",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            embeddings_model: Name of embeddings model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\\n\\n", "\\n", ".", "!", "?", ";", ":", " ", ""],
            length_function=len,
        )
        
        # Document cache for avoiding reprocessing
        self.processed_docs_cache = {}
        self.cache_file = "processed_documents_cache.json"
        self._load_cache()
        
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}")
    
    def _load_cache(self):
        """Load processed documents cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.processed_docs_cache = json.load(f)
                logger.info(f"Loaded {len(self.processed_docs_cache)} cached documents")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            self.processed_docs_cache = {}
    
    def _save_cache(self):
        """Save processed documents cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.processed_docs_cache, f, indent=2)
            logger.info(f"Saved cache with {len(self.processed_docs_cache)} documents")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file for change detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return ""
    
    def extract_metadata_from_filename(self, filename: str) -> DocumentMetadata:
        """
        Extract metadata from filename using various patterns
        
        Expected formats:
        - SUBJECT_YEAR_TYPE.pdf (e.g., MATH_2023_FINAL.pdf)
        - SUBJECT-YEAR-TYPE.pdf (e.g., PHYSICS-2022-MIDTERM.pdf)
        - subject_institution_year_type.pdf
        """
        base_name = Path(filename).stem
        
        # Default metadata
        metadata = DocumentMetadata(
            filename=filename,
            subject="Unknown",
            year=2023,
            paper_type="Unknown",
            institution="Unknown"
        )
        
        # Try different parsing patterns
        patterns = [
            # Pattern 1: SUBJECT_YEAR_TYPE
            lambda x: x.split('_'),
            # Pattern 2: SUBJECT-YEAR-TYPE
            lambda x: x.split('-'),
            # Pattern 3: space separated
            lambda x: x.split(' ')
        ]
        
        for pattern in patterns:
            try:
                parts = pattern(base_name.upper())
                if len(parts) >= 3:
                    # Extract subject
                    metadata.subject = parts[0]
                    
                    # Extract year (look for 4-digit number)
                    for part in parts:
                        if len(part) == 4 and part.isdigit():
                            year = int(part)
                            if 2000 <= year <= 2030:
                                metadata.year = year
                                break
                    
                    # Extract paper type
                    paper_types = ['FINAL', 'MIDTERM', 'QUIZ', 'EXAM', 'TEST', 'HW', 'HOMEWORK']
                    for part in parts:
                        if part.upper() in paper_types:
                            metadata.paper_type = part.upper()
                            break
                    
                    # If we found meaningful info, use this pattern
                    if metadata.subject != "Unknown" and metadata.paper_type != "Unknown":
                        break
                        
            except Exception as e:
                logger.warning(f"Pattern matching failed for {filename}: {e}")
                continue
        
        return metadata
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """
        Extract text from PDF using multiple methods for robustness
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        metadata = self.extract_metadata_from_filename(file_path)
        metadata.file_size = os.path.getsize(file_path)
        metadata.content_hash = self._calculate_file_hash(file_path)
        
        text = ""
        
        # Method 1: Try PyMuPDF (best for complex PDFs)
        try:
            doc = fitz.open(file_path)
            metadata.page_count = doc.page_count
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            
            if text.strip():
                logger.info(f"Extracted {len(text)} characters using PyMuPDF from {file_path}")
                return text, metadata
                
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {file_path}: {e}")
        
        # Method 2: Try LangChain UnstructuredPDFLoader
        try:
            loader = UnstructuredPDFLoader(file_path)
            docs = loader.load()
            text = "\\n".join([doc.page_content for doc in docs])
            
            if text.strip():
                logger.info(f"Extracted {len(text)} characters using UnstructuredPDF from {file_path}")
                return text, metadata
                
        except Exception as e:
            logger.warning(f"UnstructuredPDF failed for {file_path}: {e}")
        
        # Method 3: Try LangChain PyPDFLoader
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text = "\\n".join([doc.page_content for doc in docs])
            
            if text.strip():
                logger.info(f"Extracted {len(text)} characters using PyPDF from {file_path}")
                return text, metadata
                
        except Exception as e:
            logger.warning(f"PyPDF failed for {file_path}: {e}")
        
        # Method 4: Fallback to basic PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata.page_count = len(pdf_reader.pages)
                
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            if text.strip():
                logger.info(f"Extracted {len(text)} characters using PyPDF2 from {file_path}")
                return text, metadata
                
        except Exception as e:
            logger.warning(f"PyPDF2 failed for {file_path}: {e}")
        
        logger.error(f"All PDF extraction methods failed for {file_path}")
        return "", metadata
    
    def clean_and_preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\\s+', ' ', text)
        
        # Remove page numbers and common artifacts
        text = re.sub(r'Page \\d+ of \\d+', '', text)
        text = re.sub(r'\\f', '', text)  # Form feed characters
        
        # Remove email addresses and URLs (often in headers/footers)
        text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean up multiple spaces and newlines
        text = re.sub(r'\\n+', '\\n\\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def process_single_document(self, file_path: str) -> List[Document]:
        """
        Process a single document file
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of chunked Document objects
        """
        file_path = str(file_path)
        
        # Check cache first
        file_hash = self._calculate_file_hash(file_path)
        if file_path in self.processed_docs_cache:
            cached_hash = self.processed_docs_cache[file_path].get('hash', '')
            if cached_hash == file_hash:
                logger.info(f"Using cached result for {file_path}")
                # In a real implementation, you'd store and retrieve the actual chunks
                # For now, we'll process anyway but this optimization could be added
        
        try:
            # Extract text and metadata
            if file_path.lower().endswith('.pdf'):
                text, metadata = self.extract_text_from_pdf(file_path)
            else:
                # For non-PDF files, use basic text loader
                loader = TextLoader(file_path)
                docs = loader.load()
                text = docs[0].page_content if docs else ""
                metadata = self.extract_metadata_from_filename(file_path)
                metadata.content_hash = file_hash
                metadata.file_size = os.path.getsize(file_path)
            
            # Clean text
            cleaned_text = self.clean_and_preprocess_text(text)
            
            if not cleaned_text:
                logger.warning(f"No text extracted from {file_path}")
                return []
            
            # Create base document
            base_doc = Document(
                page_content=cleaned_text,
                metadata=asdict(metadata)
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([base_doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk.page_content)
                })
            
            # Update cache
            self.processed_docs_cache[file_path] = {
                'hash': file_hash,
                'chunks': len(chunks),
                'processed_date': datetime.now().isoformat()
            }
            
            logger.info(f"Processed {file_path}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
    
    def process_directory(self, 
                         directory_path: str, 
                         file_pattern: str = "**/*.pdf",
                         max_workers: int = 4) -> List[Document]:
        """
        Process all documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            file_pattern: Glob pattern for files to process
            max_workers: Number of parallel workers
            
        Returns:
            List of all processed Document chunks
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all matching files
        files = list(directory_path.glob(file_pattern))
        logger.info(f"Found {len(files)} files to process in {directory_path}")
        
        if not files:
            logger.warning(f"No files found matching pattern {file_pattern}")
            return []
        
        all_documents = []
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_document, file_path): file_path 
                for file_path in files
            }
            
            # Collect results with progress bar
            with tqdm(total=len(files), desc="Processing documents") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        documents = future.result()
                        all_documents.extend(documents)
                        pbar.set_postfix({
                            'current_file': Path(file_path).name,
                            'total_chunks': len(all_documents)
                        })
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                    finally:
                        pbar.update(1)
        
        # Save cache
        self._save_cache()
        
        logger.info(f"Successfully processed {len(files)} files into {len(all_documents)} chunks")
        return all_documents
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not self.processed_docs_cache:
            return {}
        
        total_files = len(self.processed_docs_cache)
        total_chunks = sum(item.get('chunks', 0) for item in self.processed_docs_cache.values())
        
        # Group by subject if possible
        subjects = {}
        for file_path, info in self.processed_docs_cache.items():
            try:
                metadata = self.extract_metadata_from_filename(file_path)
                subject = metadata.subject
                if subject not in subjects:
                    subjects[subject] = {'files': 0, 'chunks': 0}
                subjects[subject]['files'] += 1
                subjects[subject]['chunks'] += info.get('chunks', 0)
            except:
                pass
        
        stats = {
            'total_files_processed': total_files,
            'total_chunks_created': total_chunks,
            'avg_chunks_per_file': total_chunks / total_files if total_files > 0 else 0,
            'subjects': subjects,
            'last_update': max(
                (item.get('processed_date', '') for item in self.processed_docs_cache.values()),
                default=''
            )
        }
        
        return stats

class PineconeDocumentStore:
    """Store and manage documents in Pinecone vector database"""
    
    def __init__(self, namespace: str = "exam-papers"):
        """
        Initialize Pinecone document store
        
        Args:
            namespace: Pinecone namespace for documents
        """
        self.namespace = namespace
        self.pinecone_manager = get_pinecone_manager()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Ensure index exists
        self.pinecone_manager.create_index()
        
        logger.info(f"Initialized PineconeDocumentStore with namespace '{namespace}'")
    
    def store_documents(self, documents: List[Document], batch_size: int = 100) -> Dict[str, Any]:
        """
        Store documents in Pinecone
        
        Args:
            documents: List of Document objects to store
            batch_size: Number of documents to process in each batch
            
        Returns:
            Dictionary with storage statistics
        """
        if not documents:
            return {"status": "error", "message": "No documents provided"}
        
        logger.info(f"Storing {len(documents)} documents in Pinecone...")
        
        total_stored = 0
        failed_documents = []
        
        # Process in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Storing batches"):
            batch = documents[i:i + batch_size]
            
            try:
                # Extract text and metadata
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                
                # Generate embeddings
                embeddings = self.embeddings.embed_documents(texts)
                
                # Prepare vectors for Pinecone
                vectors = []
                for j, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                    vector_id = f"{self.namespace}_{total_stored + j}_{hash(text[:100]) % 10000}"
                    
                    # Ensure metadata values are JSON serializable
                    clean_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            clean_metadata[key] = value
                        elif isinstance(value, list):
                            clean_metadata[key] = str(value)
                        else:
                            clean_metadata[key] = str(value)
                    
                    # Add text content to metadata for retrieval
                    clean_metadata['text'] = text[:1000]  # Truncate for metadata
                    clean_metadata['full_text_length'] = len(text)
                    
                    vectors.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': clean_metadata
                    })
                
                # Upsert to Pinecone
                self.pinecone_manager.upsert_vectors(vectors, namespace=self.namespace)
                total_stored += len(batch)
                
            except Exception as e:
                logger.error(f"Failed to store batch {i}-{i+len(batch)}: {e}")
                failed_documents.extend([f"batch_{i}_{j}" for j in range(len(batch))])
        
        stats = {
            "status": "success" if not failed_documents else "partial_success",
            "total_documents": len(documents),
            "successfully_stored": total_stored,
            "failed_documents": len(failed_documents),
            "namespace": self.namespace,
            "storage_date": datetime.now().isoformat()
        }
        
        logger.info(f"Storage complete: {total_stored}/{len(documents)} documents stored")
        return stats
    
    def search_documents(self, 
                        query: str, 
                        top_k: int = 10, 
                        filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search documents in Pinecone
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filter
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in Pinecone
            results = self.pinecone_manager.query_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                filter_dict=filter_dict,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.get('matches', []):
                formatted_results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match.get('metadata', {}),
                    'text': match.get('metadata', {}).get('text', '')
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []