"""
Hybrid Retrieval System combining Dense (ColBERT) and Sparse (BM25) retrieval
with context re-ranking and query compression
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain.schema.embeddings import Embeddings
from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Container for retrieval results with scoring information"""
    document: Document
    score: float
    retrieval_type: str  # 'dense', 'sparse', or 'hybrid'
    rank: int

class QueryCompressor:
    """Compresses and optimizes queries for better retrieval"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def compress_query(self, query: str, max_length: int = 50) -> str:
        """
        Compress query by extracting key terms and concepts
        
        Args:
            query: Original query text
            max_length: Maximum length of compressed query
            
        Returns:
            Compressed query string
        """
        # Simple keyword extraction approach
        # In practice, you might use more sophisticated methods
        import re
        
        # Remove stop words and extract key terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        words = re.findall(r'\w+', query.lower())
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Rejoin key terms
        compressed = ' '.join(key_words[:max_length])
        
        logger.info(f"Compressed query: '{query}' -> '{compressed}'")
        return compressed

class ColBERTRetriever:
    """Dense retrieval using ColBERT-style embeddings"""
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        try:
            # Use a smaller model for demonstration
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized ColBERT-style retriever with {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load ColBERT model, using fallback: {e}")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
    def encode_documents(self, documents: List[Document]) -> np.ndarray:
        """Encode documents to dense vectors"""
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to dense vector"""
        return self.model.encode([query])[0]
    
    def retrieve(self, 
                query: str, 
                documents: List[Document], 
                document_embeddings: np.ndarray, 
                top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve documents using dense similarity
        
        Args:
            query: Search query
            documents: List of documents
            document_embeddings: Pre-computed document embeddings
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        query_embedding = self.encode_query(query)
        
        # Calculate cosine similarity
        similarities = np.dot(document_embeddings, query_embedding) / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            result = RetrievalResult(
                document=documents[idx],
                score=float(similarities[idx]),
                retrieval_type="dense",
                rank=rank
            )
            results.append(result)
            
        return results

class BM25Retriever:
    """Sparse retrieval using BM25"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        
    def fit(self, documents: List[Document]):
        """Fit BM25 on document corpus"""
        self.documents = documents
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        logger.info(f"Fitted BM25 on {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve documents using BM25 scoring
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        if self.bm25 is None:
            raise ValueError("BM25 not fitted. Call fit() first.")
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only include results with positive scores
                result = RetrievalResult(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                    retrieval_type="sparse",
                    rank=rank
                )
                results.append(result)
                
        return results

class ContextReRanker:
    """Re-ranks retrieved documents based on context and relevance"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            logger.info(f"Initialized re-ranker with {model_name}")
        except Exception as e:
            logger.warning(f"Could not load cross-encoder model: {e}")
            # Fallback to simple similarity-based re-ranking
            self.model = None
            self.similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def rerank(self, 
              query: str, 
              results: List[RetrievalResult], 
              top_k: int = 10,
              context: Optional[str] = None) -> List[RetrievalResult]:
        """
        Re-rank retrieval results based on relevance
        
        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of final results to return
            context: Optional conversation context
            
        Returns:
            Re-ranked results
        """
        if not results:
            return results
            
        if self.model is not None:
            # Use cross-encoder for re-ranking
            query_doc_pairs = []
            for result in results:
                enhanced_query = query
                if context:
                    enhanced_query = f"{context} {query}"
                query_doc_pairs.append([enhanced_query, result.document.page_content])
            
            # Get relevance scores
            relevance_scores = self.model.predict(query_doc_pairs)
            
            # Update scores and re-sort
            for i, result in enumerate(results):
                result.score = float(relevance_scores[i])
                
            # Sort by new scores
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            
        else:
            # Fallback: Use semantic similarity for re-ranking
            query_embedding = self.similarity_model.encode([query])[0]
            
            for result in results:
                doc_embedding = self.similarity_model.encode([result.document.page_content])[0]
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                # Combine with original score
                result.score = 0.7 * result.score + 0.3 * float(similarity)
            
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result.rank = i
            
        return reranked_results[:top_k]

class HybridRetriever:
    """Hybrid retrieval system combining dense and sparse retrieval with re-ranking"""
    
    def __init__(self, 
                 dense_weight: float = 0.6,
                 sparse_weight: float = 0.4,
                 enable_query_compression: bool = True,
                 enable_reranking: bool = True):
        """
        Initialize hybrid retriever
        
        Args:
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
            enable_query_compression: Whether to use query compression
            enable_reranking: Whether to enable re-ranking
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.enable_query_compression = enable_query_compression
        self.enable_reranking = enable_reranking
        
        # Initialize components
        self.query_compressor = QueryCompressor() if enable_query_compression else None
        self.colbert_retriever = ColBERTRetriever()
        self.bm25_retriever = BM25Retriever()
        self.reranker = ContextReRanker() if enable_reranking else None
        
        # Document storage
        self.documents = []
        self.document_embeddings = None
        self.is_fitted = False
        
        logger.info("Initialized HybridRetriever with dense_weight={}, sparse_weight={}".format(
            dense_weight, sparse_weight))
    
    def fit(self, documents: List[Document]):
        """
        Fit the retriever on document corpus
        
        Args:
            documents: List of documents to index
        """
        logger.info(f"Fitting HybridRetriever on {len(documents)} documents")
        
        self.documents = documents
        
        # Fit BM25
        self.bm25_retriever.fit(documents)
        
        # Pre-compute dense embeddings
        logger.info("Computing dense embeddings...")
        self.document_embeddings = self.colbert_retriever.encode_documents(documents)
        
        self.is_fitted = True
        logger.info("HybridRetriever fitting completed")
    
    def retrieve(self, 
                query: str, 
                top_k: int = 10,
                context: Optional[str] = None,
                dense_top_k: Optional[int] = None,
                sparse_top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid approach
        
        Args:
            query: Search query
            top_k: Final number of results to return
            context: Optional conversation context
            dense_top_k: Number of dense results (defaults to top_k * 2)
            sparse_top_k: Number of sparse results (defaults to top_k * 2)
            
        Returns:
            List of hybrid retrieval results
        """
        if not self.is_fitted:
            raise ValueError("Retriever not fitted. Call fit() first.")
            
        # Set default values for component retrievals
        dense_top_k = dense_top_k or (top_k * 2)
        sparse_top_k = sparse_top_k or (top_k * 2)
        
        # Optionally compress query
        search_query = query
        if self.enable_query_compression and self.query_compressor:
            search_query = self.query_compressor.compress_query(query)
        
        # Dense retrieval
        logger.info("Performing dense retrieval...")
        dense_results = self.colbert_retriever.retrieve(
            search_query, self.documents, self.document_embeddings, dense_top_k
        )
        
        # Sparse retrieval
        logger.info("Performing sparse retrieval...")
        sparse_results = self.bm25_retriever.retrieve(search_query, sparse_top_k)
        
        # Combine and normalize scores
        all_results = {}
        
        # Add dense results
        for result in dense_results:
            doc_id = id(result.document)  # Use object id as key
            if doc_id not in all_results:
                all_results[doc_id] = result
                all_results[doc_id].score *= self.dense_weight
            else:
                # If already exists from sparse, combine scores
                all_results[doc_id].score += result.score * self.dense_weight
                all_results[doc_id].retrieval_type = "hybrid"
        
        # Add sparse results
        for result in sparse_results:
            doc_id = id(result.document)
            if doc_id not in all_results:
                all_results[doc_id] = result
                all_results[doc_id].score *= self.sparse_weight
            else:
                # Combine with existing dense score
                all_results[doc_id].score += result.score * self.sparse_weight
                all_results[doc_id].retrieval_type = "hybrid"
        
        # Convert to list and sort by combined score
        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply re-ranking if enabled
        if self.enable_reranking and self.reranker:
            logger.info("Applying context re-ranking...")
            combined_results = self.reranker.rerank(
                query, combined_results, top_k * 2, context
            )
        
        # Return top-k results
        final_results = combined_results[:top_k]
        
        # Update final ranks
        for i, result in enumerate(final_results):
            result.rank = i
        
        logger.info(f"Retrieved {len(final_results)} hybrid results")
        return final_results
    
    def get_retrieval_stats(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Get statistics about retrieval results"""
        if not results:
            return {}
            
        stats = {
            "total_results": len(results),
            "dense_count": len([r for r in results if r.retrieval_type == "dense"]),
            "sparse_count": len([r for r in results if r.retrieval_type == "sparse"]),
            "hybrid_count": len([r for r in results if r.retrieval_type == "hybrid"]),
            "avg_score": np.mean([r.score for r in results]),
            "max_score": max([r.score for r in results]),
            "min_score": min([r.score for r in results])
        }
        
        return stats