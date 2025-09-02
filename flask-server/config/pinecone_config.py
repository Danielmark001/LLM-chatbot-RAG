"""
Pinecone Vector Database Configuration and Setup
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeConfig:
    """Configuration class for Pinecone vector database"""
    
    def __init__(self):
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'exam-prep-chatbot')
        self.dimension = 1536  # OpenAI ada-002 embedding dimension
        self.metric = 'cosine'
        self.cloud = 'aws'
        self.region = 'us-east-1'
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
    
    def get_serverless_spec(self):
        """Get serverless specification for Pinecone index"""
        return ServerlessSpec(
            cloud=self.cloud,
            region=self.region
        )

class PineconeManager:
    """Manager class for Pinecone operations"""
    
    def __init__(self, config: Optional[PineconeConfig] = None):
        self.config = config or PineconeConfig()
        self.pc = Pinecone(api_key=self.config.api_key)
        self.index = None
        
    def create_index(self, index_name: Optional[str] = None, force_recreate: bool = False):
        """
        Create or connect to Pinecone index
        
        Args:
            index_name: Name of the index (defaults to config index_name)
            force_recreate: Whether to delete and recreate existing index
        """
        index_name = index_name or self.config.index_name
        
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name in existing_indexes:
                if force_recreate:
                    logger.info(f"Deleting existing index: {index_name}")
                    self.pc.delete_index(index_name)
                else:
                    logger.info(f"Connecting to existing index: {index_name}")
                    self.index = self.pc.Index(index_name)
                    return self.index
            
            if index_name not in existing_indexes or force_recreate:
                logger.info(f"Creating new index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=self.config.get_serverless_spec()
                )
                
                # Wait for index to be ready
                import time
                time.sleep(30)  # Wait for index initialization
                
            self.index = self.pc.Index(index_name)
            logger.info(f"Successfully connected to index: {index_name}")
            return self.index
            
        except Exception as e:
            logger.error(f"Error creating/connecting to index: {e}")
            raise
    
    def get_index_stats(self):
        """Get statistics about the current index"""
        if not self.index:
            raise ValueError("No index connected. Call create_index() first.")
        
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            raise
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]], namespace: str = ""):
        """
        Upsert vectors to the index
        
        Args:
            vectors: List of vector dictionaries with 'id', 'values', and 'metadata'
            namespace: Namespace for the vectors
        """
        if not self.index:
            raise ValueError("No index connected. Call create_index() first.")
        
        try:
            # Batch upsert in chunks of 100
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
                logger.info(f"Upserted {total_upserted}/{len(vectors)} vectors")
                
            logger.info(f"Successfully upserted {len(vectors)} vectors to namespace '{namespace}'")
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            raise
    
    def query_vectors(self, 
                     query_vector: List[float], 
                     top_k: int = 10, 
                     namespace: str = "",
                     filter_dict: Optional[Dict] = None,
                     include_metadata: bool = True,
                     include_values: bool = False) -> Dict:
        """
        Query vectors from the index
        
        Args:
            query_vector: Query vector
            top_k: Number of top results to return
            namespace: Namespace to query
            filter_dict: Metadata filter
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values in results
        
        Returns:
            Query results dictionary
        """
        if not self.index:
            raise ValueError("No index connected. Call create_index() first.")
        
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=include_values
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            raise
    
    def delete_vectors(self, ids: List[str], namespace: str = ""):
        """Delete vectors by IDs"""
        if not self.index:
            raise ValueError("No index connected. Call create_index() first.")
        
        try:
            self.index.delete(ids=ids, namespace=namespace)
            logger.info(f"Deleted {len(ids)} vectors from namespace '{namespace}'")
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
    
    def delete_namespace(self, namespace: str):
        """Delete all vectors in a namespace"""
        if not self.index:
            raise ValueError("No index connected. Call create_index() first.")
        
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted all vectors from namespace '{namespace}'")
        except Exception as e:
            logger.error(f"Error deleting namespace: {e}")
            raise
    
    def list_namespaces(self):
        """List all namespaces in the index"""
        if not self.index:
            raise ValueError("No index connected. Call create_index() first.")
        
        try:
            stats = self.get_index_stats()
            return list(stats.get('namespaces', {}).keys())
        except Exception as e:
            logger.error(f"Error listing namespaces: {e}")
            raise

# Singleton instance for global use
_pinecone_manager = None

def get_pinecone_manager() -> PineconeManager:
    """Get or create global Pinecone manager instance"""
    global _pinecone_manager
    if _pinecone_manager is None:
        _pinecone_manager = PineconeManager()
    return _pinecone_manager