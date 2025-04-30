from pymongo import MongoClient
import os
from dotenv import load_dotenv
from bson import ObjectId

# Load environment variables
load_dotenv()

# MongoDB configuration
mongo_uri = os.environ.get('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['file_database']

def create_collection(collection_name):
    """
    Create a new MongoDB collection
    
    Args:
        collection_name (str): Name of the collection to create
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db.create_collection(collection_name)
        print(f"Collection '{collection_name}' created successfully")
        return True
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False

def drop_collection(collection_name):
    """
    Drop a MongoDB collection
    
    Args:
        collection_name (str): Name of the collection to drop
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db.drop_collection(collection_name)
        print(f"Collection '{collection_name}' dropped successfully")
        return True
    except Exception as e:
        print(f"Error dropping collection: {e}")
        return False
    
def create_document(collection_name, files):
    """
    Create or update document metadata in a collection
    
    Args:
        collection_name (str): Name of the collection
        files (list): List of file dictionaries with 'name' and 'url' keys
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        for file in files:
            doc = {
                "file": file['name'],
                "url": file['url']
            }           
            db[collection_name].update_one(
                {"file": file['name']},
                {"$set": doc},           
                upsert=True             
            )
        return True
    except Exception as e:
        print(f"Error creating document: {e}")
        return False

def delete_document(collection_name, doc_id):
    """
    Delete a document from a collection
    
    Args:
        collection_name (str): Name of the collection
        doc_id (str): MongoDB ObjectId as string
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        result = db[collection_name].delete_one({"_id": ObjectId(doc_id)})
        if result.deleted_count > 0:
            print(f"Document deleted successfully from '{collection_name}'")
            return True
        else:
            print(f"Document with ID {doc_id} not found in '{collection_name}'")
            return False
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False

def get_collections():
    """
    Get all collection names
    
    Returns:
        list: List of collection names or False on error
    """
    try:
        collections = db.list_collection_names()
        return collections
    except Exception as e:
        print(f"Error getting collections: {e}")
        return False

def get_documents(collection_name):
    """
    Get all documents in a collection
    
    Args:
        collection_name (str): Name of the collection
        
    Returns:
        list: List of documents with ObjectId converted to string or False on error
    """
    try:
        documents = list(db[collection_name].find())
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            doc['_id'] = str(doc['_id'])
        return documents
    except Exception as e:
        print(f"Error getting documents: {e}")
        return False
