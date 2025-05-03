from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Blob Storage configuration
storage_account_name = "sc1015filestorage"

# Get a Blob Service Client instance
def get_blob_service_client():
    """
    Get a BlobServiceClient instance using environment variables
    
    Returns:
        BlobServiceClient: The initialized client or None if error
    """
    try:
        connection_string = os.environ.get('AZURE_CONN_STRING')
        if not connection_string:
            print("Warning: AZURE_CONN_STRING environment variable is not set")
            # For testing purposes only:
            return DummyBlobServiceClient()
        
        return BlobServiceClient.from_connection_string(connection_string)
    except Exception as error:
        print(f"Error creating BlobServiceClient: {error}")
        # For testing purposes only:
        return DummyBlobServiceClient()

# Dummy class for testing without Azure connection
class DummyBlobServiceClient:
    """
    A dummy implementation of BlobServiceClient for testing without Azure
    """
    def __init__(self):
        self.account_name = "dummy"
        self.credential = type('obj', (object,), {'account_key': 'dummy_key'})
    
    def create_container(self, container_name):
        print(f"[DUMMY] Creating container: {container_name}")
        return DummyContainerClient(container_name)
    
    def get_container_client(self, container_name):
        print(f"[DUMMY] Getting container client: {container_name}")
        return DummyContainerClient(container_name)

class DummyContainerClient:
    """
    A dummy implementation of ContainerClient for testing without Azure
    """
    def __init__(self, container_name):
        self.container_name = container_name
    
    def delete_container(self):
        print(f"[DUMMY] Deleting container: {self.container_name}")
        return True
    
    def list_blobs(self, **kwargs):
        print(f"[DUMMY] Listing blobs with kwargs: {kwargs}")
        return []
    
    def get_blob_client(self, blob_name):
        print(f"[DUMMY] Getting blob client: {blob_name}")
        return DummyBlobClient(blob_name)

class DummyBlobClient:
    """
    A dummy implementation of BlobClient for testing without Azure
    """
    def __init__(self, blob_name):
        self.blob_name = blob_name
    
    def exists(self):
        print(f"[DUMMY] Checking if blob exists: {self.blob_name}")
        return False
    
    def delete_blob(self):
        print(f"[DUMMY] Deleting blob: {self.blob_name}")
        return True
    
    def upload_blob(self, data, **kwargs):
        print(f"[DUMMY] Uploading blob: {self.blob_name}")
        return {"request_id": "dummy_request_id"}

def createContainer(container_name):
    """
    Create a new Azure Blob Storage container
    
    Args:
        container_name (str): Name of the container to create
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.create_container(container_name.lower())
        print(f"Container '{container_name}' created successfully. Request ID: {container_client.container_name}")
        return True
    except Exception as error:
        print(f"Error creating container: {error}")
        return False

def delete_blob_storage_container(container_name):
    """
    Delete an Azure Blob Storage container
    
    Args:
        container_name (str): Name of the container to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(container_name.lower())
        container_client.delete_container()
        print(f"Container '{container_name}' deleted successfully")
        return True
    except Exception as error:
        print(f"Error deleting container: {error}")
        return False
    
def upload_to_azure_blob_storage(container_name, files):
    """
    Upload files to Azure Blob Storage container
    Files are uploaded twice: once to the container root and once to a 'new/' folder
    
    Args:
        container_name (str): Name of the container
        files (list): List of file objects from request.files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(container_name)

        # Clear the 'new/' folder first to prevent duplicates
        blobs_list = container_client.list_blobs(name_starts_with="new/")
        for blob in blobs_list:
            blob_client = container_client.get_blob_client(blob)
            blob_client.delete_blob()
            print(f"Deleted blob: {blob.name}")
        
        # Upload each file
        for file in files:
            # Upload to "new/" folder for vector processing
            blob_client_in_folder = container_client.get_blob_client(f"new/{file.filename}")
            file.seek(0)  # Reset file pointer to beginning
            upload_response1 = blob_client_in_folder.upload_blob(file, overwrite=True)
            print(f"File '{file.filename}' uploaded to 'new/' folder. Request ID: {upload_response1['request_id']}")
            
            # Upload directly to container root for direct access
            blob_client_direct = container_client.get_blob_client(f"{file.filename}")
            file.seek(0)  # Reset file pointer to beginning
            upload_response2 = blob_client_direct.upload_blob(file, overwrite=True)
            print(f"File '{file.filename}' uploaded to container root. Request ID: {upload_response2['request_id']}")
        
        return True
    except Exception as error:
        print(f"Error uploading files: {error}")
        return False
    
def delete_from_azure_blob_storage(container_name, blob_name):
    """
    Delete a blob from Azure Blob Storage container
    Deletes both from the container root and from the 'new/' folder
    
    Args:
        container_name (str): Name of the container
        blob_name (str): Name of the blob to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        blob_service_client = get_blob_service_client()
        # Get container client
        container_client = blob_service_client.get_container_client(container_name)

        # Delete from container root
        root_blob_client = container_client.get_blob_client(blob_name)
        if root_blob_client.exists():
            root_blob_client.delete_blob()
            print(f"File '{blob_name}' deleted from container root")
        
        # Delete from 'new/' folder
        new_folder_blob_name = f"new/{blob_name}"
        new_folder_blob_client = container_client.get_blob_client(new_folder_blob_name)
        if new_folder_blob_client.exists():
            new_folder_blob_client.delete_blob()
            print(f"File '{new_folder_blob_name}' deleted from 'new/' folder")
            
        return True
    except Exception as error:
        print(f"Error deleting file: {error}")
        return False

def generate_sas_token(container_name, blob_name):
    """
    Generate a Shared Access Signature (SAS) token for a blob
    
    Args:
        container_name (str): Name of the container
        blob_name (str): Name of the blob
        
    Returns:
        str: SAS token
    """
    try:
        # Get storage key from environment
        storage_account_key = os.environ.get('AZURE_STORAGE_KEY')
        if not storage_account_key:
            print("Warning: AZURE_STORAGE_KEY environment variable is not set")
            return "dummy_sas_token"
        
        blob_service_client = get_blob_service_client()
        # Generate SAS token with read permission for 1 hour
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=storage_account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        return sas_token
    except Exception as error:
        print(f"Error generating SAS token: {error}")
        return "dummy_sas_token"
