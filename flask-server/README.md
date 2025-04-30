# Document Management System Backend

This is the backend service for the Document Management System, a platform for organizing and managing course materials with embedded search capabilities.

## Features

- Document uploading and management
- Vector embeddings for semantic search
- Integration with Azure Blob Storage, MongoDB, and Azure Cognitive Search
- RESTful API for CRUD operations on collections and documents

## Setup Instructions

### Setting up the environment

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Create a `.env` file in the root directory with the following variables:

```
AZURE_CONN_STRING=your_azure_connection_string
AZURE_STORAGE_KEY=your_azure_storage_key
AZURE_COGNITIVE_SEARCH_ENDPOINT=your_cognitive_search_endpoint
AZURE_COGNITIVE_SEARCH_API_KEY=your_cognitive_search_api_key
AZURE_ENDPOINT=your_azure_openai_endpoint
OPENAI_API_KEY=your_openai_api_key
MONGO_URI=your_mongodb_connection_string
```

### Running the server

```bash
python server.py
```

The server will start on http://localhost:5000

## API Endpoints

### Collections
- `GET /api/collections` - Get all collections
- `PUT /api/createcollection` - Create a new collection
- `DELETE /api/dropcollection` - Delete a collection
- `DELETE /api/dropIndex` - Delete a search index

### Documents
- `GET /api/collections/<collection_name>` - Get all documents in a collection
- `PUT /api/<collection_name>/createdocument` - Upload documents to a collection
- `DELETE /api/<collection_name>/deletedocument` - Delete a document
- `DELETE /api/<collection_name>/deleteembeddings` - Delete document embeddings

### Vector Store
- `POST /vectorstore` - Store document embeddings
- `PUT /createindex` - Create a search index
