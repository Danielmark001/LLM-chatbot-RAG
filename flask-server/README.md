# Exam Preparation Chatbot Server

This is the backend server for the Exam Preparation Chatbot application. It handles document management, vector embeddings, and the RAG (Retrieval Augmented Generation) functionality.

## Architecture

The server is built with Flask and integrates several Azure services:

- **Azure Blob Storage**: Stores the PDF documents (past exam papers)
- **Azure Cognitive Search**: Provides vector search capabilities for RAG
- **Azure OpenAI**: Powers the embeddings and language model capabilities
- **MongoDB**: Stores metadata for document management

## Core Components

1. **Document Management**
   - Upload, store, and manage exam papers with metadata
   - Endpoints for CRUD operations on collections and documents

2. **Vector Store Operations**
   - Process documents into embeddings for semantic search
   - Create and manage search indices in Azure Cognitive Search

3. **RAG Implementation**
   - LangChain integration for advanced RAG capabilities
   - Multiple RAG modes (Basic, Few-Shot, Conversational, Agent-Based)
   - Session management for maintaining conversation history

## API Endpoints

### Collection Management

- `PUT /api/createcollection`: Create a new collection (course)
- `GET /api/collections`: Get all collections
- `GET /api/collections/<collection_name>`: Get all documents in a collection
- `DELETE /api/dropcollection`: Delete a collection

### Document Management

- `PUT /api/<collection_name>/createdocument`: Upload documents to a collection
- `DELETE /api/<collection_name>/deletedocument`: Delete a document from a collection

### Vector Store Operations

- `PUT /createindex`: Create a search index for vector embeddings
- `POST /vectorstore`: Process documents and store them in the vector store
- `DELETE /api/<collection_name>/deleteembeddings`: Delete document embeddings from the search index
- `DELETE /api/dropIndex`: Delete a search index

### Chat Endpoints

- `POST /api/chat/basic/<collection_name>`: Basic RAG chat endpoint (no conversation history)
- `POST /api/chat/few-shot/<collection_name>`: Few-shot RAG chat endpoint (no conversation history)
- `POST /api/chat/conversational/<collection_name>`: Conversational RAG chat endpoint (maintains conversation history)
- `POST /api/chat/agent/<collection_name>`: Agent-based RAG chat endpoint with advanced reasoning
- `POST /api/chat/reset/<session_id>`: Reset a chat session

## Setup and Configuration

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the required environment variables:
```
OPENAI_API_KEY=your_openai_api_key
AZURE_ENDPOINT=your_azure_endpoint
AZURE_CONN_STRING=your_azure_blob_connection_string
AZURE_STORAGE_KEY=your_azure_storage_key
AZURE_COGNITIVE_SEARCH_ENDPOINT=your_cognitive_search_endpoint
AZURE_COGNITIVE_SEARCH_API_KEY=your_cognitive_search_api_key
MONGO_URI=your_mongodb_connection_string
```

3. Start the server:
```bash
python server.py
```

## LangChain Integration

The server integrates LangChain for advanced RAG capabilities. The implementation is located in the `langchain_rag/` directory and includes:

- Basic RAG chains
- Few-shot prompting
- Conversational memory
- LangGraph for agentic workflows

See the `langchain_rag/README.md` file for more details on the RAG implementation.
