from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import CharacterTextSplitter
from mongo_helper import create_collection, create_document, drop_collection, delete_document, get_collections, get_documents
from blob_storage_helper import createContainer, delete_blob_storage_container, upload_to_azure_blob_storage, delete_from_azure_blob_storage, generate_sas_token
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, 
    SearchField, 
    SearchFieldDataType, 
    SimpleField, 
    SearchableField, 
    VectorSearch, 
    VectorSearchProfile, 
    HnswAlgorithmConfiguration
)
from dotenv import load_dotenv
import uuid
from pathlib import Path
from azure.storage.blob import BlobServiceClient

# Import RAG functionality
from langchain_rag import (
    create_basic_rag_chain,
    create_few_shot_rag_chain,
    create_conversational_rag_chain,
    build_agent_graph,
    chat_with_exam_bot
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Azure configuration
connection_string = os.environ.get('AZURE_CONN_STRING')

# Initialize Azure services
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="FYP-SCSE23-1127-text-embedding-ada-002", 
    api_key=os.environ.get('OPENAI_API_KEY'),
    azure_endpoint=os.environ.get('AZURE_ENDPOINT')
)
blob_service_client = BlobServiceClient.from_connection_string(os.environ.get('AZURE_CONN_STRING'))

# Store chat sessions
chat_sessions = {}

# Vector store operations
@app.route("/vectorstore", methods=['POST'])
def store_documents():
    """
    Process documents and store them in the vector store.
    Handles both adding new documents and updating existing ones.
    """
    data = request.json
    containername = data.get('containername')

    if not containername:
        return jsonify({"error": "Container name is required"}), 400
    
    # Initialize document tracking lists
    docs_to_add = []
    docs_to_update = []
    docs_to_update_id = []
    
    docs_to_add_final = []
    docs_to_add_page_content = []
    docs_to_add_embeddings = []
    docs_to_add_filename = []

    docs_to_update_final = []
    docs_to_update_page_content = []
    docs_to_update_embeddings = []
    docs_to_update_filename = []

    try:   
        # Initialize search client
        search_client = SearchClient(
            endpoint=os.environ.get('AZURE_COGNITIVE_SEARCH_ENDPOINT'),
            index_name=containername,
            credential=AzureKeyCredential(os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'))
        )

        # Load documents from blob storage
        loader = AzureBlobStorageContainerLoader(
            conn_str=os.environ.get('AZURE_CONN_STRING'),
            container=containername,
            prefix='new/'
        )
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = loader.load()

        # Determine which documents to add vs update
        for doc in documents:
            path_to_check = doc.metadata['source']
            filename_to_check = Path(path_to_check).name
            search_results = search_client.search(filter=f"filename eq '{filename_to_check}'")
            first_result = next(search_results, None)
            
            if first_result is not None:
                # Document exists - update it
                search_results = search_client.search(filter=f"filename eq '{filename_to_check}'")
                for result in search_results:
                    docs_to_update_id.append(result['id'])
                    
                split_docs_to_update = text_splitter.split_documents([doc])
                for sdoc in split_docs_to_update:
                    docs_to_update.append(sdoc)                    
            else:
                # New document - add it
                split_docs_to_add = text_splitter.split_documents([doc])
                for adoc in split_docs_to_add:
                    docs_to_add.append(adoc)    

        # Process documents to update
        for doc in docs_to_update:
            path_to_update = doc.metadata['source']
            filename_to_update = Path(path_to_update).name
            docs_to_update_filename.append(filename_to_update)
            docs_to_update_page_content.append(doc.page_content)
        
        # Process documents to add
        for doc in docs_to_add:
            path_to_add = doc.metadata['source']
            filename_to_add = Path(path_to_add).name
            docs_to_add_filename.append(filename_to_add)
            docs_to_add_page_content.append(doc.page_content)

        # Generate embeddings
        if docs_to_update_page_content:
            docs_to_update_embeddings = embeddings.embed_documents(docs_to_update_page_content)
        
        if docs_to_add_page_content:
            docs_to_add_embeddings = embeddings.embed_documents(docs_to_add_page_content)

        # Prepare documents for update
        if docs_to_update_page_content:
            for i in range(len(docs_to_update_page_content)):
                docs_to_update_final.append({
                    'id': docs_to_update_id[i],
                    'content': docs_to_update_page_content[i],
                    'content_vector': docs_to_update_embeddings[i],
                    'filename': docs_to_update_filename[i]
                })
            # Update documents in search index
            search_client.merge_documents(docs_to_update_final)

        # Prepare documents for addition
        if docs_to_add_page_content:
            for i in range(len(docs_to_add_page_content)):
                docs_to_add_final.append({
                    'id': str(uuid.uuid4()),
                    'content': docs_to_add_page_content[i],
                    'content_vector': docs_to_add_embeddings[i],
                    'filename': docs_to_add_filename[i]
                })
            # Add documents to search index
            search_client.upload_documents(docs_to_add_final)

        return jsonify({"message": "Data loaded into vectorstore successfully"}), 201

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/createindex", methods=['PUT'])
def create_index():
    """Create a search index for vector embeddings"""
    data = request.json
    collection_name = data.get('collectionName')
    
    # Initialize search index client
    client = SearchIndexClient(
        os.environ.get('AZURE_COGNITIVE_SEARCH_ENDPOINT'), 
        AzureKeyCredential(os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'))
    )
    
    # Define index fields
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            searchable=True,
            filterable=True,
            retrievable=True,
            stored=True,
            sortable=False,
            facetable=False
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=False,
            retrievable=True,
            stored=True,
            sortable=False,
            facetable=False
        ),
        SearchField(
            name="content_vector", 
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True, 
            vector_search_dimensions=1536, 
            vector_search_profile_name="my-vector-config"
        ),
        SearchableField(
            name="filename",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        )
    ]

    # Define vector search configuration
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
    )

    try:
        # Create or update the search index
        searchindex = SearchIndex(name=collection_name, fields=fields, vector_search=vector_search)
        client.create_or_update_index(index=searchindex)
        return jsonify({"message": "Index created successfully"}), 201

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# Collection management
@app.route('/api/createcollection', methods=['PUT'])
def create_course():
    """Create a new collection (course)"""
    data = request.json
    collection_name = data.get('collectionName')
    if not collection_name:
        return jsonify({"error": "Collection name is required"}), 400
    
    # Format collection name (lowercase with hyphens)
    collection_name = collection_name.lower().replace(' ', '-')
    
    try:
        # Create Azure blob storage container
        create_success_container = createContainer(collection_name)
        if create_success_container:
            # Create MongoDB collection
            create_success_collection = create_collection(collection_name)
            if create_success_collection:
                return jsonify({"message": "Collection created successfully!"}), 201
            else:
                return jsonify({'error': 'Failed to create collection'}), 500
        else:
            return jsonify({'error': 'Failed to create container'}), 500
    except Exception as error:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/collections', methods=['GET'])
def get_containers():
    """Get all collections"""
    collections = get_collections()
    return jsonify(collections), 201

@app.route('/api/collections/<collection_name>', methods=['GET'])
def get_files(collection_name):
    """Get all documents in a collection"""
    documents = get_documents(collection_name)
    return jsonify(documents), 201

# Document management
@app.route('/api/<collection_name>/createdocument', methods=['PUT'])
def upload_document(collection_name):
    """Upload documents to a collection"""
    files = request.files.getlist('files')
    container_name = collection_name.lower().replace(' ', '-')
    files_with_links = []
    
    try:
        # Upload files to Azure blob storage
        upload_success = upload_to_azure_blob_storage(container_name, files)
        if upload_success:
            # Create SAS tokens for each file
            for file in files:
                sas_token = generate_sas_token(collection_name, file.filename)
                blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{file.filename}?{sas_token}"
                files_with_links.append({
                    "name": file.filename,
                    "url": blob_url
                })
            
            # Create document metadata in MongoDB
            create_document_success = create_document(collection_name, files_with_links)
            if create_document_success:
                return jsonify({"message": "Documents created successfully!"}), 201
            else:
                return jsonify({'error': 'Failed to create documents'}), 500
        else:
            return jsonify({'error': 'Failed to upload files to Azure Blob Storage'}), 500
    except Exception as error:
        print(f"Error processing files upload: {error}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/dropcollection', methods=['DELETE'])
def delete_course():
    """Delete a collection (course)"""
    data = request.json
    collection_name = data.get('collectionName')
    if not collection_name:
        return jsonify({"error": "Collection name is required"}), 400
    
    # Format collection name (lowercase with hyphens)
    collection_name = collection_name.lower().replace(' ', '-')
    
    try:
        # Delete Azure blob storage container
        delete_success_container = delete_blob_storage_container(collection_name)
        if delete_success_container:
            # Delete MongoDB collection
            delete_success_collection = drop_collection(collection_name)
            if delete_success_collection:
                return jsonify({"message": "Collection deleted successfully!"}), 201
            else:
                return jsonify({'error': 'Failed to delete collection'}), 500
        else:
            return jsonify({'error': 'Failed to delete container'}), 500
    except Exception as error:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/<collection_name>/deletedocument', methods=['DELETE'])
def delete_file(collection_name):
    """Delete a document from a collection"""
    data = request.json
    file_name = data.get('fileName')
    file_id = data.get('_id')
    container_name = collection_name.lower().replace(' ', '-')
    
    try:
        # Delete file from Azure blob storage
        delete_success = delete_from_azure_blob_storage(container_name, file_name)
        if delete_success:
            # Delete document metadata from MongoDB
            delete_document_success = delete_document(collection_name, file_id)
            if delete_document_success:
                return jsonify({"message": "Document deleted successfully!"}), 201
            else:
                return jsonify({'error': 'Failed to delete document'}), 500
        else:
            return jsonify({'error': 'Failed to delete file from Azure Blob Storage'}), 500
    except Exception as error:
        print(f"Error processing file deletion: {error}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route("/api/<collection_name>/deleteembeddings", methods=['DELETE'])
def delete_embeddings(collection_name):
    """Delete document embeddings from the search index"""
    data = request.json
    blob_name = data.get('fileName')
    
    try:
        # Initialize search client
        search_client = SearchClient(
            os.environ.get('AZURE_COGNITIVE_SEARCH_ENDPOINT'), 
            collection_name, 
            AzureKeyCredential(os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'))
        )
        
        # Find documents to delete by filename
        search_result = search_client.search(filter=f"filename eq '{blob_name}'")
        ids_to_delete = []
        
        for result in search_result:
            ids_to_delete.append({'id': result['id']})
        
        # Delete documents from search index
        if ids_to_delete:
            search_client.delete_documents(ids_to_delete)

        return jsonify({"message": "Data deleted from vectorstore successfully"}), 201

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/dropIndex', methods=['DELETE'])
def delete_index():
    """Delete a search index"""
    data = request.json
    collection_name = data.get('collectionName')
    collection_name = collection_name.lower().replace(' ', '-')
    
    try:
        # Initialize search index client
        client = SearchIndexClient(
            os.environ.get('AZURE_COGNITIVE_SEARCH_ENDPOINT'), 
            AzureKeyCredential(os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'))
        )
        
        # Delete the search index
        client.delete_index(collection_name)
        return jsonify({"message": "Index deleted successfully"}), 201
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# NEW RAG ENDPOINTS

@app.route('/api/chat/basic/<collection_name>', methods=['POST'])
def basic_rag_chat(collection_name):
    """
    Basic RAG chat endpoint - no conversation history.
    """
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        # Format collection name (lowercase with hyphens)
        collection_name = collection_name.lower().replace(' ', '-')
        
        # Create basic RAG chain
        rag_chain = create_basic_rag_chain(collection_name)
        
        # Generate answer
        answer = rag_chain.invoke(question)
        
        return jsonify({
            "answer": answer,
            "collection": collection_name
        }), 200
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/few-shot/<collection_name>', methods=['POST'])
def few_shot_rag_chat(collection_name):
    """
    Few-shot RAG chat endpoint - no conversation history.
    """
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        # Format collection name (lowercase with hyphens)
        collection_name = collection_name.lower().replace(' ', '-')
        
        # Create few-shot RAG chain
        rag_chain = create_few_shot_rag_chain(collection_name)
        
        # Generate answer
        answer = rag_chain.invoke(question)
        
        return jsonify({
            "answer": answer,
            "collection": collection_name
        }), 200
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/conversational/<collection_name>', methods=['POST'])
def conversational_rag_chat(collection_name):
    """
    Conversational RAG chat endpoint - maintains conversation history.
    """
    data = request.json
    question = data.get('question')
    session_id = data.get('session_id')
    
    if not question or not session_id:
        return jsonify({"error": "Question and session_id are required"}), 400
    
    try:
        # Format collection name (lowercase with hyphens)
        collection_name = collection_name.lower().replace(' ', '-')
        
        # Get or create chat history for this session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "chat_history": [],
                "collection": collection_name
            }
        
        # Create conversational RAG chain
        rag_chain = create_conversational_rag_chain(collection_name)
        
        # Generate answer
        answer = rag_chain.invoke({
            "question": question,
            "chat_history": chat_sessions[session_id]["chat_history"]
        })
        
        # Update chat history
        chat_sessions[session_id]["chat_history"].append({"role": "human", "content": question})
        chat_sessions[session_id]["chat_history"].append({"role": "ai", "content": answer})
        
        return jsonify({
            "answer": answer,
            "collection": collection_name,
            "session_id": session_id
        }), 200
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/reset/<session_id>', methods=['POST'])
def reset_chat_session(session_id):
    """
    Reset a chat session.
    """
    if session_id in chat_sessions:
        collection = chat_sessions[session_id]["collection"]
        chat_sessions[session_id] = {
            "chat_history": [],
            "collection": collection
        }
        return jsonify({"message": f"Session {session_id} reset successfully"}), 200
    else:
        return jsonify({"error": "Session not found"}), 404

@app.route('/api/chat/agent/<collection_name>', methods=['POST'])
def agent_rag_chat(collection_name):
    """
    Agent-based RAG chat endpoint with advanced reasoning.
    """
    data = request.json
    question = data.get('question')
    session_id = data.get('session_id')
    
    if not question or not session_id:
        return jsonify({"error": "Question and session_id are required"}), 400
    
    try:
        # Format collection name (lowercase with hyphens)
        collection_name = collection_name.lower().replace(' ', '-')
        
        # Get or create chat history for this session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "chat_history": [],
                "collection": collection_name,
                "agent_graph": build_agent_graph(collection_name)
            }
        elif "agent_graph" not in chat_sessions[session_id]:
            chat_sessions[session_id]["agent_graph"] = build_agent_graph(collection_name)
        
        # Get chat history and agent graph
        chat_history = chat_sessions[session_id]["chat_history"]
        agent_graph = chat_sessions[session_id]["agent_graph"]
        
        # Generate answer
        answer, updated_history = chat_with_exam_bot(agent_graph, question, chat_history)
        
        # Update chat history
        chat_sessions[session_id]["chat_history"] = updated_history
        
        return jsonify({
            "answer": answer,
            "collection": collection_name,
            "session_id": session_id
        }), 200
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
