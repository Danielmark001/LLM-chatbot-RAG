
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from dotenv import load_dotenv
import traceback

# Import RAG functionality
try:
    from langchain_rag import (
        create_basic_rag_chain,
        create_few_shot_rag_chain,
        create_conversational_rag_chain,
        build_agent_graph,
        chat_with_exam_bot
    )
except Exception as e:
    print(f"Error importing langchain_rag: {e}")
    traceback.print_exc()

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Store chat sessions
chat_sessions = {}

@app.route("/test", methods=['GET'])
def test_endpoint():
    return jsonify({"message": "RAG test server is working!"}), 200

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
        print(f"Creating basic RAG chain for collection: {collection_name}")
        rag_chain = create_basic_rag_chain(collection_name)
        
        # Generate answer
        print(f"Invoking RAG chain with question: {question}")
        answer = rag_chain.invoke(question)
        
        return jsonify({
            "answer": answer,
            "collection": collection_name
        }), 200
        
    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in basic_rag_chat: {error_message}")
        print(traceback_str)
        return jsonify({
            "error": error_message,
            "traceback": traceback_str
        }), 500

if __name__ == "__main__":
    print("Starting RAG test server...")
    app.run(debug=True, port=5002)
