# LangChain RAG Implementation

This module provides the core Retrieval Augmented Generation (RAG) functionality for the Exam Preparation Chatbot using LangChain and LangGraph.

## Core Components

### RAG Chains

The implementation provides multiple types of RAG chains with increasing complexity:

1. **Basic RAG** (`create_basic_rag_chain`): Simple retrieval and generation without conversation history.
2. **Few-Shot RAG** (`create_few_shot_rag_chain`): Enhances the RAG chain with few-shot examples to improve response quality.
3. **Conversational RAG** (`create_conversational_rag_chain`): Maintains conversation history and understands context from previous exchanges.
4. **Agent-Based RAG** (`build_agent_graph`): Advanced workflow using LangGraph for multi-step reasoning and specialized handling of different question types.

### Key Features

- **Question Classification**: Automatically categorizes questions by subject and required solution approach.
- **Context-Aware Retrieval**: Considers conversation history to handle follow-up questions.
- **Step-by-Step Problem Solving**: Handles calculation questions with detailed work shown.
- **Smart Answer Formulation**: Tailors response format based on the question type and available information.

## Azure Integration

The implementation is designed to work with:

- Azure OpenAI for embeddings and LLM capabilities
- Azure Cognitive Search as the vector database
- Azure Blob Storage for document storage

## Using the Module

The main interface functions are:

```python
# Basic usage
answer = create_basic_rag_chain("collection-name").invoke("What is photosynthesis?")

# With conversation history
answer = create_conversational_rag_chain("collection-name").invoke({
    "question": "What does it produce?",
    "chat_history": [
        {"role": "human", "content": "What is photosynthesis?"},
        {"role": "ai", "content": "Photosynthesis is the process..."}
    ]
})

# Using the agent-based approach
agent_graph = build_agent_graph("collection-name")
answer, updated_history = chat_with_exam_bot(
    agent_graph, 
    "How do I calculate the derivative of sin(x)?", 
    chat_history
)
```

## Configuration

The module uses environment variables for Azure service configuration:

- `OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_ENDPOINT`: Azure OpenAI endpoint
- `AZURE_COGNITIVE_SEARCH_ENDPOINT`: Azure Cognitive Search endpoint
- `AZURE_COGNITIVE_SEARCH_API_KEY`: Azure Cognitive Search API key
