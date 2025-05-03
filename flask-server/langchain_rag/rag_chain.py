"""
LangChain RAG implementation for the Exam Preparation Chatbot.
This module provides the core functionality for retrieval-augmented generation
using Azure Cognitive Search as the vector store and Azure OpenAI for LLM capabilities.
"""

import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser

# LangChain Azure OpenAI Integration
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# LangChain Community integrations
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.chat_message_histories import ChatMessageHistory

# LangGraph for agentic workflows
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Embeddings configuration - with fallback to OpenAI if Azure fails
try:
    # Try Azure OpenAI first
    EMBEDDINGS = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",  # Changed to match standard deployment name
        api_key=os.environ.get('OPENAI_API_KEY'),
        azure_endpoint=os.environ.get('AZURE_ENDPOINT')
    )
    print("Using Azure OpenAI Embeddings")
except Exception as e:
    print(f"Azure OpenAI embeddings failed: {e}")
    # Fall back to regular OpenAI
    from langchain_openai import OpenAIEmbeddings
    EMBEDDINGS = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    print("Falling back to standard OpenAI Embeddings")

# LLM Configuration - with fallback to OpenAI if Azure fails
try:
    # Try Azure OpenAI first
    LLM = AzureChatOpenAI(
        azure_deployment="gpt-4",  # Changed to match standard deployment name
        api_version="2023-05-15",
        azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    print("Using Azure OpenAI Chat")
except Exception as e:
    print(f"Azure OpenAI chat failed: {e}")
    # Fall back to regular OpenAI
    from langchain_openai import ChatOpenAI
    LLM = ChatOpenAI(
        model="gpt-4-turbo",
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    print("Falling back to standard OpenAI Chat")

# Vector store configuration with simple fallback
def get_azure_search_store(index_name: str) -> Any:
    """
    Initialize and return a vector store with Azure Cognitive Search or fall back to a simple in-memory store.
    
    Args:
        index_name (str): The name of the vector store index
        
    Returns:
        Any: The initialized vector store
    """
    
    try:
        # Try Azure Cognitive Search first
        vector_store = AzureSearch(
            azure_search_endpoint=os.environ.get('AZURE_COGNITIVE_SEARCH_ENDPOINT'),
            azure_search_key=os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'),
            index_name=index_name,
            embedding_function=EMBEDDINGS.embed_query,
            vector_field_name="content_vector",
            content_field_name="content",
            metadata_field_name="metadata"
        )
        print(f"Using Azure Cognitive Search for index: {index_name}")
        
    except Exception as e:
        print(f"Azure Cognitive Search failed: {e}")
        print("Falling back to simple in-memory store")
        
        # Create a simple document for testing
        from langchain.schema import Document
        from langchain_community.vectorstores import FAISS
        
        # Create test documents
        docs = [
            Document(
                page_content="This is a test document for the exam preparation chatbot.",
                metadata={"source": "test", "filename": "test.txt"}
            ),
            Document(
                page_content="Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.",
                metadata={"source": "test", "filename": "biology.txt"}
            ),
            Document(
                page_content="The derivative of a function represents its rate of change at any given point.",
                metadata={"source": "test", "filename": "math.txt"}
            )
        ]
        
        try:
            # Try FAISS (doesn't require additional dependencies)
            vector_store = FAISS.from_documents(docs, EMBEDDINGS)
            print("Using FAISS vector store")
        except Exception as e:
            print(f"FAISS failed: {e}")
            
            # Last resort - create a simple memory store
            from langchain_community.vectorstores.memory import MemoryVectorStore
            vector_store = MemoryVectorStore(embedding_function=EMBEDDINGS)
            vector_store.add_documents(docs)
            print("Using in-memory vector store")
        
    return vector_store

# Expert tutor system prompt
SYSTEM_PROMPT = """You are an expert exam tutor helping students prepare for exams. 
Use the following retrieved content from past exam papers to provide a helpful and accurate response to the student's question.

Retrieved content:
{context}

Instructions:
- Answer the question based on the retrieved content
- If the retrieved content doesn't contain the answer, say so honestly
- Provide step-by-step explanations when appropriate
- Use mathematical notation properly when needed
- Cite the specific paper you're referencing (e.g., filename)
- Maintain a supportive, educational tone
- If the question requires calculations, show your work completely
- Relate your answer to similar exam questions when possible

Remember to tailor your approach to the subject area. For math, science, and technical subjects, focus on step-by-step problem-solving. For humanities and social sciences, focus on key concepts, arguments, and examples.
"""

# Few-shot examples for better model performance
FEW_SHOT_EXAMPLES = [
    {"question": "What's the derivative of f(x) = x^3 + 2x^2 - 5x + 3?",
     "answer": """To find the derivative of f(x) = x^3 + 2x^2 - 5x + 3, I'll apply the power rule and the sum rule of differentiation.

The power rule states that the derivative of x^n is n*x^(n-1).

For f(x) = x^3 + 2x^2 - 5x + 3:

- The derivative of x^3 is 3x^2
- The derivative of 2x^2 is 2(2x^1) = 4x
- The derivative of -5x is -5
- The derivative of the constant 3 is 0

Combining these terms: f'(x) = 3x^2 + 4x - 5

This matches the solution from the MATH_2019_MIDTERM paper."""},
    
    {"question": "What are the key differences between DNA and RNA?",
     "answer": """Based on the past biology exam papers, there are several key structural and functional differences between DNA and RNA:

**Structural Differences:**
1. **Sugar component**: DNA contains deoxyribose, while RNA contains ribose (which has an additional OH group on the 2' carbon)
2. **Nucleotide bases**: DNA uses Adenine, Thymine, Cytosine, and Guanine; RNA uses Adenine, Uracil (instead of Thymine), Cytosine, and Guanine
3. **Structure**: DNA is typically double-stranded in a double helix; RNA is usually single-stranded
4. **Stability**: DNA is more stable due to its deoxyribose sugar and double-stranded nature; RNA is less stable

**Functional Differences:**
1. **Location**: DNA is primarily found in the nucleus (with some in mitochondria and chloroplasts); RNA exists in both the nucleus and cytoplasm
2. **Role**: DNA stores genetic information long-term; RNA has multiple roles including protein synthesis (mRNA, tRNA, rRNA) and regulatory functions
3. **Replication**: DNA can self-replicate; RNA is synthesized from DNA through transcription

This information appears in both the BIOLOGY_2021_MIDTERM and BIOLOGY_2022_FINAL papers."""}
]

def format_docs(docs: List[Document]) -> str:
    """
    Format a list of documents into a string for the prompt.
    
    Args:
        docs (List[Document]): List of retrieved documents
        
    Returns:
        str: Formatted string of document content
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        filename = doc.metadata.get("filename", "Unknown Source")
        content = doc.page_content
        formatted_docs.append(f"Document {i+1} (Source: {filename}):\n{content}\n")
    
    return "\n".join(formatted_docs)

def create_basic_rag_chain(index_name: str) -> Any:
    """
    Create a basic RAG chain for answering exam questions.
    
    Args:
        index_name (str): The name of the Azure Cognitive Search index
        
    Returns:
        Any: The RAG chain
    """
    # Initialize vector store and retriever
    vector_store = get_azure_search_store(index_name)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | LLM
        | StrOutputParser()
    )
    
    return rag_chain

def create_few_shot_rag_chain(index_name: str) -> Any:
    """
    Create a RAG chain with few-shot examples for better performance.
    
    Args:
        index_name (str): The name of the Azure Cognitive Search index
        
    Returns:
        Any: The few-shot RAG chain
    """
    # Initialize vector store and retriever
    vector_store = get_azure_search_store(index_name)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create messages for few-shot examples
    few_shot_messages = []
    for example in FEW_SHOT_EXAMPLES:
        few_shot_messages.append(HumanMessage(content=example["question"]))
        few_shot_messages.append(AIMessage(content=example["answer"]))
    
    # Create the prompt template with few-shot examples
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        *[(msg.type, msg.content) for msg in few_shot_messages],
        ("human", "{question}")
    ])
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | LLM
        | StrOutputParser()
    )
    
    return rag_chain

def create_conversational_rag_chain(index_name: str) -> Any:
    """
    Create a conversational RAG chain that maintains conversation history.
    
    Args:
        index_name (str): The name of the Azure Cognitive Search index
        
    Returns:
        Any: The conversational RAG chain
    """
    # Initialize vector store and retriever
    vector_store = get_azure_search_store(index_name)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Set up the contextualization prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a conversation history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        that can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed, and otherwise return it as is."""),
        ("human", "{chat_history}\n\nLatest user question: {question}"),
    ])
    
    # Function to format chat history
    def format_chat_history(chat_history):
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    
    # Contextualizer function to reformulate the question based on chat history
    def contextualize_question(inputs):
        if not inputs.get("chat_history"):
            return inputs["question"]
        
        formatted_history = format_chat_history(inputs["chat_history"])
        contextualized_question = contextualize_q_prompt.invoke({
            "chat_history": formatted_history, 
            "question": inputs["question"]
        })
        return contextualized_question.content
    
    # Create the conversational prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # Create the conversational chain
    conversational_rag_chain = RunnablePassthrough.assign(
        context=RunnableLambda(contextualize_question) | retriever | format_docs,
        chat_history=lambda x: x.get("chat_history", [])
    ) | prompt | LLM | StrOutputParser()
    
    return conversational_rag_chain

# Advanced agent state definition
class AgentState(dict):
    """State tracked across agent steps."""
    question: str
    chat_history: List[Dict[str, str]]
    retrieved_documents: Optional[List[Document]] = None
    need_more_info: bool = False
    need_calculation: bool = False
    subject_area: Optional[str] = None
    intermediate_work: Optional[str] = None
    final_answer: Optional[str] = None

def classify_question(state: AgentState):
    """Classify the question to determine the appropriate solution path."""
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the student's question and classify it according to:
        1. Subject area (math, physics, biology, chemistry, economics, etc.)
        2. Whether it requires retrieval of specific exam content
        3. Whether it requires calculation or step-by-step problem solving
        
        Respond with a JSON object with the following structure:
        {
            "subject_area": "[subject]",
            "need_retrieval": true/false,
            "need_calculation": true/false
        }
        """),
        ("human", "{question}")
    ])
    
    # Run the classification
    classification_chain = classification_prompt | LLM | StrOutputParser() | (lambda x: eval(x.replace('true', 'True').replace('false', 'False')))
    result = classification_chain.invoke({"question": state["question"]})
    
    # Update the state
    state["subject_area"] = result["subject_area"]
    state["need_more_info"] = result["need_retrieval"]
    state["need_calculation"] = result["need_calculation"]
    
    return state

def retrieve_information(state: AgentState, index_name: str):
    """Retrieve relevant documents from the vector store."""
    if state["need_more_info"]:
        # Initialize vector store and retriever
        vector_store = get_azure_search_store(index_name)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Use the contextual retriever if chat history exists
        if state.get("chat_history", []):
            # Set up the contextualization prompt
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", """Given a conversation history and the latest user question 
                which might reference context in the chat history, formulate a standalone question 
                that can be understood without the chat history."""),
                ("human", "{chat_history}\n\nLatest user question: {question}"),
            ])
            
            # Format chat history
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in state["chat_history"]])
            
            # Get contextualized question
            contextualized_q = contextualize_q_prompt.invoke({
                "chat_history": formatted_history, 
                "question": state["question"]
            }).content
            
            # Retrieve documents using contextualized question
            retrieved_docs = retriever.invoke(contextualized_q)
        else:
            # Use the base retriever if no chat history
            retrieved_docs = retriever.invoke(state["question"])
        
        state["retrieved_documents"] = retrieved_docs
    
    return state

def solve_problem(state: AgentState):
    """Perform calculations or step-by-step problem solving if needed."""
    if state["need_calculation"]:
        solve_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in {subject_area}. Work through this problem step by step,
            showing all your work. If you need to perform calculations, do them carefully.
            
            {retrieved_context}
            """),
            ("human", "{question}")
        ])
        
        # Prepare context from retrieved documents if available
        retrieved_context = ""
        if state.get("retrieved_documents"):
            retrieved_context = "Retrieved information:\n" + format_docs(state["retrieved_documents"])
        else:
            retrieved_context = "No specific exam content retrieved. Solving based on general knowledge."
        
        # Run the solver
        solve_chain = solve_prompt | LLM | StrOutputParser()
        work = solve_chain.invoke({
            "question": state["question"],
            "subject_area": state["subject_area"],
            "retrieved_context": retrieved_context
        })
        
        state["intermediate_work"] = work
    
    return state

def formulate_answer(state: AgentState):
    """Generate the final answer based on the accumulated information."""
    # Prepare the prompt template based on the available information
    if state["need_calculation"] and state["need_more_info"]:
        # Complex question requiring both retrieval and calculation
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert exam tutor. Provide a comprehensive answer to the student's question.
            Use the retrieved past exam content and the step-by-step work to create a thorough explanation.
            
            Retrieved content:
            {retrieved_content}
            
            Step-by-step work:
            {intermediate_work}
            
            Instructions:
            - Cite the specific papers you're referencing
            - Ensure your explanation is clear and educational
            - Use proper formatting for mathematical notation if needed
            - Relate your answer to similar exam questions when possible"""),
            ("human", "{question}")
        ])
    elif state["need_more_info"]:
        # Question requiring mainly retrieval
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
    elif state["need_calculation"]:
        # Question requiring mainly calculation
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert exam tutor in {subject_area}. Provide a comprehensive answer 
            to the student's question based on the step-by-step work. Make sure your explanation is 
            clear and educational, with proper mathematical notation if needed.
            
            Step-by-step work:
            {intermediate_work}"""),
            ("human", "{question}")
        ])
    else:
        # General question not requiring special handling
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert exam tutor in {subject_area}. Provide a comprehensive answer 
            to the student's question based on your knowledge. Make sure your explanation is 
            clear and educational."""),
            ("human", "{question}")
        ])
    
    # Prepare the input for the prompt
    prompt_input = {
        "question": state["question"],
        "subject_area": state["subject_area"]
    }
    
    # Add retrieved content if available
    if state.get("retrieved_documents"):
        retrieved_content = format_docs(state["retrieved_documents"])
        prompt_input["context"] = retrieved_content
        prompt_input["retrieved_content"] = retrieved_content
    
    # Add intermediate work if available
    if state.get("intermediate_work"):
        prompt_input["intermediate_work"] = state["intermediate_work"]
    
    # Run the answer generation
    answer_chain = answer_prompt | LLM | StrOutputParser()
    answer = answer_chain.invoke(prompt_input)
    
    state["final_answer"] = answer
    return state

def router(state: AgentState):
    """Determine the next node in the workflow based on the current state."""
    if state.get("final_answer") is not None:
        # If we have a final answer, we're done
        return END
    
    if state.get("subject_area") is None:
        # If we haven't classified the question yet, do that first
        return "classify"
    
    if state.get("need_more_info") and state.get("retrieved_documents") is None:
        # If we need to retrieve documents but haven't yet, do that next
        return "retrieve"
    
    if state.get("need_calculation") and state.get("intermediate_work") is None:
        # If we need to solve a problem but haven't yet, do that next
        return "solve"
    
    # Otherwise, formulate the answer
    return "answer"

def build_agent_graph(index_name: str):
    """Build the complete agent workflow graph."""
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes to the graph
    workflow.add_node("classify", classify_question)
    workflow.add_node("retrieve", lambda state: retrieve_information(state, index_name))
    workflow.add_node("solve", solve_problem)
    workflow.add_node("answer", formulate_answer)
    
    # Add edges based on the router logic
    workflow.add_conditional_edges("classify", router)
    workflow.add_conditional_edges("retrieve", router)
    workflow.add_conditional_edges("solve", router)
    workflow.add_conditional_edges("answer", router)
    
    # Set the entry point
    workflow.set_entry_point("classify")
    
    # Compile the graph into a runnable
    return workflow.compile()

def chat_with_exam_bot(agent_graph, question, chat_history=None):
    """
    Interact with the exam preparation chatbot.
    
    Args:
        agent_graph: The compiled agent graph
        question (str): The user's question
        chat_history (List[Dict]): Optional chat history
        
    Returns:
        Tuple[str, List[Dict]]: The answer and updated chat history
    """
    # Initialize chat history if not provided
    if chat_history is None:
        chat_history = []
    
    # Run the agent
    result = agent_graph.invoke({
        "question": question,
        "chat_history": chat_history
    })
    
    # Update chat history
    chat_history.append({"role": "human", "content": question})
    chat_history.append({"role": "ai", "content": result["final_answer"]})
    
    return result["final_answer"], chat_history
