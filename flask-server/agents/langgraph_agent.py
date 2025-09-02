"""
Advanced LangGraph Agent for Exam Preparation Chatbot
Implements sophisticated multi-step reasoning workflows with hybrid retrieval
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import json
from datetime import datetime

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser

# Import our custom components
from retrieval.hybrid_retrieval import HybridRetriever, RetrievalResult
from config.pinecone_config import get_pinecone_manager
from ingestion.document_processor import PineconeDocumentStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State structure for the LangGraph agent"""
    # Input
    question: str
    chat_history: List[Dict[str, str]]
    user_preferences: Optional[Dict[str, Any]]
    
    # Analysis
    question_type: Optional[str]  # factual, calculation, explanation, comparison
    subject_area: Optional[str]
    difficulty_level: Optional[str]  # beginner, intermediate, advanced
    requires_calculation: bool
    requires_retrieval: bool
    requires_step_by_step: bool
    
    # Retrieval
    compressed_query: Optional[str]
    retrieved_documents: List[RetrievalResult]
    retrieval_stats: Optional[Dict[str, Any]]
    
    # Reasoning
    reasoning_steps: List[str]
    intermediate_calculations: List[str]
    solution_approach: Optional[str]
    
    # Output
    final_answer: str
    confidence_score: float
    sources_cited: List[str]
    follow_up_suggestions: List[str]
    
    # Metadata
    processing_time: float
    tokens_used: int

@dataclass
class ExamBotConfig:
    """Configuration for the exam preparation bot"""
    model_name: str = "gpt-4-turbo"
    temperature: float = 0.1
    max_tokens: int = 2048
    enable_hybrid_retrieval: bool = True
    retrieval_top_k: int = 10
    confidence_threshold: float = 0.7
    enable_citations: bool = True
    enable_follow_ups: bool = True

class QuestionAnalyzer:
    """Analyzes questions to determine processing approach"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def analyze_question(self, state: AgentState) -> AgentState:
        """Analyze the user's question to determine processing strategy"""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational assistant. Analyze the student's question and provide a JSON response with the following analysis:

{{
    "question_type": "factual|calculation|explanation|comparison|application",
    "subject_area": "math|physics|chemistry|biology|economics|computer_science|other",
    "difficulty_level": "beginner|intermediate|advanced",
    "requires_calculation": true|false,
    "requires_retrieval": true|false,
    "requires_step_by_step": true|false,
    "key_concepts": ["concept1", "concept2"],
    "solution_approach": "brief description of how to approach this question"
}}

Examples:
- "What is the derivative of x^2?" → calculation=true, retrieval=false, step_by_step=true
- "Explain photosynthesis" → retrieval=true, calculation=false, step_by_step=false
- "Compare mitosis and meiosis" → retrieval=true, calculation=false, step_by_step=false"""),
            ("human", "Analyze this question: {question}")
        ])
        
        try:
            analysis_chain = analysis_prompt | self.llm | StrOutputParser()
            result = analysis_chain.invoke({"question": state["question"]})
            
            # Parse JSON response
            analysis = json.loads(result.strip())
            
            # Update state
            state.update({
                "question_type": analysis.get("question_type", "factual"),
                "subject_area": analysis.get("subject_area", "other"),
                "difficulty_level": analysis.get("difficulty_level", "intermediate"),
                "requires_calculation": analysis.get("requires_calculation", False),
                "requires_retrieval": analysis.get("requires_retrieval", True),
                "requires_step_by_step": analysis.get("requires_step_by_step", False),
                "solution_approach": analysis.get("solution_approach", "")
            })
            
            logger.info(f"Question analysis: {analysis}")
            
        except Exception as e:
            logger.error(f"Question analysis failed: {e}")
            # Fallback analysis
            state.update({
                "question_type": "factual",
                "subject_area": "other",
                "difficulty_level": "intermediate",
                "requires_calculation": False,
                "requires_retrieval": True,
                "requires_step_by_step": False
            })
        
        return state

class QueryProcessor:
    """Processes and compresses queries for optimal retrieval"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def process_query(self, state: AgentState) -> AgentState:
        """Process and potentially compress the query for retrieval"""
        
        if not state["requires_retrieval"]:
            state["compressed_query"] = state["question"]
            return state
        
        # Create query compression prompt
        compression_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at optimizing search queries for academic content retrieval.
            
Given a student's question and chat history, create an optimized search query that will retrieve the most relevant exam papers and educational content.

Rules:
1. Extract key concepts and terms
2. Remove unnecessary words (the, a, an, etc.)
3. Add relevant academic synonyms
4. Consider the subject area context
5. Keep it concise but comprehensive

Chat History Context: {chat_history}
Subject Area: {subject_area}"""),
            ("human", "Original question: {question}\\n\\nCreate an optimized search query:")
        ])
        
        try:
            # Format chat history
            chat_context = "\\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in state.get("chat_history", [])[-3:]  # Last 3 exchanges
            ])
            
            compression_chain = compression_prompt | self.llm | StrOutputParser()
            compressed_query = compression_chain.invoke({
                "question": state["question"],
                "chat_history": chat_context,
                "subject_area": state.get("subject_area", "general")
            })
            
            state["compressed_query"] = compressed_query.strip()
            logger.info(f"Query compressed: '{state['question']}' -> '{state['compressed_query']}'")
            
        except Exception as e:
            logger.error(f"Query compression failed: {e}")
            state["compressed_query"] = state["question"]
        
        return state

class HybridRetriever:
    """Enhanced hybrid retriever with Pinecone integration"""
    
    def __init__(self):
        self.document_store = PineconeDocumentStore()
        
    def retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents using hybrid approach"""
        
        if not state["requires_retrieval"]:
            state["retrieved_documents"] = []
            state["retrieval_stats"] = {}
            return state
        
        try:
            query = state.get("compressed_query", state["question"])
            top_k = state.get("user_preferences", {}).get("max_results", 10)
            
            # Create filter based on subject area
            filter_dict = None
            if state.get("subject_area") and state["subject_area"] != "other":
                filter_dict = {"subject": state["subject_area"]}
            
            # Search in Pinecone
            search_results = self.document_store.search_documents(
                query=query,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            # Convert to RetrievalResult format
            retrieved_docs = []
            for i, result in enumerate(search_results):
                doc = Document(
                    page_content=result.get("text", ""),
                    metadata=result.get("metadata", {})
                )
                
                retrieval_result = RetrievalResult(
                    document=doc,
                    score=result.get("score", 0.0),
                    retrieval_type="hybrid",
                    rank=i
                )
                retrieved_docs.append(retrieval_result)
            
            state["retrieved_documents"] = retrieved_docs
            state["retrieval_stats"] = {
                "total_results": len(retrieved_docs),
                "avg_score": sum(r.score for r in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                "max_score": max((r.score for r in retrieved_docs), default=0),
                "search_query": query
            }
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents with avg score {state['retrieval_stats']['avg_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            state["retrieved_documents"] = []
            state["retrieval_stats"] = {"error": str(e)}
        
        return state

class MathSolver:
    """Specialized component for mathematical problem solving"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def solve_mathematical_problem(self, state: AgentState) -> AgentState:
        """Solve mathematical problems with step-by-step reasoning"""
        
        if not state["requires_calculation"]:
            return state
        
        # Create context from retrieved documents
        context = ""
        if state["retrieved_documents"]:
            relevant_docs = state["retrieved_documents"][:3]  # Top 3 most relevant
            context = "\\n\\n".join([
                f"Reference {i+1} (Score: {doc.score:.3f}):\\n{doc.document.page_content[:500]}..."
                for i, doc in enumerate(relevant_docs)
            ])
        
        solving_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert mathematics tutor. Solve the given problem step by step, showing all work clearly.

Retrieved Context (if available):
{context}

Instructions:
1. Break down the problem into clear steps
2. Show all calculations
3. Explain the reasoning behind each step
4. Provide the final answer
5. If using formulas or theorems, cite them
6. Use proper mathematical notation

Format your response as:
STEP 1: [Description]
[Work/Calculation]

STEP 2: [Description]
[Work/Calculation]

...

FINAL ANSWER: [Clear final answer]"""),
            ("human", "Solve this problem: {question}")
        ])
        
        try:
            solving_chain = solving_prompt | self.llm | StrOutputParser()
            solution = solving_chain.invoke({
                "question": state["question"],
                "context": context
            })
            
            # Extract steps and intermediate calculations
            steps = []
            calculations = []
            
            lines = solution.split('\\n')
            current_step = ""
            
            for line in lines:
                if line.strip().startswith("STEP"):
                    if current_step:
                        steps.append(current_step.strip())
                    current_step = line.strip()
                elif line.strip().startswith("FINAL ANSWER"):
                    if current_step:
                        steps.append(current_step.strip())
                    # Extract final answer
                    final_part = line.replace("FINAL ANSWER:", "").strip()
                    if final_part:
                        calculations.append(f"Final Answer: {final_part}")
                elif current_step:
                    current_step += "\\n" + line
            
            if current_step and not any(step in current_step for step in steps):
                steps.append(current_step.strip())
            
            state["reasoning_steps"] = steps
            state["intermediate_calculations"] = calculations
            
            logger.info(f"Mathematical problem solved with {len(steps)} steps")
            
        except Exception as e:
            logger.error(f"Mathematical solving failed: {e}")
            state["reasoning_steps"] = ["Error in mathematical solving"]
            state["intermediate_calculations"] = []
        
        return state

class AnswerSynthesizer:
    """Synthesizes final answers from all available information"""
    
    def __init__(self, llm, config: ExamBotConfig):
        self.llm = llm
        self.config = config
        
    def synthesize_answer(self, state: AgentState) -> AgentState:
        """Create the final comprehensive answer"""
        
        # Prepare context sections
        retrieved_context = ""
        if state["retrieved_documents"]:
            retrieved_context = "\\n\\n".join([
                f"Source {i+1} ({result.document.metadata.get('filename', 'Unknown')}):\\n{result.document.page_content[:800]}..."
                for i, result in enumerate(state["retrieved_documents"][:5])
            ])
        
        reasoning_context = ""
        if state.get("reasoning_steps"):
            reasoning_context = "\\n".join(state["reasoning_steps"])
        
        calculation_context = ""
        if state.get("intermediate_calculations"):
            calculation_context = "\\n".join(state["intermediate_calculations"])
        
        # Create synthesis prompt based on question type
        if state["question_type"] == "calculation" and state["requires_calculation"]:
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert exam tutor providing comprehensive answers to student questions.

Retrieved Academic Content:
{retrieved_context}

Step-by-Step Solution:
{reasoning_context}

Calculations:
{calculation_context}

Provide a comprehensive answer that:
1. Directly answers the student's question
2. Shows clear step-by-step work for calculations
3. Explains key concepts and reasoning
4. Cites relevant sources when available
5. Uses proper academic formatting
6. Suggests related topics for further study

Question Type: {question_type}
Subject Area: {subject_area}
Difficulty Level: {difficulty_level}"""),
                ("human", "Question: {question}")
            ])
        else:
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert exam tutor providing comprehensive answers to student questions.

Retrieved Academic Content:
{retrieved_context}

Additional Context:
{reasoning_context}

Provide a comprehensive answer that:
1. Directly answers the student's question
2. Explains key concepts clearly
3. Provides examples when helpful
4. Cites relevant sources from past exam papers
5. Uses proper academic formatting
6. Adapts complexity to the difficulty level: {difficulty_level}
7. Suggests related topics for further study

Question Type: {question_type}
Subject Area: {subject_area}"""),
                ("human", "Question: {question}")
            ])
        
        try:
            synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
            final_answer = synthesis_chain.invoke({
                "question": state["question"],
                "retrieved_context": retrieved_context,
                "reasoning_context": reasoning_context,
                "calculation_context": calculation_context,
                "question_type": state.get("question_type", "factual"),
                "subject_area": state.get("subject_area", "general"),
                "difficulty_level": state.get("difficulty_level", "intermediate")
            })
            
            state["final_answer"] = final_answer
            
            # Extract sources cited
            sources = []
            for result in state["retrieved_documents"][:5]:
                filename = result.document.metadata.get("filename", "Unknown")
                if filename not in sources:
                    sources.append(filename)
            
            state["sources_cited"] = sources
            
            # Calculate confidence based on retrieval scores and question complexity
            if state["retrieved_documents"]:
                avg_retrieval_score = sum(r.score for r in state["retrieved_documents"][:3]) / min(3, len(state["retrieved_documents"]))
                confidence = min(0.95, avg_retrieval_score + 0.1)  # Add small boost, cap at 95%
            else:
                confidence = 0.6  # Lower confidence without retrieval
            
            # Adjust based on question complexity
            if state["requires_calculation"] and state.get("reasoning_steps"):
                confidence += 0.1
            
            state["confidence_score"] = max(0.3, min(0.95, confidence))
            
            # Generate follow-up suggestions
            if self.config.enable_follow_ups:
                state["follow_up_suggestions"] = self._generate_follow_ups(state)
            
            logger.info(f"Final answer synthesized with confidence {state['confidence_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            state["final_answer"] = "I apologize, but I encountered an error while processing your question. Please try rephrasing your question."
            state["confidence_score"] = 0.2
            state["sources_cited"] = []
            state["follow_up_suggestions"] = []
        
        return state
    
    def _generate_follow_ups(self, state: AgentState) -> List[str]:
        """Generate relevant follow-up questions"""
        try:
            followup_prompt = ChatPromptTemplate.from_messages([
                ("system", """Based on the student's question and subject area, suggest 2-3 relevant follow-up questions that would help deepen their understanding.

Subject: {subject_area}
Question Type: {question_type}
Original Question: {question}

Provide follow-up questions that:
1. Build on the current topic
2. Explore related concepts
3. Apply the knowledge in different contexts
4. Are appropriate for the difficulty level

Format as a simple list, one question per line."""),
                ("human", "Generate follow-up questions:")
            ])
            
            followup_chain = followup_prompt | self.llm | StrOutputParser()
            result = followup_chain.invoke({
                "subject_area": state.get("subject_area", "general"),
                "question_type": state.get("question_type", "factual"),
                "question": state["question"]
            })
            
            # Parse the result into a list
            follow_ups = [line.strip() for line in result.split('\\n') if line.strip() and not line.strip().startswith('-')]
            return follow_ups[:3]  # Limit to 3 suggestions
            
        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            return []

class ExamPrepAgent:
    """Main LangGraph agent for exam preparation chatbot"""
    
    def __init__(self, config: ExamBotConfig = None):
        """
        Initialize the exam preparation agent
        
        Args:
            config: Configuration for the agent
        """
        self.config = config or ExamBotConfig()
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Initialize components
        self.question_analyzer = QuestionAnalyzer(self.llm)
        self.query_processor = QueryProcessor(self.llm)
        self.retriever = HybridRetriever()
        self.math_solver = MathSolver(self.llm)
        self.answer_synthesizer = AnswerSynthesizer(self.llm, self.config)
        
        # Build the workflow graph
        self.app = self._build_graph()
        
        logger.info("ExamPrepAgent initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_question", self.question_analyzer.analyze_question)
        workflow.add_node("process_query", self.query_processor.process_query)
        workflow.add_node("retrieve_documents", self.retriever.retrieve_documents)
        workflow.add_node("solve_math", self.math_solver.solve_mathematical_problem)
        workflow.add_node("synthesize_answer", self.answer_synthesizer.synthesize_answer)
        
        # Define the workflow edges
        workflow.add_edge(START, "analyze_question")
        workflow.add_edge("analyze_question", "process_query")
        workflow.add_edge("process_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "solve_math")
        workflow.add_edge("solve_math", "synthesize_answer")
        workflow.add_edge("synthesize_answer", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def chat(self, question: str, chat_history: List[Dict[str, str]] = None, user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a chat message through the agent
        
        Args:
            question: User's question
            chat_history: Previous conversation history
            user_preferences: User preferences and settings
            
        Returns:
            Complete response with answer and metadata
        """
        start_time = datetime.now()
        
        # Prepare initial state
        initial_state = {
            "question": question,
            "chat_history": chat_history or [],
            "user_preferences": user_preferences or {},
            "question_type": None,
            "subject_area": None,
            "difficulty_level": None,
            "requires_calculation": False,
            "requires_retrieval": True,
            "requires_step_by_step": False,
            "compressed_query": None,
            "retrieved_documents": [],
            "retrieval_stats": {},
            "reasoning_steps": [],
            "intermediate_calculations": [],
            "solution_approach": None,
            "final_answer": "",
            "confidence_score": 0.0,
            "sources_cited": [],
            "follow_up_suggestions": [],
            "processing_time": 0.0,
            "tokens_used": 0
        }
        
        try:
            # Run the agent workflow
            result = self.app.invoke(initial_state)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result["processing_time"] = processing_time
            
            logger.info(f"Chat completed in {processing_time:.2f}s with confidence {result['confidence_score']:.3f}")
            
            return {
                "answer": result["final_answer"],
                "confidence": result["confidence_score"],
                "sources": result["sources_cited"],
                "follow_ups": result["follow_up_suggestions"],
                "metadata": {
                    "question_type": result["question_type"],
                    "subject_area": result["subject_area"],
                    "difficulty_level": result["difficulty_level"],
                    "processing_time": processing_time,
                    "retrieval_stats": result["retrieval_stats"],
                    "reasoning_steps": result["reasoning_steps"]
                }
            }
            
        except Exception as e:
            logger.error(f"Agent processing failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again or rephrase your question.",
                "confidence": 0.1,
                "sources": [],
                "follow_ups": [],
                "metadata": {
                    "error": str(e),
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            }

# Global agent instance
_agent_instance = None

def get_exam_prep_agent(config: ExamBotConfig = None) -> ExamPrepAgent:
    """Get or create global agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ExamPrepAgent(config)
    return _agent_instance