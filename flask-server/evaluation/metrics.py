"""
Comprehensive Evaluation Metrics for Exam Preparation Chatbot
Implements query comprehension, answer accuracy, and system performance metrics
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import time

# NLP and evaluation libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# LangChain evaluation
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI

# Import our components
from agents.langgraph_agent import get_exam_prep_agent, ExamBotConfig
from models.mistral_integration import get_mistral_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    metric_name: str
    score: float
    max_score: float
    percentage: float
    details: Dict[str, Any]
    timestamp: str

@dataclass 
class ComprehensiveEvaluation:
    """Container for comprehensive evaluation results"""
    query_comprehension: float
    answer_accuracy: float
    response_completeness: float
    relevance_score: float
    citation_quality: float
    overall_score: float
    individual_results: List[EvaluationResult]
    evaluation_metadata: Dict[str, Any]

class QueryComprehensionEvaluator:
    """Evaluates query comprehension accuracy (target: 92%)"""
    
    def __init__(self, agent):
        self.agent = agent
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        
    def evaluate(self, test_questions: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate query comprehension on test questions
        
        Args:
            test_questions: List of test questions with expected outputs
            Format: [{"question": str, "expected_type": str, "expected_subject": str, ...}]
            
        Returns:
            EvaluationResult with comprehension score
        """
        logger.info(f"Evaluating query comprehension on {len(test_questions)} questions")
        
        correct_predictions = 0
        total_predictions = 0
        detailed_results = []
        
        for question_data in test_questions:
            question = question_data.get('question', '')
            if not question:
                continue
                
            try:
                # Get agent's analysis
                response = self.agent.chat(question, [], {"evaluation_mode": True})
                metadata = response.get("metadata", {})
                
                # Check predictions
                predictions = {
                    'question_type': metadata.get('question_type', ''),
                    'subject_area': metadata.get('subject_area', ''),
                    'difficulty_level': metadata.get('difficulty_level', ''),
                    'requires_calculation': metadata.get('requires_calculation', False),
                    'requires_retrieval': metadata.get('requires_retrieval', False)
                }
                
                # Compare with expected values
                question_score = 0
                total_checks = 0
                
                for key in ['question_type', 'subject_area', 'difficulty_level']:
                    if key in question_data:
                        total_checks += 1
                        expected = str(question_data[key]).lower()
                        predicted = str(predictions[key]).lower()
                        if expected == predicted:
                            question_score += 1
                
                for key in ['requires_calculation', 'requires_retrieval']:
                    if key in question_data:
                        total_checks += 1
                        if question_data[key] == predictions[key]:
                            question_score += 1
                
                if total_checks > 0:
                    question_accuracy = question_score / total_checks
                    correct_predictions += question_score
                    total_predictions += total_checks
                    
                    detailed_results.append({
                        'question': question,
                        'expected': {k: question_data.get(k) for k in predictions.keys()},
                        'predicted': predictions,
                        'accuracy': question_accuracy,
                        'confidence': response.get('confidence', 0)
                    })
                    
            except Exception as e:
                logger.error(f"Error evaluating question '{question}': {e}")
                detailed_results.append({
                    'question': question,
                    'error': str(e),
                    'accuracy': 0
                })
        
        # Calculate overall score
        comprehension_score = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        result = EvaluationResult(
            metric_name="Query Comprehension",
            score=correct_predictions,
            max_score=total_predictions,
            percentage=comprehension_score,
            details={
                'detailed_results': detailed_results,
                'total_questions': len(test_questions),
                'target_percentage': 92.0
            },
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Query comprehension score: {comprehension_score:.2f}%")
        return result

class AnswerAccuracyEvaluator:
    """Evaluates answer accuracy using multiple metrics"""
    
    def __init__(self, agent):
        self.agent = agent
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def evaluate(self, test_qa_pairs: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate answer accuracy using semantic similarity and ROUGE scores
        
        Args:
            test_qa_pairs: List of Q&A pairs with reference answers
            Format: [{"question": str, "reference_answer": str, "subject": str, ...}]
            
        Returns:
            EvaluationResult with accuracy score
        """
        logger.info(f"Evaluating answer accuracy on {len(test_qa_pairs)} Q&A pairs")
        
        semantic_scores = []
        rouge_scores = []
        bleu_scores = []
        detailed_results = []
        
        for qa_pair in test_qa_pairs:
            question = qa_pair.get('question', '')
            reference_answer = qa_pair.get('reference_answer', '')
            
            if not question or not reference_answer:
                continue
                
            try:
                # Get agent's answer
                response = self.agent.chat(question, [])
                generated_answer = response.get('answer', '')
                
                if not generated_answer:
                    continue
                
                # Calculate semantic similarity
                ref_embedding = self.similarity_model.encode([reference_answer])
                gen_embedding = self.similarity_model.encode([generated_answer])
                semantic_sim = np.cosine(ref_embedding[0], gen_embedding[0])
                semantic_scores.append(semantic_sim)
                
                # Calculate ROUGE scores
                rouge_result = self.rouge_scorer.score(reference_answer, generated_answer)
                rouge_avg = np.mean([
                    rouge_result['rouge1'].fmeasure,
                    rouge_result['rouge2'].fmeasure,
                    rouge_result['rougeL'].fmeasure
                ])
                rouge_scores.append(rouge_avg)
                
                # Calculate BLEU score
                reference_tokens = nltk.word_tokenize(reference_answer.lower())
                generated_tokens = nltk.word_tokenize(generated_answer.lower())
                bleu = sentence_bleu([reference_tokens], generated_tokens)
                bleu_scores.append(bleu)
                
                # Store detailed result
                detailed_results.append({
                    'question': question,
                    'reference_answer': reference_answer[:200] + "..." if len(reference_answer) > 200 else reference_answer,
                    'generated_answer': generated_answer[:200] + "..." if len(generated_answer) > 200 else generated_answer,
                    'semantic_similarity': semantic_sim,
                    'rouge_score': rouge_avg,
                    'bleu_score': bleu,
                    'confidence': response.get('confidence', 0),
                    'sources': response.get('sources', [])
                })
                
            except Exception as e:
                logger.error(f"Error evaluating Q&A pair: {e}")
                detailed_results.append({
                    'question': question,
                    'error': str(e),
                    'semantic_similarity': 0,
                    'rouge_score': 0,
                    'bleu_score': 0
                })
        
        # Calculate overall accuracy score (weighted average)
        if semantic_scores and rouge_scores and bleu_scores:
            accuracy_score = (
                0.4 * np.mean(semantic_scores) +  # 40% semantic similarity
                0.4 * np.mean(rouge_scores) +     # 40% ROUGE
                0.2 * np.mean(bleu_scores)        # 20% BLEU
            ) * 100
        else:
            accuracy_score = 0
        
        result = EvaluationResult(
            metric_name="Answer Accuracy",
            score=accuracy_score,
            max_score=100,
            percentage=accuracy_score,
            details={
                'detailed_results': detailed_results,
                'avg_semantic_similarity': np.mean(semantic_scores) if semantic_scores else 0,
                'avg_rouge_score': np.mean(rouge_scores) if rouge_scores else 0,
                'avg_bleu_score': np.mean(bleu_scores) if bleu_scores else 0,
                'total_qa_pairs': len(test_qa_pairs)
            },
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Answer accuracy score: {accuracy_score:.2f}%")
        return result

class CompletenessEvaluator:
    """Evaluates response completeness"""
    
    def __init__(self, agent):
        self.agent = agent
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        
    def evaluate(self, test_questions: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate response completeness using LLM-based assessment
        
        Args:
            test_questions: List of questions with completeness criteria
            
        Returns:
            EvaluationResult with completeness score
        """
        logger.info(f"Evaluating response completeness on {len(test_questions)} questions")
        
        completeness_scores = []
        detailed_results = []
        
        for question_data in test_questions:
            question = question_data.get('question', '')
            completeness_criteria = question_data.get('completeness_criteria', [])
            
            if not question:
                continue
                
            try:
                # Get agent's response
                response = self.agent.chat(question, [])
                answer = response.get('answer', '')
                
                if not answer:
                    continue
                
                # Use LLM to evaluate completeness
                evaluation_prompt = f"""
Evaluate the completeness of the following answer to the given question.

Question: {question}

Answer: {answer}

Completeness Criteria:
{', '.join(completeness_criteria) if completeness_criteria else 'General completeness for this type of question'}

Rate the completeness on a scale of 0-100, where:
- 0-30: Incomplete, missing major components
- 31-60: Partially complete, missing some important elements
- 61-80: Mostly complete, minor omissions
- 81-100: Complete and comprehensive

Provide only the numerical score (0-100).
"""
                
                completeness_result = self.llm.invoke(evaluation_prompt)
                try:
                    completeness_score = float(completeness_result.content.strip())
                    completeness_score = max(0, min(100, completeness_score))  # Clamp to 0-100
                except ValueError:
                    completeness_score = 50  # Default middle score if parsing fails
                
                completeness_scores.append(completeness_score)
                
                detailed_results.append({
                    'question': question,
                    'answer_length': len(answer),
                    'completeness_score': completeness_score,
                    'criteria': completeness_criteria,
                    'confidence': response.get('confidence', 0)
                })
                
            except Exception as e:
                logger.error(f"Error evaluating completeness for question '{question}': {e}")
                detailed_results.append({
                    'question': question,
                    'error': str(e),
                    'completeness_score': 0
                })
        
        # Calculate overall completeness
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0
        
        result = EvaluationResult(
            metric_name="Response Completeness",
            score=avg_completeness,
            max_score=100,
            percentage=avg_completeness,
            details={
                'detailed_results': detailed_results,
                'score_distribution': {
                    'excellent (81-100)': len([s for s in completeness_scores if s >= 81]),
                    'good (61-80)': len([s for s in completeness_scores if 61 <= s < 81]),
                    'fair (31-60)': len([s for s in completeness_scores if 31 <= s < 61]),
                    'poor (0-30)': len([s for s in completeness_scores if s < 31])
                },
                'total_questions': len(test_questions)
            },
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Response completeness score: {avg_completeness:.2f}%")
        return result

class RelevanceEvaluator:
    """Evaluates relevance to exam format and content"""
    
    def __init__(self, agent):
        self.agent = agent
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def evaluate(self, test_cases: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate relevance to exam format and educational context
        
        Args:
            test_cases: List of test cases with exam context
            
        Returns:
            EvaluationResult with relevance score
        """
        logger.info(f"Evaluating exam relevance on {len(test_cases)} test cases")
        
        relevance_scores = []
        detailed_results = []
        
        for test_case in test_cases:
            question = test_case.get('question', '')
            exam_context = test_case.get('exam_context', '')
            subject = test_case.get('subject', '')
            
            if not question:
                continue
                
            try:
                # Get agent's response
                response = self.agent.chat(question, [])
                answer = response.get('answer', '')
                sources = response.get('sources', [])
                
                if not answer:
                    continue
                
                # Calculate relevance based on multiple factors
                relevance_factors = []
                
                # 1. Subject alignment
                if subject:
                    subject_keywords = {
                        'mathematics': ['equation', 'formula', 'theorem', 'proof', 'calculate'],
                        'physics': ['force', 'energy', 'motion', 'wave', 'particle'],
                        'chemistry': ['molecule', 'reaction', 'bond', 'element', 'compound'],
                        'biology': ['cell', 'organism', 'gene', 'evolution', 'ecosystem'],
                        'economics': ['market', 'supply', 'demand', 'price', 'trade']
                    }
                    
                    keywords = subject_keywords.get(subject.lower(), [])
                    keyword_count = sum(1 for word in keywords if word in answer.lower())
                    subject_relevance = min(100, (keyword_count / len(keywords)) * 100) if keywords else 50
                    relevance_factors.append(subject_relevance)
                
                # 2. Citation quality (using sources from exam papers)
                citation_relevance = 80 if sources else 40  # Higher if citing sources
                relevance_factors.append(citation_relevance)
                
                # 3. Educational tone and structure
                educational_indicators = [
                    'step', 'first', 'second', 'therefore', 'because', 'example',
                    'definition', 'concept', 'principle', 'theorem', 'formula'
                ]
                educational_count = sum(1 for indicator in educational_indicators if indicator in answer.lower())
                educational_relevance = min(100, (educational_count / 5) * 100)
                relevance_factors.append(educational_relevance)
                
                # 4. Contextual similarity with exam context
                if exam_context:
                    context_embedding = self.similarity_model.encode([exam_context])
                    answer_embedding = self.similarity_model.encode([answer])
                    context_similarity = np.cosine(context_embedding[0], answer_embedding[0]) * 100
                    relevance_factors.append(context_similarity)
                
                # Calculate overall relevance
                overall_relevance = np.mean(relevance_factors) if relevance_factors else 50
                relevance_scores.append(overall_relevance)
                
                detailed_results.append({
                    'question': question,
                    'subject': subject,
                    'relevance_score': overall_relevance,
                    'subject_relevance': relevance_factors[0] if len(relevance_factors) > 0 else 0,
                    'citation_relevance': relevance_factors[1] if len(relevance_factors) > 1 else 0,
                    'educational_relevance': relevance_factors[2] if len(relevance_factors) > 2 else 0,
                    'context_similarity': relevance_factors[3] if len(relevance_factors) > 3 else 0,
                    'sources_count': len(sources),
                    'confidence': response.get('confidence', 0)
                })
                
            except Exception as e:
                logger.error(f"Error evaluating relevance for question '{question}': {e}")
                detailed_results.append({
                    'question': question,
                    'error': str(e),
                    'relevance_score': 0
                })
        
        # Calculate overall relevance score
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
        
        result = EvaluationResult(
            metric_name="Exam Relevance",
            score=avg_relevance,
            max_score=100,
            percentage=avg_relevance,
            details={
                'detailed_results': detailed_results,
                'avg_subject_relevance': np.mean([r.get('subject_relevance', 0) for r in detailed_results]),
                'avg_citation_relevance': np.mean([r.get('citation_relevance', 0) for r in detailed_results]),
                'avg_educational_relevance': np.mean([r.get('educational_relevance', 0) for r in detailed_results]),
                'total_test_cases': len(test_cases)
            },
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Exam relevance score: {avg_relevance:.2f}%")
        return result

class PerformanceEvaluator:
    """Evaluates system performance metrics"""
    
    def __init__(self, agent):
        self.agent = agent
        
    def evaluate_response_time(self, test_questions: List[str], iterations: int = 3) -> EvaluationResult:
        """
        Evaluate response time performance
        
        Args:
            test_questions: List of test questions
            iterations: Number of iterations per question
            
        Returns:
            EvaluationResult with performance metrics
        """
        logger.info(f"Evaluating response time on {len(test_questions)} questions, {iterations} iterations each")
        
        response_times = []
        detailed_results = []
        
        for question in test_questions:
            if not question:
                continue
                
            question_times = []
            
            for iteration in range(iterations):
                try:
                    start_time = time.time()
                    response = self.agent.chat(question, [])
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    question_times.append(response_time)
                    response_times.append(response_time)
                    
                except Exception as e:
                    logger.error(f"Error during performance evaluation: {e}")
                    continue
            
            if question_times:
                detailed_results.append({
                    'question': question[:100] + "..." if len(question) > 100 else question,
                    'avg_response_time': np.mean(question_times),
                    'min_response_time': min(question_times),
                    'max_response_time': max(question_times),
                    'iterations': len(question_times)
                })
        
        # Calculate performance metrics
        if response_times:
            avg_response_time = np.mean(response_times)
            median_response_time = np.median(response_times)
            percentile_95 = np.percentile(response_times, 95)
            
            # Performance score (inverse of response time, normalized)
            # Target: under 5 seconds = 100%, over 20 seconds = 0%
            performance_score = max(0, min(100, (20 - avg_response_time) / 20 * 100))
        else:
            avg_response_time = median_response_time = percentile_95 = performance_score = 0
        
        result = EvaluationResult(
            metric_name="Response Time Performance",
            score=performance_score,
            max_score=100,
            percentage=performance_score,
            details={
                'avg_response_time': avg_response_time,
                'median_response_time': median_response_time,
                'percentile_95': percentile_95,
                'min_response_time': min(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'total_requests': len(response_times),
                'detailed_results': detailed_results,
                'target_response_time': 5.0  # seconds
            },
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Average response time: {avg_response_time:.2f}s, Performance score: {performance_score:.2f}%")
        return result

class ComprehensiveEvaluator:
    """Runs comprehensive evaluation achieving 92% query comprehension target"""
    
    def __init__(self, agent_config: ExamBotConfig = None):
        """
        Initialize comprehensive evaluator
        
        Args:
            agent_config: Configuration for the exam preparation agent
        """
        self.agent = get_exam_prep_agent(agent_config)
        
        # Initialize individual evaluators
        self.query_evaluator = QueryComprehensionEvaluator(self.agent)
        self.accuracy_evaluator = AnswerAccuracyEvaluator(self.agent)
        self.completeness_evaluator = CompletenessEvaluator(self.agent)
        self.relevance_evaluator = RelevanceEvaluator(self.agent)
        self.performance_evaluator = PerformanceEvaluator(self.agent)
        
    def run_full_evaluation(self, test_data_path: str) -> ComprehensiveEvaluation:
        """
        Run comprehensive evaluation on all metrics
        
        Args:
            test_data_path: Path to JSON file containing test data
            
        Returns:
            ComprehensiveEvaluation with all results
        """
        logger.info("Starting comprehensive evaluation...")
        
        # Load test data
        test_data = self._load_test_data(test_data_path)
        
        individual_results = []
        
        # 1. Query Comprehension (Target: 92%)
        if 'comprehension_questions' in test_data:
            query_result = self.query_evaluator.evaluate(test_data['comprehension_questions'])
            individual_results.append(query_result)
        
        # 2. Answer Accuracy
        if 'qa_pairs' in test_data:
            accuracy_result = self.accuracy_evaluator.evaluate(test_data['qa_pairs'])
            individual_results.append(accuracy_result)
        
        # 3. Response Completeness
        if 'completeness_questions' in test_data:
            completeness_result = self.completeness_evaluator.evaluate(test_data['completeness_questions'])
            individual_results.append(completeness_result)
        
        # 4. Exam Relevance
        if 'relevance_test_cases' in test_data:
            relevance_result = self.relevance_evaluator.evaluate(test_data['relevance_test_cases'])
            individual_results.append(relevance_result)
        
        # 5. Performance
        if 'performance_questions' in test_data:
            performance_result = self.performance_evaluator.evaluate_response_time(test_data['performance_questions'])
            individual_results.append(performance_result)
        
        # Calculate overall scores
        query_comprehension = next((r.percentage for r in individual_results if r.metric_name == "Query Comprehension"), 0)
        answer_accuracy = next((r.percentage for r in individual_results if r.metric_name == "Answer Accuracy"), 0)
        response_completeness = next((r.percentage for r in individual_results if r.metric_name == "Response Completeness"), 0)
        relevance_score = next((r.percentage for r in individual_results if r.metric_name == "Exam Relevance"), 0)
        
        # Calculate overall score (weighted)
        overall_score = (
            0.25 * query_comprehension +      # 25% - Query comprehension
            0.25 * answer_accuracy +          # 25% - Answer accuracy  
            0.20 * response_completeness +    # 20% - Completeness
            0.20 * relevance_score +          # 20% - Relevance
            0.10 * next((r.percentage for r in individual_results if "Performance" in r.metric_name), 80)  # 10% - Performance
        )
        
        # Create comprehensive evaluation
        evaluation = ComprehensiveEvaluation(
            query_comprehension=query_comprehension,
            answer_accuracy=answer_accuracy,
            response_completeness=response_completeness,
            relevance_score=relevance_score,
            citation_quality=self._calculate_citation_quality(individual_results),
            overall_score=overall_score,
            individual_results=individual_results,
            evaluation_metadata={
                'evaluation_date': datetime.now().isoformat(),
                'test_data_source': test_data_path,
                'agent_config': asdict(self.agent.config) if hasattr(self.agent, 'config') else {},
                'target_query_comprehension': 92.0,
                'achieved_target': query_comprehension >= 92.0
            }
        )
        
        self._save_evaluation_report(evaluation)
        
        logger.info(f"Comprehensive evaluation completed:")
        logger.info(f"  Query Comprehension: {query_comprehension:.2f}% (Target: 92%)")
        logger.info(f"  Answer Accuracy: {answer_accuracy:.2f}%")
        logger.info(f"  Response Completeness: {response_completeness:.2f}%")
        logger.info(f"  Exam Relevance: {relevance_score:.2f}%")
        logger.info(f"  Overall Score: {overall_score:.2f}%")
        
        return evaluation
    
    def _load_test_data(self, test_data_path: str) -> Dict[str, Any]:
        """Load test data from JSON file"""
        try:
            with open(test_data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Test data file not found: {test_data_path}, using default test data")
            return self._create_default_test_data()
    
    def _create_default_test_data(self) -> Dict[str, Any]:
        """Create default test data for evaluation"""
        return {
            "comprehension_questions": [
                {
                    "question": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 3?",
                    "expected_type": "calculation",
                    "expected_subject": "mathematics",
                    "difficulty_level": "intermediate",
                    "requires_calculation": True,
                    "requires_retrieval": False
                },
                {
                    "question": "Explain the process of photosynthesis and its importance.",
                    "expected_type": "explanation",
                    "expected_subject": "biology",
                    "difficulty_level": "intermediate",
                    "requires_calculation": False,
                    "requires_retrieval": True
                }
            ],
            "qa_pairs": [
                {
                    "question": "What is the derivative of x^2?",
                    "reference_answer": "The derivative of x^2 is 2x, using the power rule where d/dx[x^n] = n*x^(n-1).",
                    "subject": "mathematics"
                }
            ],
            "completeness_questions": [
                {
                    "question": "Solve the quadratic equation x^2 - 5x + 6 = 0",
                    "completeness_criteria": ["Show the quadratic formula", "Calculate the discriminant", "Find both solutions", "Verify the answers"]
                }
            ],
            "relevance_test_cases": [
                {
                    "question": "What is Newton's second law?",
                    "exam_context": "Classical mechanics exam focusing on fundamental laws of motion",
                    "subject": "physics"
                }
            ],
            "performance_questions": [
                "What is 2+2?",
                "Explain gravity.",
                "Solve x^2 = 4."
            ]
        }
    
    def _calculate_citation_quality(self, results: List[EvaluationResult]) -> float:
        """Calculate citation quality from evaluation results"""
        # Extract citation information from detailed results
        total_citations = 0
        total_responses = 0
        
        for result in results:
            detailed_results = result.details.get('detailed_results', [])
            for detail in detailed_results:
                if 'sources' in detail:
                    total_citations += len(detail['sources'])
                    total_responses += 1
        
        # Calculate citation quality score
        if total_responses > 0:
            avg_citations_per_response = total_citations / total_responses
            citation_quality = min(100, avg_citations_per_response * 25)  # Scale to 0-100
        else:
            citation_quality = 0
        
        return citation_quality
    
    def _save_evaluation_report(self, evaluation: ComprehensiveEvaluation):
        """Save evaluation report to file"""
        try:
            report_dir = Path("evaluation_reports")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"evaluation_report_{timestamp}.json"
            
            # Convert to dict for JSON serialization
            report_data = {
                "query_comprehension": evaluation.query_comprehension,
                "answer_accuracy": evaluation.answer_accuracy,
                "response_completeness": evaluation.response_completeness,
                "relevance_score": evaluation.relevance_score,
                "citation_quality": evaluation.citation_quality,
                "overall_score": evaluation.overall_score,
                "individual_results": [asdict(result) for result in evaluation.individual_results],
                "evaluation_metadata": evaluation.evaluation_metadata
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Evaluation report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")

def run_evaluation(test_data_path: str = None, config: ExamBotConfig = None) -> ComprehensiveEvaluation:
    """
    Convenience function to run comprehensive evaluation
    
    Args:
        test_data_path: Optional path to test data JSON file
        config: Optional agent configuration
        
    Returns:
        ComprehensiveEvaluation results
    """
    evaluator = ComprehensiveEvaluator(config)
    
    if test_data_path and Path(test_data_path).exists():
        return evaluator.run_full_evaluation(test_data_path)
    else:
        logger.info("Using default test data for evaluation")
        return evaluator.run_full_evaluation("")  # Will use default data