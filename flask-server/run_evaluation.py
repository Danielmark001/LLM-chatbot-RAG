#!/usr/bin/env python3
"""
Comprehensive Evaluation Runner for Exam Preparation Chatbot
Runs all evaluation metrics and generates detailed reports
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our evaluation system
from evaluation.metrics import run_evaluation, ComprehensiveEvaluator
from agents.langgraph_agent import ExamBotConfig
from config.pinecone_config import get_pinecone_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

def create_sample_test_data(output_path: str):
    """Create comprehensive sample test data for evaluation"""
    
    sample_data = {
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
                "question": "Explain the process of photosynthesis and its importance in ecosystems.",
                "expected_type": "explanation", 
                "expected_subject": "biology",
                "difficulty_level": "intermediate",
                "requires_calculation": False,
                "requires_retrieval": True
            },
            {
                "question": "Compare and contrast mitosis and meiosis.",
                "expected_type": "comparison",
                "expected_subject": "biology",
                "difficulty_level": "advanced",
                "requires_calculation": False,
                "requires_retrieval": True
            },
            {
                "question": "Solve the system of equations: 2x + 3y = 12 and x - y = 1",
                "expected_type": "calculation",
                "expected_subject": "mathematics",
                "difficulty_level": "intermediate",
                "requires_calculation": True,
                "requires_retrieval": False
            },
            {
                "question": "What is Newton's second law of motion?",
                "expected_type": "factual",
                "expected_subject": "physics",
                "difficulty_level": "beginner",
                "requires_calculation": False,
                "requires_retrieval": True
            },
            {
                "question": "Describe the structure and function of mitochondria.",
                "expected_type": "explanation",
                "expected_subject": "biology", 
                "difficulty_level": "intermediate",
                "requires_calculation": False,
                "requires_retrieval": True
            },
            {
                "question": "Calculate the area of a circle with radius 5 cm.",
                "expected_type": "calculation",
                "expected_subject": "mathematics",
                "difficulty_level": "beginner",
                "requires_calculation": True,
                "requires_retrieval": False
            },
            {
                "question": "Explain the concept of supply and demand in economics.",
                "expected_type": "explanation",
                "expected_subject": "economics", 
                "difficulty_level": "intermediate",
                "requires_calculation": False,
                "requires_retrieval": True
            },
            {
                "question": "What are the key principles of object-oriented programming?",
                "expected_type": "explanation",
                "expected_subject": "computer_science",
                "difficulty_level": "intermediate", 
                "requires_calculation": False,
                "requires_retrieval": True
            },
            {
                "question": "Find the limit of (x^2 - 4)/(x - 2) as x approaches 2.",
                "expected_type": "calculation",
                "expected_subject": "mathematics",
                "difficulty_level": "advanced",
                "requires_calculation": True,
                "requires_retrieval": False
            }
        ],
        
        "qa_pairs": [
            {
                "question": "What is the derivative of x^2?",
                "reference_answer": "The derivative of x^2 is 2x. This is found using the power rule: d/dx[x^n] = n·x^(n-1). So for x^2, we have n=2, which gives us 2·x^(2-1) = 2x.",
                "subject": "mathematics"
            },
            {
                "question": "What is photosynthesis?",
                "reference_answer": "Photosynthesis is the process by which plants and other organisms convert light energy, usually from the sun, into chemical energy stored in glucose. The process occurs in two stages: the light-dependent reactions in the thylakoids and the light-independent reactions (Calvin cycle) in the stroma of chloroplasts. The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.",
                "subject": "biology"
            },
            {
                "question": "What is Newton's second law?",
                "reference_answer": "Newton's second law states that the force acting on an object is equal to its mass times its acceleration: F = ma. This law describes the relationship between the net force applied to an object and the resulting acceleration, taking into account the object's mass.",
                "subject": "physics"
            },
            {
                "question": "What is the area of a circle?",
                "reference_answer": "The area of a circle is calculated using the formula A = πr², where A is the area, π (pi) is approximately 3.14159, and r is the radius of the circle. This formula shows that the area is proportional to the square of the radius.",
                "subject": "mathematics"
            },
            {
                "question": "What is supply and demand?",
                "reference_answer": "Supply and demand is a fundamental economic principle that describes the relationship between the availability of a product (supply) and the desire for that product (demand). When supply is high and demand is low, prices tend to fall. When demand is high and supply is low, prices tend to rise. The equilibrium point where supply meets demand determines the market price.",
                "subject": "economics"
            }
        ],
        
        "completeness_questions": [
            {
                "question": "Solve the quadratic equation x^2 - 5x + 6 = 0",
                "completeness_criteria": [
                    "Identify the quadratic formula or factoring method",
                    "Show all calculation steps",
                    "Find both solutions",
                    "Verify the answers by substitution"
                ]
            },
            {
                "question": "Explain cellular respiration",
                "completeness_criteria": [
                    "Define cellular respiration",
                    "Describe the three main stages (glycolysis, citric acid cycle, electron transport)",
                    "Mention ATP production",
                    "Include the overall chemical equation"
                ]
            },
            {
                "question": "Describe the water cycle",
                "completeness_criteria": [
                    "Define the water cycle",
                    "Explain evaporation and transpiration",
                    "Describe condensation and precipitation",
                    "Mention groundwater and runoff"
                ]
            }
        ],
        
        "relevance_test_cases": [
            {
                "question": "What is the quadratic formula?",
                "exam_context": "Algebra exam focusing on solving quadratic equations and polynomial functions",
                "subject": "mathematics"
            },
            {
                "question": "Explain DNA replication",
                "exam_context": "Molecular biology exam covering genetic processes and cell division",
                "subject": "biology"  
            },
            {
                "question": "What is Ohm's law?",
                "exam_context": "Introductory physics exam on electricity and circuits",
                "subject": "physics"
            },
            {
                "question": "Describe market equilibrium",
                "exam_context": "Microeconomics exam on market structures and price determination",
                "subject": "economics"
            }
        ],
        
        "performance_questions": [
            "What is 2 + 2?",
            "Define gravity",
            "What is the capital of France?",
            "Explain photosynthesis briefly",
            "Solve x + 5 = 10",
            "What is Newton's first law?",
            "Define mitosis",
            "What is π (pi)?",
            "Explain supply and demand",
            "What is the derivative of x^3?"
        ]
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Sample test data created at {output_path}")
    return sample_data

def run_comprehensive_evaluation(test_data_path: str = None, 
                                config: ExamBotConfig = None,
                                create_sample: bool = False):
    """Run comprehensive evaluation with detailed reporting"""
    
    logger.info("Starting comprehensive evaluation of exam preparation chatbot...")
    logger.info("Target: 92% query comprehension accuracy")
    
    # Create sample test data if requested
    if create_sample or not test_data_path:
        if not test_data_path:
            test_data_path = "sample_test_data.json"
        create_sample_test_data(test_data_path)
    
    try:
        # Initialize evaluator
        logger.info("Initializing evaluation system...")
        evaluator = ComprehensiveEvaluator(config)
        
        # Check system status before evaluation
        logger.info("Checking system components...")
        try:
            pinecone_manager = get_pinecone_manager()
            index_stats = pinecone_manager.get_index_stats()
            logger.info(f"Pinecone index stats: {index_stats}")
        except Exception as e:
            logger.warning(f"Pinecone status check failed: {e}")
        
        # Run evaluation
        logger.info("Running comprehensive evaluation...")
        results = evaluator.run_full_evaluation(test_data_path)
        
        # Print detailed results
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"   Overall Score: {results.overall_score:.2f}%")
        print(f"   Target Achievement: {'✅ PASSED' if results.query_comprehension >= 92.0 else '❌ NEEDS IMPROVEMENT'}")
        
        print(f"\n🧠 QUERY COMPREHENSION")
        print(f"   Score: {results.query_comprehension:.2f}%")
        print(f"   Target: 92.0%")
        print(f"   Status: {'✅ TARGET ACHIEVED' if results.query_comprehension >= 92.0 else '⚠️  BELOW TARGET'}")
        
        print(f"\n✅ ANSWER ACCURACY") 
        print(f"   Score: {results.answer_accuracy:.2f}%")
        
        print(f"\n📝 RESPONSE COMPLETENESS")
        print(f"   Score: {results.response_completeness:.2f}%")
        
        print(f"\n🎯 EXAM RELEVANCE")
        print(f"   Score: {results.relevance_score:.2f}%")
        
        print(f"\n📚 CITATION QUALITY")
        print(f"   Score: {results.citation_quality:.2f}%")
        
        # Detailed breakdown
        print(f"\n📋 DETAILED RESULTS")
        for result in results.individual_results:
            print(f"   • {result.metric_name}: {result.percentage:.2f}% ({result.score:.1f}/{result.max_score:.1f})")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        if results.query_comprehension < 92.0:
            print("   • Improve question type classification accuracy")
            print("   • Enhance subject area detection")
            print("   • Fine-tune query understanding models")
        
        if results.answer_accuracy < 85.0:
            print("   • Improve retrieval quality with more relevant documents")
            print("   • Fine-tune answer generation models")
            print("   • Enhance context integration")
        
        if results.response_completeness < 80.0:
            print("   • Expand response templates for better coverage")
            print("   • Improve multi-step reasoning capabilities")
            print("   • Enhance citation and source integration")
        
        if results.relevance_score < 80.0:
            print("   • Improve subject-specific vocabulary")
            print("   • Enhance educational context understanding")
            print("   • Better integration with exam paper content")
        
        print(f"\n📈 SYSTEM METADATA")
        metadata = results.evaluation_metadata
        print(f"   • Evaluation Date: {metadata.get('evaluation_date', 'N/A')}")
        print(f"   • Test Data: {metadata.get('test_data_source', 'N/A')}")
        print(f"   • Target Achieved: {metadata.get('achieved_target', False)}")
        
        print("\n" + "="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def main():
    """Main evaluation runner with command line interface"""
    
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation of exam preparation chatbot")
    parser.add_argument(
        "--test-data", 
        type=str,
        help="Path to test data JSON file (will create sample if not provided)"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample test data file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-turbo",
        help="LLM model to use for evaluation (default: gpt-4-turbo)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Model temperature (default: 0.1)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for model responses (default: 2048)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_reports",
        help="Directory to save evaluation reports (default: ./evaluation_reports)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Configure agent
    config = ExamBotConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_hybrid_retrieval=True,
        retrieval_top_k=10,
        confidence_threshold=0.7,
        enable_citations=True,
        enable_follow_ups=True
    )
    
    try:
        # Run evaluation
        results = run_comprehensive_evaluation(
            test_data_path=args.test_data,
            config=config,
            create_sample=args.create_sample
        )
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(args.output_dir) / f"evaluation_summary_{timestamp}.json"
        
        summary_data = {
            "evaluation_summary": {
                "overall_score": results.overall_score,
                "query_comprehension": results.query_comprehension,
                "answer_accuracy": results.answer_accuracy,
                "response_completeness": results.response_completeness,
                "relevance_score": results.relevance_score,
                "citation_quality": results.citation_quality,
                "target_achieved": results.query_comprehension >= 92.0,
                "evaluation_date": datetime.now().isoformat(),
                "config": {
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens
                }
            },
            "individual_results": [
                {
                    "metric": result.metric_name,
                    "score": result.percentage,
                    "details": result.details
                }
                for result in results.individual_results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        exit_code = 0 if results.query_comprehension >= 92.0 else 1
        logger.info(f"Evaluation completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ EVALUATION FAILED: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()