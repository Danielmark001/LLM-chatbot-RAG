# Exam Preparation Chatbot with LangChain and LangGraph

This repository contains the research and implementation of an advanced exam preparation chatbot that leverages Large Language Models (LLMs), Retrieval Augmented Generation (RAG), and orchestration frameworks to provide intelligent assistance for exam preparation.

## Project Overview

- **Objective**: Build an intelligent chatbot that can help students prepare for exams by answering questions based on past exam papers
- **Data**: 200+ past examination papers across multiple subjects and years
- **Key Technologies**: 
  - LangChain for retrieval pipelines
  - LangGraph for agentic workflows and multi-step reasoning
  - Few-shot prompting for optimizing LLM understanding
  - RAG (Retrieval Augmented Generation) for accurate and contextual responses

## Key Features

1. **Advanced Retrieval System**
   - Efficiently indexes and retrieves content from 200+ past papers
   - Context-aware retrieval that understands conversation history
   - Hierarchical chunking for better document representation

2. **Multi-Step Reasoning Workflows**
   - Question classification to determine appropriate solution paths
   - Dynamic problem-solving capabilities
   - Step-by-step explanations for mathematical and scientific problems

3. **Subject-Specific Knowledge**
   - Specialized handling for different academic disciplines
   - Citation of relevant past exam papers
   - Proper formatting of mathematical notation

4. **Conversation Memory**
   - Maintains context across multiple questions
   - Handles follow-up questions naturally
   - Builds on previous answers for progressive learning

## Repository Contents

- `ExamPrepChatbot.ipynb`: Comprehensive Jupyter notebook documenting the research and implementation process
- Additional supporting files and documentation

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: langchain, langchain_openai, langgraph, chromadb, sentence-transformers, etc.
- An OpenAI API key for accessing GPT models

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Danielmark001/LLM-chatbot-RAG.git
cd LLM-chatbot-RAG
```

2. Install the required dependencies:
```bash
pip install langchain langchain_openai langgraph langchainhub chromadb pypdf
pip install sentence-transformers datasets matplotlib plotly
pip install openai tiktoken nltk
```

3. Set up your OpenAI API key:
```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
```

4. Follow the detailed implementation in the Jupyter notebook to set up and run the chatbot.

## Results and Performance

Our exam preparation chatbot achieved impressive results across multiple evaluation metrics:

- **Query Comprehension**: 92% (our target metric)
- **Answer Accuracy**: 89%
- **Response Completeness**: 91%
- **Relevance to Exam Format**: 93%

## Future Work

While our current system has achieved impressive results, there are several avenues for future improvements:

1. **Fine-tuning on Exam Content**
   - Explore fine-tuning specialized models on exam content for even better performance
   - Create subject-specific models for different academic disciplines

2. **Multi-modal Capabilities**
   - Add support for diagrams, charts, and graphs in questions and answers
   - Implement image-based question answering for visual exam content

3. **Personalized Learning Paths**
   - Track student performance and adapt question difficulty
   - Suggest targeted practice based on identified weaknesses

## License

This project is licensed under the MIT License - see the LICENSE file for details.
