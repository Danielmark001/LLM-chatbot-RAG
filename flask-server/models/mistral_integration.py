"""
Fine-tuned Mistral-7B Integration for Domain-Specific Reasoning
Includes model loading, fine-tuning utilities, and inference optimization
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import torch
from pathlib import Path
from dataclasses import dataclass

# Transformers and model-related imports
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    pipeline
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    PeftModel,
    prepare_model_for_kbit_training
)
import datasets
from datasets import Dataset
from accelerate import Accelerator

# LangChain integration
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema import LLMResult, Generation
from langchain.callbacks.manager import CallbackManagerForLLMRun

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MistralConfig:
    """Configuration for Mistral model integration"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    max_length: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    use_flash_attention: bool = True
    
    # LoRA configuration for fine-tuning
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

class ExamDatasetProcessor:
    """Process exam preparation datasets for fine-tuning"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def create_exam_training_data(self, 
                                 exam_papers_dir: str,
                                 questions_answers_file: Optional[str] = None) -> Dataset:
        """
        Create training dataset from exam papers and Q&A pairs
        
        Args:
            exam_papers_dir: Directory containing exam papers
            questions_answers_file: Optional JSON file with Q&A pairs
            
        Returns:
            Hugging Face Dataset
        """
        training_data = []
        
        # Load existing Q&A pairs if provided
        if questions_answers_file and os.path.exists(questions_answers_file):
            try:
                with open(questions_answers_file, 'r') as f:
                    qa_data = json.load(f)
                
                for item in qa_data:
                    if 'question' in item and 'answer' in item:
                        training_data.append({
                            'input': self._format_instruction_prompt(item['question']),
                            'output': item['answer'],
                            'subject': item.get('subject', 'general'),
                            'difficulty': item.get('difficulty', 'medium')
                        })
                        
                logger.info(f"Loaded {len(training_data)} Q&A pairs from {questions_answers_file}")
                
            except Exception as e:
                logger.error(f"Error loading Q&A file: {e}")
        
        # Generate synthetic training data from exam papers
        if os.path.exists(exam_papers_dir):
            synthetic_data = self._generate_synthetic_qa_from_papers(exam_papers_dir)
            training_data.extend(synthetic_data)
        
        # Convert to Hugging Face Dataset
        if training_data:
            dataset = Dataset.from_list(training_data)
            return dataset.map(self._tokenize_function, batched=True)
        else:
            logger.warning("No training data found")
            return Dataset.from_dict({'input_ids': [], 'attention_mask': [], 'labels': []})
    
    def _format_instruction_prompt(self, question: str, context: str = "") -> str:
        """Format question as instruction prompt for Mistral"""
        if context:
            prompt = f"""<s>[INST] You are an expert exam tutor. Use the following context to answer the student's question accurately and comprehensively.

Context: {context}

Question: {question} [/INST]"""
        else:
            prompt = f"""<s>[INST] You are an expert exam tutor. Answer the following question accurately and provide step-by-step explanations when appropriate.

Question: {question} [/INST]"""
        
        return prompt
    
    def _generate_synthetic_qa_from_papers(self, papers_dir: str) -> List[Dict[str, Any]]:
        """Generate synthetic Q&A pairs from exam papers using a larger model"""
        # This would typically use a larger model like GPT-4 to generate training data
        # For demonstration, we'll create a few example pairs
        
        synthetic_data = [
            {
                'input': self._format_instruction_prompt("What is the derivative of f(x) = x^3 + 2x^2 - 5x + 3?"),
                'output': """To find the derivative of f(x) = x³ + 2x² - 5x + 3, I'll apply the power rule to each term.

The power rule states that d/dx[x^n] = n·x^(n-1).

For each term:
- d/dx[x³] = 3x²
- d/dx[2x²] = 2·2x¹ = 4x
- d/dx[-5x] = -5
- d/dx[3] = 0 (constant term)

Therefore: f'(x) = 3x² + 4x - 5""",
                'subject': 'mathematics',
                'difficulty': 'medium'
            },
            {
                'input': self._format_instruction_prompt("Explain the process of photosynthesis."),
                'output': """Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in two main stages:

1. **Light-dependent reactions** (in thylakoids):
   - Chlorophyll absorbs light energy
   - Water molecules are split (photolysis): 2H₂O → 4H⁺ + 4e⁻ + O₂
   - ATP and NADPH are produced

2. **Light-independent reactions** (Calvin cycle in stroma):
   - CO₂ is fixed into organic molecules
   - ATP and NADPH from stage 1 provide energy
   - Glucose is produced

Overall equation: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂

This process is crucial for life on Earth as it produces oxygen and forms the base of most food chains.""",
                'subject': 'biology',
                'difficulty': 'medium'
            }
        ]
        
        logger.info(f"Generated {len(synthetic_data)} synthetic training examples")
        return synthetic_data
    
    def _tokenize_function(self, examples):
        """Tokenize training examples"""
        # Tokenize inputs and outputs
        model_inputs = self.tokenizer(
            examples['input'], 
            truncation=True, 
            padding=True, 
            max_length=1024
        )
        
        labels = self.tokenizer(
            examples['output'], 
            truncation=True, 
            padding=True, 
            max_length=1024
        )
        
        # Set labels (for causal LM, input_ids become labels)
        model_inputs['labels'] = labels['input_ids']
        
        return model_inputs

class MistralFineTuner:
    """Fine-tune Mistral models for exam preparation tasks"""
    
    def __init__(self, config: MistralConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Setup base model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Setup quantization config
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map=self.config.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if quantization_config else torch.float32
        )
        
        logger.info("Model and tokenizer loaded successfully")
        
    def setup_lora(self):
        """Setup LoRA for parameter-efficient fine-tuning"""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_model_and_tokenizer() first.")
        
        # Prepare model for k-bit training if using quantization
        if self.config.load_in_4bit or self.config.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied successfully")
        
    def fine_tune(self, 
                  train_dataset: Dataset,
                  output_dir: str = "./mistral_exam_model",
                  num_epochs: int = 3,
                  learning_rate: float = 2e-4,
                  batch_size: int = 4,
                  gradient_accumulation_steps: int = 4):
        """
        Fine-tune the model on exam preparation data
        
        Args:
            train_dataset: Training dataset
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
        """
        if self.peft_model is None:
            raise ValueError("PEFT model not setup. Call setup_lora() first.")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",  # Can be changed to "steps" if you have eval data
            save_strategy="steps",
            warmup_steps=100,
            fp16=True,
            optim="adamw_torch",
            remove_unused_columns=False,
            dataloader_pin_memory=False
        )
        
        # Custom data collator for instruction tuning
        def data_collator(features):
            batch = {}
            batch['input_ids'] = torch.stack([torch.tensor(f['input_ids']) for f in features])
            batch['attention_mask'] = torch.stack([torch.tensor(f['attention_mask']) for f in features])
            batch['labels'] = torch.stack([torch.tensor(f['labels']) for f in features])
            return batch
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        logger.info("Starting fine-tuning...")
        
        # Start training
        try:
            trainer.train()
            
            # Save the final model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise
    
    def load_fine_tuned_model(self, model_path: str):
        """Load a previously fine-tuned model"""
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        # Load base model first
        if self.model is None:
            self.setup_model_and_tokenizer()
        
        # Load PEFT model
        self.peft_model = PeftModel.from_pretrained(self.model, model_path)
        
        logger.info("Fine-tuned model loaded successfully")

class MistralLLM:
    """LangChain-compatible wrapper for Mistral model"""
    
    def __init__(self, 
                 model_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
                 config: MistralConfig = None,
                 fine_tuned_path: Optional[str] = None):
        """
        Initialize Mistral LLM
        
        Args:
            model_path: Path to base model
            config: Model configuration
            fine_tuned_path: Optional path to fine-tuned model
        """
        self.config = config or MistralConfig()
        self.config.model_name = model_path
        
        # Setup model
        self.fine_tuner = MistralFineTuner(self.config)
        self.fine_tuner.setup_model_and_tokenizer()
        
        # Load fine-tuned model if provided
        if fine_tuned_path and os.path.exists(fine_tuned_path):
            self.fine_tuner.load_fine_tuned_model(fine_tuned_path)
            self.model = self.fine_tuner.peft_model
        else:
            self.model = self.fine_tuner.model
        
        self.tokenizer = self.fine_tuner.tokenizer
        
        # Create inference pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Create LangChain wrapper
        self.llm = HuggingFacePipeline(
            pipeline=self.pipeline,
            model_kwargs={
                "temperature": self.config.temperature,
                "max_length": self.config.max_length,
                "top_p": self.config.top_p
            }
        )
        
        logger.info("MistralLLM initialized successfully")
    
    def generate(self, 
                prompt: str, 
                max_length: Optional[int] = None,
                temperature: Optional[float] = None) -> str:
        """
        Generate text using the Mistral model
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Generation temperature
            
        Returns:
            Generated text
        """
        # Format as instruction if not already formatted
        if not prompt.strip().startswith("<s>[INST]"):
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt
        
        # Generate
        outputs = self.pipeline(
            formatted_prompt,
            max_length=max_length or self.config.max_length,
            temperature=temperature or self.config.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract generated text (remove input prompt)
        generated = outputs[0]['generated_text']
        
        # Remove the input prompt from the output
        if "[/INST]" in generated:
            response = generated.split("[/INST]", 1)[1].strip()
        else:
            response = generated.replace(formatted_prompt, "").strip()
        
        return response
    
    def get_langchain_llm(self):
        """Get the LangChain-compatible LLM instance"""
        return self.llm

class MistralModelManager:
    """Manager for Mistral model operations"""
    
    def __init__(self):
        self.models = {}
        self.default_config = MistralConfig()
    
    def load_model(self, 
                   model_name: str,
                   model_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
                   fine_tuned_path: Optional[str] = None,
                   config: Optional[MistralConfig] = None) -> MistralLLM:
        """Load and cache a Mistral model"""
        
        if model_name in self.models:
            logger.info(f"Returning cached model: {model_name}")
            return self.models[model_name]
        
        logger.info(f"Loading new model: {model_name}")
        model = MistralLLM(
            model_path=model_path,
            config=config or self.default_config,
            fine_tuned_path=fine_tuned_path
        )
        
        self.models[model_name] = model
        return model
    
    def fine_tune_model(self,
                       base_model_path: str,
                       training_data_path: str,
                       output_path: str,
                       model_name: str = "fine_tuned_mistral",
                       config: Optional[MistralConfig] = None) -> str:
        """
        Fine-tune a Mistral model on exam preparation data
        
        Returns:
            Path to the fine-tuned model
        """
        config = config or self.default_config
        
        # Setup fine-tuner
        fine_tuner = MistralFineTuner(config)
        fine_tuner.setup_model_and_tokenizer()
        fine_tuner.setup_lora()
        
        # Process training data
        dataset_processor = ExamDatasetProcessor(fine_tuner.tokenizer)
        
        if training_data_path.endswith('.json'):
            # Load from JSON file
            train_dataset = dataset_processor.create_exam_training_data(
                exam_papers_dir="",  # Not needed for JSON
                questions_answers_file=training_data_path
            )
        else:
            # Load from directory
            train_dataset = dataset_processor.create_exam_training_data(
                exam_papers_dir=training_data_path
            )
        
        if len(train_dataset) == 0:
            raise ValueError("No training data found")
        
        # Fine-tune
        fine_tuner.fine_tune(
            train_dataset=train_dataset,
            output_dir=output_path
        )
        
        # Load the fine-tuned model
        fine_tuned_model = self.load_model(
            model_name=model_name,
            model_path=base_model_path,
            fine_tuned_path=output_path,
            config=config
        )
        
        return output_path

# Global model manager
_model_manager = None

def get_mistral_manager() -> MistralModelManager:
    """Get or create global Mistral model manager"""
    global _model_manager
    if _model_manager is None:
        _model_manager = MistralModelManager()
    return _model_manager