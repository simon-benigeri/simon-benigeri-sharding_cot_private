#!/usr/bin/env python3
"""
Simple Hugging Face SFT Training

Direct SFT training using Hugging Face transformers and datasets.
Much simpler than Axolotl for basic use cases.
"""
import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os
import mlflow
import mlflow.pytorch
import gc
import time
import psutil

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_jsonl_dataset(file_path: str) -> Dict[str, Dataset]:
    """
    Load JSONL dataset into Hugging Face Dataset format with train/validation split.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Dictionary with 'train' and 'validation' datasets
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Create dataset and split into train (90%) and validation (10%)
    full_dataset = Dataset.from_list(data)
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    
    return {
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    }


def format_prompt(example, system_prompt=None):
    """
    Format example into training prompt.
    
    Args:
        example: Dictionary with 'question' and 'answer' keys
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Formatted prompt string with question and answer for SFT
    """
    question = example.get('question', '')
    answer = example.get('answer', '')
    
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that solves math problems step by step. Always end your answer with '#### [number]' where [number] is the final numerical answer."
    
    # Format with system prompt for proper SFT on QA pairs
    if system_prompt:
        return f"{system_prompt}\nQuestion: {question}\nAnswer: {answer}"
    else:
        return f"Question: {question}\nAnswer: {answer}"


def find_optimal_batch_size(model, tokenizer, dataset, max_length: int, max_batch_size: int = 16) -> int:
    """
    Find the maximum batch size that fits in GPU memory.
    
    Args:
        model: The model to test
        tokenizer: The tokenizer
        dataset: Sample dataset for testing
        max_length: Maximum sequence length
        max_batch_size: Maximum batch size to try
        
    Returns:
        Optimal batch size
    """
    print("🔍 Finding optimal batch size...")
    
    if not torch.cuda.is_available():
        print("   No GPU available, using batch size 1")
        return 1
    
    # Get a sample batch for testing (skip this for now, we'll test with dummy data)
    
    # Test batch sizes from 1 to max_batch_size
    for batch_size in range(1, max_batch_size + 1):
        try:
            print(f"   Testing batch size: {batch_size}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Create a dummy batch
            dummy_inputs = {
                'input_ids': torch.randint(0, tokenizer.vocab_size, (batch_size, max_length)).cuda(),
                'attention_mask': torch.ones(batch_size, max_length).cuda(),
                'labels': torch.randint(0, tokenizer.vocab_size, (batch_size, max_length)).cuda()
            }
            
            # Test forward pass
            with torch.no_grad():
                outputs = model(**dummy_inputs)
                loss = outputs.loss
            
            # If we get here, batch size works
            print(f"   ✅ Batch size {batch_size} works")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   ❌ Batch size {batch_size} failed (OOM)")
                # Clear memory and return previous batch size
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                optimal_batch_size = batch_size - 1
                print(f"   🎯 Optimal batch size: {optimal_batch_size}")
                return optimal_batch_size
            else:
                raise e
    
    print(f"   🎯 Optimal batch size: {max_batch_size}")
    return max_batch_size


def validate_dataset(dataset: Dataset) -> bool:
    """
    Validate that dataset has required columns.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = ['question', 'answer']
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        print(f"   Available columns: {dataset.column_names}")
        return False
    
    return True


def tokenize_function(examples: Dict[str, Any], tokenizer, max_length: int = 512, system_prompt: str = None) -> Dict[str, Any]:
    """
    Tokenize the examples for QA training.
    
    Args:
        examples: Batch of examples (dict with lists when batched=True)
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Tokenized examples
    """
    # Handle batched examples (dict with lists)
    if isinstance(examples.get('answer'), list):
        # Batched case: examples is a dict with lists
        prompts = [format_prompt({'question': q, 'answer': a}, system_prompt) for q, a in zip(examples['question'], examples['answer'])]
    else:
        # Single example case
        prompts = [format_prompt(examples, system_prompt)]
    
    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Set labels to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Simple Hugging Face SFT Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   # Train on original dataset (answer-only) - uses OLMo-2-0425-1B by default
   python hf_sft_train.py --dataset sft_data/gsm8k/10/original.jsonl
   
   # Train on alternative COTs (answer-only)
   python hf_sft_train.py --dataset sft_data/gsm8k/10/alternative_cots_reasoning_only_decontextualized.jsonl
   
   # Auto-find optimal batch size
   python hf_sft_train.py --dataset sft_data/gsm8k/10/original.jsonl --auto-batch-size
   
   # Use a different model
   python hf_sft_train.py --dataset sft_data/gsm8k/10/original.jsonl --model gpt2
   
   # Enable mixed precision for faster training
   python hf_sft_train.py --dataset sft_data/gsm8k/10/original.jsonl --enable-mixed-precision
   
   # Resume training from checkpoint
   python hf_sft_train.py --dataset sft_data/gsm8k/10/original.jsonl --resume-from-checkpoint ./hf_sft_output/checkpoint-1000
   
   # Advanced training with custom parameters
   python hf_sft_train.py --dataset sft_data/gsm8k/10/original.jsonl \
       --epochs 5 --lr 1e-4 --auto-batch-size --max-batch-size 32 \
       --weight-decay 0.01 --gradient-clip 1.0 --lr-scheduler cosine --enable-mixed-precision
        """
    )
    
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to JSONL dataset file")
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B",
                       help="Base model to fine-tune")
    parser.add_argument("--output-dir", type=str, default="./hf_sft_output",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging-steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--experiment-name", type=str, default="hf_sft_experiment",
                       help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                       help="MLflow run name (auto-generated if not provided)")
    parser.add_argument("--auto-batch-size", action="store_true",
                       help="Automatically find maximum batch size that fits in GPU memory")
    parser.add_argument("--max-batch-size", type=int, default=16,
                       help="Maximum batch size to try when using --auto-batch-size")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Resume training from checkpoint directory")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay for optimizer")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                       help="Gradient clipping value")
    parser.add_argument("--lr-scheduler", type=str, default="linear",
                       choices=["linear", "cosine", "constant"],
                       help="Learning rate scheduler type")
    parser.add_argument("--enable-mixed-precision", action="store_true",
                       help="Enable mixed precision training (bf16/fp16)")
    parser.add_argument("--system-prompt", type=str, default=None,
                       help="System prompt to prepend to all examples (default: math problem solving prompt)")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        return
    
    print(f"🚀 Starting HF SFT Training")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {args.model}")
    print(f"   Output: {args.output_dir}")
    
    # Setup MLflow tracking
    print("📊 Setting up MLflow tracking...")
    mlflow.set_experiment(args.experiment_name)
    
    # Generate run name if not provided
    if args.run_name is None:
        dataset_name = Path(args.dataset).stem
        model_name = args.model.split('/')[-1]
        args.run_name = f"{dataset_name}_{model_name}"
    
    print(f"   Experiment: {args.experiment_name}")
    print(f"   Run: {args.run_name}")
    
    # Load dataset
    print("📊 Loading dataset...")
    datasets = load_jsonl_dataset(args.dataset)
    train_dataset = datasets['train']
    validation_dataset = datasets['validation']
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Validation: {len(validation_dataset)} examples")
    
    # Validate dataset format
    print("🔍 Validating dataset format...")
    if not validate_dataset(train_dataset):
        print("❌ Dataset validation failed!")
        return
    print("   ✅ Dataset format is valid")
    
    # Load model and tokenizer
    print("🤖 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get model's default max length
    model_max_length = getattr(tokenizer, 'model_max_length', None)
    if model_max_length is None:
        # Fallback for models that don't specify max length
        model_max_length = 2048
    
    # Cap max_length on resource-constrained systems (Mac/CPU)
    is_mac = torch.backends.mps.is_available()
    is_cpu_only = not torch.cuda.is_available() and not is_mac
    is_resource_constrained = is_mac or is_cpu_only
    
    if is_resource_constrained and model_max_length > 2048:
        print(f"   Detected resource-constrained system, capping max_length at 2048 (model default: {model_max_length})")
        model_max_length = 2048
    
    # Use model's max length if not specified by user
    if args.max_length == 512:  # Default value
        args.max_length = model_max_length
        print(f"   Using model's max length: {args.max_length}")
    else:
        print(f"   Using user-specified max length: {args.max_length}")
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"   Model moved to GPU: {torch.cuda.get_device_name()}")
    
    # Find optimal batch size if requested
    if args.auto_batch_size:
        optimal_batch_size = find_optimal_batch_size(model, tokenizer, train_dataset, args.max_length, args.max_batch_size)
        args.batch_size = optimal_batch_size
        print(f"   Using optimal batch size: {args.batch_size}")
    
    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length, args.system_prompt),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_validation_dataset = validation_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length, args.system_prompt),
        batched=True,
        remove_columns=validation_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        bf16=args.enable_mixed_precision and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=args.enable_mixed_precision and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to=None,  # Disable wandb/tensorboard for simplicity
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=args.weight_decay,
        max_grad_norm=args.gradient_clip,
        lr_scheduler_type=args.lr_scheduler,
        dataloader_num_workers=0,  # Disable multiprocessing to avoid fork warnings
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer to avoid deprecation warning
    )
    
    # Train with MLflow tracking
    print("🏋️ Starting training...")
    start_time = time.time()
    
    try:
        with mlflow.start_run(run_name=args.run_name):
            # Log parameters
            mlflow.log_params({
                "model": args.model,
                "dataset": args.dataset,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "gradient_accumulation": args.gradient_accumulation,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "gradient_clip": args.gradient_clip,
                "lr_scheduler": args.lr_scheduler,
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "model_parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            })
            
            # Train
            train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Log final metrics
            final_metrics = trainer.evaluate()
            
            # Add computed metrics
            final_metrics.update({
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics.get("train_runtime", training_time),
                "eval_perplexity": torch.exp(torch.tensor(final_metrics["eval_loss"])).item(),
                "training_time_seconds": training_time,
                "training_time_minutes": training_time / 60,
                "samples_per_second": len(train_dataset) * args.epochs / training_time,
            })
            
            # Log GPU memory if available
            if torch.cuda.is_available():
                final_metrics.update({
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "gpu_name": torch.cuda.get_device_name(),
                })
            
            mlflow.log_metrics(final_metrics)
            
            # Log model with descriptive name
            model_name = args.model.split('/')[-1]  # Get model name without org prefix
            dataset_name = Path(args.dataset).stem  # Get dataset filename without extension
            descriptive_name = f"{model_name}_{dataset_name}"
            
            mlflow.pytorch.log_model(
                pytorch_model=trainer.model,
                name=descriptive_name
            )
            
            print(f"📊 Training completed! Final metrics: {final_metrics}")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("💡 You can resume training with --resume-from-checkpoint")
        raise e
    
    # Save model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ Training completed! Model saved to: {args.output_dir}")
    print(f"📊 MLflow experiment: {args.experiment_name}")
    print(f"   Run: {args.run_name}")
    print(f"   View results: mlflow ui")
    
    # Test the model
    print("\n🧪 Testing the model...")
    try:
        # Use the trained model directly instead of loading from disk
        test_model = trainer.model
        test_tokenizer = tokenizer
        
        # Move to CPU for testing if needed
        if torch.cuda.is_available():
            test_model = test_model.cuda()
    except Exception as e:
        print(f"⚠️ Could not test model: {str(e)}")
        print("   Model saved to disk, you can test it manually")
        return
    
    # Sample test using a real example from the dataset
    test_question = "Ines had $20 in her purse. She bought 3 pounds of peaches, which are $2 per pound at the local farmers' market. How much did she have left?"
    test_prompt = f"Question: {test_question}\nAnswer:"
    
    inputs = test_tokenizer(test_prompt, return_tensors="pt")
    
    # Move inputs to same device as model
    device = next(test_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = test_model.generate(
            **inputs,
            max_length=100,
            temperature=0.3,
            do_sample=True,
            pad_token_id=test_tokenizer.eos_token_id
        )
    
    response = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test Question: {test_question}")
    print(f"Model Response: {response}")


if __name__ == "__main__":
    main() 