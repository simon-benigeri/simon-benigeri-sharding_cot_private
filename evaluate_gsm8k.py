#!/usr/bin/env python3
"""
GSM8K Evaluation Script

Evaluates trained models on GSM8K test set by:
1. Loading GSM8K test dataset
2. Running inference to generate answers
3. Extracting final numerical answers (after #### pattern)
4. Computing accuracy against ground truth
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Answer extraction regex - matches numbers after ####
ANS_RE = re.compile(r"#### (\-?[0-9,]+\.?[0-9]*)\s*$", re.MULTILINE)
INVALID_ANS = "[invalid]"


def load_gsm8k_test() -> List[Dict]:
    """
    Load GSM8K test dataset.
    
    Returns:
        List of test examples with question and answer
    """
    print("📊 Loading GSM8K test dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Convert to our format
    examples = []
    for item in dataset:
        examples.append({
            "question": item["question"],
            "answer": item["answer"],
            "final_answer": extract_final_answer(item["answer"])
        })
    
    print(f"   Loaded {len(examples)} test examples")
    return examples


def extract_final_answer(text: str) -> str:
    """
    Extract final numerical answer from text using #### pattern.
    
    Args:
        text: Text containing answer (e.g., "Step 1...\n#### 42")
        
    Returns:
        Final numerical answer as string
    """
    match = ANS_RE.search(text)
    if match:
        answer = match.group(1).strip().replace(',', '')
        return answer
    return INVALID_ANS


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 256, system_prompt: str = None) -> str:
    """
    Generate answer for a given question using the trained model.
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        question: Question to answer
        max_new_tokens: Maximum new tokens to generate
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Generated answer text
    """
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that solves math problems step by step. Always end your answer with '#### [number]' where [number] is the final numerical answer."
    
    # Format prompt to match training format exactly
    if system_prompt:
        prompt = f"{system_prompt}\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        add_special_tokens=True,
        truncation=True,
        max_length=512
    )
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (the generated answer)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return answer.strip()


def evaluate_model(model, tokenizer, test_examples: List[Dict], max_new_tokens: int = 256, system_prompt: str = None) -> Dict:
    """
    Evaluate model on test examples.
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        test_examples: List of test examples
        max_new_tokens: Maximum new tokens to generate
        system_prompt: Optional system prompt to use
        
    Returns:
        Evaluation results dictionary
    """
    print("🔍 Running evaluation...")
    
    correct = 0
    total = len(test_examples)
    results = []
    
    for example in tqdm(test_examples, desc="Evaluating"):
        # Generate answer
        generated_answer = generate_answer(model, tokenizer, example["question"], max_new_tokens, system_prompt)
        
        # Extract final answer from generation
        predicted_answer = extract_final_answer(generated_answer)
        true_answer = example["final_answer"]
        
        # Check if correct
        is_correct = (predicted_answer != INVALID_ANS and 
                     normalize_number(predicted_answer) == normalize_number(true_answer))
        
        if is_correct:
            correct += 1
        
        # Store result
        results.append({
            "question": example["question"],
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "generated_text": generated_answer,
            "correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }


def normalize_number(num_str: str) -> str:
    """
    Normalize number string for comparison.
    
    Args:
        num_str: Number as string
        
    Returns:
        Normalized number string
    """
    if not num_str or num_str == INVALID_ANS:
        return ""
    
    # Remove commas and spaces
    normalized = num_str.replace(',', '').replace(' ', '').strip()
    
    # Try to convert to float and back to handle different formats
    try:
        num = float(normalized)
        
        # Handle special cases
        if num == 0:
            return "0"
        
        # Handle integers vs floats
        if num.is_integer():
            return str(int(num))
        else:
            # For decimals, limit to reasonable precision
            return f"{num:.6f}".rstrip('0').rstrip('.')
            
    except (ValueError, TypeError):
        # If we can't convert to float, return empty string
        return ""


def save_results(results: Dict, output_file: str):
    """
    Save evaluation results to file.
    
    Args:
        results: Evaluation results
        output_file: Output file path
    """
    print(f"💾 Saving results to {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on GSM8K test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate trained model
  python evaluate_gsm8k.py --model ./hf_sft_output

  # Evaluate with custom settings
  python evaluate_gsm8k.py --model ./hf_sft_output --max-new-tokens 128 --output eval_results.json

  # Evaluate subset for quick testing
  python evaluate_gsm8k.py --model ./hf_sft_output --num-examples 100
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                       help="Maximum new tokens to generate")
    parser.add_argument("--num-examples", type=int, default=None,
                       help="Number of examples to evaluate (default: all)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (auto-generated if not provided)")
    parser.add_argument("--experiment-name", type=str, default="gsm8k_evaluation",
                       help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                       help="MLflow run name")
    parser.add_argument("--system-prompt", type=str, default=None,
                       help="System prompt to use during evaluation (default: math problem solving prompt)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    # Auto-generate output path if not provided
    if args.output is None:
        # Extract model directory name and create eval directory structure
        model_path = Path(args.model)
        model_dir_name = model_path.name
        
        # Create eval directory structure: ./eval/model_name/
        eval_dir = Path("./eval") / model_dir_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with number of examples
        if args.num_examples:
            filename = f"gsm8k_eval_{args.num_examples}examples.json"
        else:
            filename = "gsm8k_eval_full.json"
        
        args.output = str(eval_dir / filename)
    
    print(f"🚀 Starting GSM8K evaluation")
    print(f"   Model: {args.model}")
    print(f"   Output: {args.output}")
    print(f"   Max new tokens: {args.max_new_tokens}")
    
    # Load test data
    test_examples = load_gsm8k_test()
    
    # Limit examples if specified
    if args.num_examples:
        test_examples = test_examples[:args.num_examples]
        print(f"   Evaluating on {len(test_examples)} examples")
    
    # Load model
    print(f"🤖 Loading model from {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        print(f"   ✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Start MLflow tracking
    mlflow.set_experiment(args.experiment_name)
    
    # Generate run name
    if args.run_name is None:
        model_name = Path(args.model).name
        num_examples_str = f"{args.num_examples}ex" if args.num_examples else "full"
        args.run_name = f"eval_{model_name}_{num_examples_str}"
    
    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_param("model_path", args.model)
        mlflow.log_param("max_new_tokens", args.max_new_tokens)
        mlflow.log_param("num_examples", args.num_examples or len(test_examples))
        
        # Run evaluation
        eval_results = evaluate_model(model, tokenizer, test_examples, args.max_new_tokens, args.system_prompt)
        
        # Log metrics
        mlflow.log_metric("accuracy", eval_results["accuracy"])
        mlflow.log_metric("correct", eval_results["correct"])
        mlflow.log_metric("total", eval_results["total"])
        
        # Print results
        print(f"🎯 Evaluation Results:")
        print(f"   Accuracy: {eval_results['accuracy']:.4f}")
        print(f"   Correct: {eval_results['correct']}/{eval_results['total']}")
        
        # Save results
        save_results(eval_results, args.output)
        
        # Log results file as artifact
        mlflow.log_artifact(args.output)
        
        print(f"✅ Evaluation complete!")


if __name__ == "__main__":
    main() 