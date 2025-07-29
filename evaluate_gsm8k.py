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
    # Look for #### pattern
    match = re.search(r'####\s*([0-9,.-]+)', text)
    if match:
        # Clean up the number (remove commas, etc.)
        answer = match.group(1).replace(',', '').strip()
        return answer
    
    # Fallback: look for numbers at the end
    numbers = re.findall(r'[0-9,.-]+', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    
    return ""


def generate_answer(model, tokenizer, question: str, max_length: int = 512) -> str:
    """
    Generate answer for a given question using the trained model.
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        question: Question to answer
        max_length: Maximum generation length
        
    Returns:
        Generated answer text
    """
    # Format prompt
    prompt = f"Question: {question}\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract just the answer part
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the original prompt to get just the generated answer
    if "Answer:" in full_response:
        answer = full_response.split("Answer:", 1)[1].strip()
    else:
        answer = full_response
    
    return answer


def evaluate_model(model, tokenizer, test_examples: List[Dict], max_length: int = 512) -> Dict:
    """
    Evaluate model on test examples.
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        test_examples: List of test examples
        max_length: Maximum generation length
        
    Returns:
        Evaluation results dictionary
    """
    print("🔍 Running evaluation...")
    
    correct = 0
    total = len(test_examples)
    results = []
    
    for example in tqdm(test_examples, desc="Evaluating"):
        # Generate answer
        generated_answer = generate_answer(model, tokenizer, example["question"], max_length)
        
        # Extract final answer from generation
        predicted_answer = extract_final_answer(generated_answer)
        true_answer = example["final_answer"]
        
        # Check if correct (normalize for comparison)
        is_correct = normalize_number(predicted_answer) == normalize_number(true_answer)
        
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
    
    accuracy = correct / total
    
    eval_results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "detailed_results": results
    }
    
    print(f"✅ Evaluation completed!")
    print(f"   Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return eval_results


def normalize_number(num_str: str) -> str:
    """
    Normalize number string for comparison.
    
    Args:
        num_str: Number as string
        
    Returns:
        Normalized number string
    """
    if not num_str:
        return ""
    
    # Remove commas and spaces
    normalized = num_str.replace(',', '').replace(' ', '').strip()
    
    # Try to convert to float and back to handle different formats
    try:
        # Handle integers vs floats
        if '.' in normalized:
            num = float(normalized)
            # If it's actually an integer, format as int
            if num.is_integer():
                return str(int(num))
            else:
                return str(num)
        else:
            return str(int(normalized))
    except (ValueError, TypeError):
        return normalized


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
  python evaluate_gsm8k.py --model ./hf_sft_output --max-length 256 --output eval_results.json

  # Evaluate subset for quick testing
  python evaluate_gsm8k.py --model ./hf_sft_output --num-examples 100
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--num-examples", type=int, default=None,
                       help="Number of examples to evaluate (default: all)")
    parser.add_argument("--output", type=str, default="gsm8k_eval_results.json",
                       help="Output file for results")
    parser.add_argument("--experiment-name", type=str, default="gsm8k_evaluation",
                       help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                       help="MLflow run name")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    print(f"🚀 Starting GSM8K Evaluation")
    print(f"   Model: {args.model}")
    print(f"   Max Length: {args.max_length}")
    print(f"   Output: {args.output}")
    
    # Setup MLflow
    mlflow.set_experiment(args.experiment_name)
    
    if args.run_name is None:
        model_name = Path(args.model).name
        args.run_name = f"eval_{model_name}"
    
    # Load model and tokenizer
    print("🤖 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"   Model moved to GPU: {torch.cuda.get_device_name()}")
    
    # Load test dataset
    test_examples = load_gsm8k_test()
    
    # Limit examples if specified
    if args.num_examples:
        test_examples = test_examples[:args.num_examples]
        print(f"   Limited to {len(test_examples)} examples")
    
    # Run evaluation with MLflow tracking
    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_params({
            "model_path": args.model,
            "max_length": args.max_length,
            "num_examples": len(test_examples),
            "total_test_examples": len(load_gsm8k_test()) if not args.num_examples else args.num_examples
        })
        
        # Evaluate
        results = evaluate_model(model, tokenizer, test_examples, args.max_length)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": results["accuracy"],
            "correct_answers": results["correct"],
            "total_examples": results["total"]
        })
        
        # Save results
        save_results(results, args.output)
        
        print(f"📊 MLflow experiment: {args.experiment_name}")
        print(f"   Run: {args.run_name}")
        print(f"   Final Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main() 