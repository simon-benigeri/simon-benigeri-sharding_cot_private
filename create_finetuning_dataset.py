#!/usr/bin/env python3
"""
Create Finetuning Dataset

Creates JSONL datasets for finetuning from either:
1. Original dataset samples (using seed and n_samples)
2. Alternative COT datasets (combining all alternative_cots.jsonl files from a directory)

Output format:
{"split": "train", "id": "204", "question": "...", "answer": "...", "final_answer": "..."}
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Import dataset loaders
from dataset_loader import DatasetLoader
from gsm8k_loader import GSM8KLoader

# Dataset mapping
DATASET_LOADERS = {
    'gsm8k': GSM8KLoader,
    # Add other datasets here as they become available
}


def create_original_dataset(dataset_name: str, n_samples: int, seed: int, split: str = "train") -> List[Dict[str, Any]]:
    """
    Create dataset from original data samples.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'gsm8k')
        n_samples: Number of samples to include
        seed: Random seed for sampling
        split: Dataset split ('train' or 'test')
        
    Returns:
        List of examples in the required format
    """
    if dataset_name not in DATASET_LOADERS:
        available = list(DATASET_LOADERS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available datasets: {available}")
    
    # Initialize loader
    loader_class = DATASET_LOADERS[dataset_name]
    loader = loader_class()
    print(f"📊 Loading {dataset_name} dataset...")
    loader.load_dataset()
    
    # Sample data
    if split == "train":
        samples = loader.sample_train_data(n=n_samples, seed=seed)
    elif split == "test":
        samples = loader.sample_test_data(n=n_samples, seed=seed)
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
    
    # Convert to required format
    dataset = []
    for sample in samples:
        # Extract ID from example_id (e.g., "gsm8k_train_204" -> "204")
        example_id = sample.get('example_id', 'unknown')
        id_parts = example_id.split('_')
        sample_id = id_parts[-1] if len(id_parts) > 0 else example_id
        
        dataset.append({
            "split": split,
            "id": sample_id,
            "question": sample.get('question', ''),
            "answer": sample.get('answer', ''),
            "final_answer": sample.get('final_answer', '')
        })
    
    return dataset


def create_alternative_cot_dataset(alternative_cots_dir: str) -> List[Dict[str, Any]]:
    """
    Create dataset from alternative COT directory.
    
    Args:
        alternative_cots_dir: Path to directory containing alternative COT folders
        
    Returns:
        List of examples in the required format
    """
    alt_cots_path = Path(alternative_cots_dir)
    if not alt_cots_path.exists():
        raise ValueError(f"Directory not found: {alternative_cots_dir}")
    
    dataset = []
    
    # Find all example directories
    example_dirs = [d for d in alt_cots_path.iterdir() if d.is_dir()]
    print(f"📁 Found {len(example_dirs)} example directories")
    
    for example_dir in tqdm(example_dirs, desc="Processing examples"):
        # Extract ID from directory name (e.g., "gsm8k_train_204" -> "204")
        dir_name = example_dir.name
        id_parts = dir_name.split('_')
        sample_id = id_parts[-1] if len(id_parts) > 0 else dir_name
        
        # Look for alternative_cots.jsonl file
        jsonl_file = example_dir / "alternative_cots.jsonl"
        if not jsonl_file.exists():
            print(f"⚠️  No alternative_cots.jsonl found in {example_dir}")
            continue
        
        # Read the JSONL file
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        example = json.loads(line.strip())
                        
                        # Ensure all required fields are present
                        if "question" not in example:
                            example["question"] = ""
                        if "answer" not in example:
                            example["answer"] = ""
                        if "final_answer" not in example:
                            example["final_answer"] = ""
                        
                        # Create new example with consistent key ordering
                        ordered_example = {
                            "split": "train",  # Alternative COTs are always training data
                            "id": sample_id,
                            "question": example["question"],
                            "answer": example["answer"],
                            "final_answer": example["final_answer"]
                        }
                        
                        dataset.append(ordered_example)
                        
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Invalid JSON on line {line_num} in {jsonl_file}: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ Error reading {jsonl_file}: {e}")
            continue
    
    return dataset


def detect_dataset_type_from_path(alternative_cots_dir: str) -> str:
    """
    Detect the dataset type from the alternative COTs directory path.
    
    Args:
        alternative_cots_dir: Path to alternative COTs directory
        
    Returns:
        Dataset type string
    """
    path = Path(alternative_cots_dir)
    
    # Extract the last part of the path which should contain the type
    if path.name.startswith('alternative_cots_'):
        return path.name
    else:
        # Fallback: try to extract from parent directory
        parent_name = path.parent.name
        if parent_name.startswith('alternative_cots_'):
            return parent_name
        else:
            # Default fallback
            return "alternative_cots_unknown"


def save_dataset(dataset: List[Dict[str, Any]], output_file: str):
    """
    Save dataset to JSONL file.
    
    Args:
        dataset: List of examples
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved {len(dataset)} examples to {output_file}")


def get_output_path(dataset_name: str, n_samples: int, dataset_type: str) -> str:
    """
    Generate standardized output path for SFT data.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'gsm8k')
        n_samples: Number of samples (used for directory name)
        dataset_type: Type of dataset ('original', 'alternative_cots_reasoning_only_direct', etc.)
        
    Returns:
        Output file path
    """
    return f"sft_data/{dataset_name}/{n_samples}/{dataset_type}.jsonl"


def main():
    """Main function with argument parser."""
    parser = argparse.ArgumentParser(
        description="Create finetuning datasets in JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   # Create original dataset from GSM8K
   python create_finetuning_dataset.py original --dataset gsm8k --n_samples 10 --seed 42
   
   # Create alternative COT dataset
   python create_finetuning_dataset.py alternative --input sharded_data/gsm8k/alternative_cots_reasoning_only_decontextualized --dataset gsm8k --n_samples 10
   
   # Create test dataset
   python create_finetuning_dataset.py original --dataset gsm8k --n_samples 50 --seed 123 --split test

 Output Structure:
   sft_data/gsm8k/10/original.jsonl
   sft_data/gsm8k/10/alternative_cots_reasoning_only_decontextualized.jsonl
   sft_data/gsm8k/10/alternative_cots_reasoning_only_direct.jsonl

 Output Format:
   {"split": "train", "id": "204", "question": "...", "answer": "...", "final_answer": "..."}
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Dataset creation mode')
    
    # Original dataset parser
    orig_parser = subparsers.add_parser('original', help='Create dataset from original data')
    orig_parser.add_argument('--dataset', type=str, required=True,
                           choices=list(DATASET_LOADERS.keys()),
                           help='Dataset to use')
    orig_parser.add_argument('--n_samples', type=int, required=True,
                           help='Number of samples to include')
    orig_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed for sampling')
    orig_parser.add_argument('--split', type=str, default='train',
                           choices=['train', 'test'],
                           help='Dataset split')
    
    # Alternative COT dataset parser
    alt_parser = subparsers.add_parser('alternative', help='Create dataset from alternative COTs')
    alt_parser.add_argument('--input', type=str, required=True,
                          help='Directory containing alternative COT folders')
    alt_parser.add_argument('--dataset', type=str, required=True,
                          choices=list(DATASET_LOADERS.keys()),
                          help='Dataset name (e.g., gsm8k)')
    alt_parser.add_argument('--n_samples', type=int, required=True,
                          help='Number of samples (used for directory structure)')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    try:
        if args.mode == 'original':
            print(f"🚀 Creating original dataset: {args.dataset} | {args.n_samples} samples | {args.split} split")
            dataset = create_original_dataset(
                dataset_name=args.dataset,
                n_samples=args.n_samples,
                seed=args.seed,
                split=args.split
            )
            output_file = get_output_path(args.dataset, args.n_samples, "original")
        elif args.mode == 'alternative':
            print(f"🚀 Creating alternative COT dataset from: {args.input}")
            dataset = create_alternative_cot_dataset(args.input)
            dataset_type = detect_dataset_type_from_path(args.input)
            output_file = get_output_path(args.dataset, args.n_samples, dataset_type)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return
        
        # Save dataset
        save_dataset(dataset, output_file)
        
        # Print summary
        print(f"\n📊 Dataset Summary:")
        print(f"   Total examples: {len(dataset)}")
        if dataset:
            splits = {}
            for example in dataset:
                split = example.get('split', 'unknown')
                splits[split] = splits.get(split, 0) + 1
            
            for split, count in splits.items():
                print(f"   {split}: {count}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 