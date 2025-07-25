#!/usr/bin/env python3
"""
Save Alternative Reasoning Paths as COTs

Reads reasoning paths JSON from generate_reasoning_paths.py and saves 
alternative reasoning paths as alternative chain-of-thought texts.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def path_to_cot_text(path: Dict[str, Any], path_number: int) -> str:
    """Convert a single path to chain-of-thought text."""
    texts = path.get('texts', [])
    sequence = path.get('sequence', [])
    
    if not texts:
        return f"# Alternative COT {path_number}\n(No reasoning steps found)\n"
    
    cot_lines = [f"# Alternative COT {path_number}"]
    cot_lines.append(f"# Sequence: {' → '.join(sequence)}")
    cot_lines.append("")
    
    for step_text in texts:
        # Add the reasoning step text directly without enumeration
        cot_lines.append(step_text)
    
    cot_lines.append("")  # Add blank line at end
    return "\n".join(cot_lines)


def save_example_cots(example: Dict[str, Any], output_dir: Path, max_cots: int = 10) -> Dict[str, Any]:
    """Save alternative COTs for a single example."""
    example_id = example.get('example_id', 'unknown')
    
    # Handle errors
    if 'error' in example:
        return {'example_id': example_id, 'error': example['error'], 'cots_saved': 0}
    
    # Basic info
    total_paths = example.get('total_paths', 0)
    method = example.get('method', 'unknown')
    complexity = example.get('complexity_estimate', {}).get('complexity', 'unknown')
    
    if total_paths == 0:
        return {'example_id': example_id, 'error': 'No paths found', 'cots_saved': 0}
    
    # Create example directory
    example_dir = output_dir / example_id
    example_dir.mkdir(exist_ok=True)
    
    # Limit paths to save as COTs (sample if too many)
    paths = example.get('paths', [])
    if len(paths) > max_cots:
        import random
        paths_to_save = random.sample(paths, max_cots)
        sampling_note = f" (sampled {max_cots} of {len(paths)} paths)"
    else:
        paths_to_save = paths
        sampling_note = ""
    
    # Save individual COT files
    for i, path in enumerate(paths_to_save, 1):
        cot_text = path_to_cot_text(path, i)
        cot_file = example_dir / f"cot_{i:03d}.txt"
        
        with open(cot_file, 'w', encoding='utf-8') as f:
            f.write(cot_text)
    
    # Save combined file with all COTs
    combined_file = example_dir / "all_cots.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write(f"Alternative Chain-of-Thought Reasoning Paths\n")
        f.write(f"Example: {example_id}\n")
        f.write(f"Total Paths Generated: {total_paths} (method: {method}, complexity: {complexity})\n")
        f.write(f"COTs Saved: {len(paths_to_save)}{sampling_note}\n")
        f.write(f"{'='*80}\n\n")
        
        for i, path in enumerate(paths_to_save, 1):
            cot_text = path_to_cot_text(path, i)
            f.write(cot_text)
            f.write("\n" + "-"*60 + "\n\n")
    
    # Save metadata
    metadata = {
        'example_id': example_id,
        'total_paths_generated': total_paths,
        'cots_saved': len(paths_to_save),
        'sampling_applied': len(paths) > max_cots,
        'max_cots_limit': max_cots,
        'method': method,
        'complexity_estimate': example.get('complexity_estimate', {}),
        'files_created': {
            'individual_cots': [f"cot_{i:03d}.txt" for i in range(1, len(paths_to_save) + 1)],
            'combined_file': "all_cots.txt"
        }
    }
    
    metadata_file = example_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return {'example_id': example_id, 'cots_saved': len(paths_to_save), 'path': str(example_dir)}


def preview_example_cots(example: Dict[str, Any], max_preview: int = 3):
    """Preview first few COTs for a single example."""
    example_id = example.get('example_id', 'unknown')
    
    # Handle errors
    if 'error' in example:
        print(f"❌ {example_id}: {example['error']}")
        return
    
    # Basic info
    total_paths = example.get('total_paths', 0)
    method = example.get('method', 'unknown')
    complexity = example.get('complexity_estimate', {}).get('complexity', 'unknown')
    
    print(f"\n{'='*80}")
    print(f"📋 EXAMPLE: {example_id}")
    print(f"🔢 Total Paths: {total_paths} (method: {method}, complexity: {complexity})")
    print(f"{'='*80}")
    
    if total_paths == 0:
        print("❌ No paths found for this example")
        return
    
    # Preview first few COTs
    paths = example.get('paths', [])
    paths_to_preview = paths[:max_preview]
    
    for i, path in enumerate(paths_to_preview, 1):
        print(f"\n📝 PREVIEW - Alternative COT {i}:")
        print("-" * 50)
        
        texts = path.get('texts', [])
        sequence = path.get('sequence', [])
        
        print(f"Sequence: {' → '.join(sequence)}")
        print()
        
        for step_text in texts:
            print(f"  {step_text}")
    
    if len(paths) > max_preview:
        print(f"\n... ({len(paths) - max_preview} more COTs will be saved)")


def display_summary(paths_data: List[Dict[str, Any]]):
    """Display overall summary statistics."""
    total_examples = len(paths_data)
    successful = len([ex for ex in paths_data if 'error' not in ex])
    failed = total_examples - successful
    
    total_paths = sum(ex.get('total_paths', 0) for ex in paths_data)
    examples_with_multiple = len([ex for ex in paths_data if ex.get('total_paths', 0) > 1])
    
    # Method breakdown
    methods = {}
    complexities = {}
    
    for ex in paths_data:
        if 'error' not in ex:
            method = ex.get('method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
            
            complexity = ex.get('complexity_estimate', {}).get('complexity', 'unknown')
            complexities[complexity] = complexities.get(complexity, 0) + 1
    
    print("📊 SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total Examples: {total_examples}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total Reasoning Paths: {total_paths}")
    print(f"Examples with Multiple Paths: {examples_with_multiple}")
    
    if successful > 0:
        avg_paths = total_paths / successful
        print(f"Average Paths per Example: {avg_paths:.1f}")
    
    print(f"\nGeneration Methods:")
    for method, count in methods.items():
        print(f"  {method.title()}: {count}")
    
    print(f"\nComplexity Distribution:")
    for complexity, count in complexities.items():
        print(f"  {complexity}: {count}")


def save_alternative_cots(paths_file: str, output_dir: str = None, 
                         example_filter: str = None, preview_only: bool = False, max_preview: int = 3, max_cots: int = 10):
    """Save alternative COTs from reasoning paths file."""
    # Load data
    try:
        with open(paths_file, 'r') as f:
            paths_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {paths_file}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in file: {e}")
        return
    
    print(f"📁 Loading reasoning paths from: {paths_file}")
    
    # Show summary
    display_summary(paths_data)
    
    # Set output directory
    if output_dir is None:
        paths_path = Path(paths_file)
        
        # Extract segmentation style from filename
        # Expected patterns: gsm8k_pipeline_results_{segmentation}_reasoning_paths.json
        filename = paths_path.stem
        segmentation = "regular"  # default
        
        # Try to extract segmentation from filename
        if "_reasoning_paths" in filename:
            # Remove the _reasoning_paths suffix
            base_name = filename.replace("_reasoning_paths", "")
            
            # Check for segmentation patterns (check longer patterns first)
            if base_name.endswith("_reasoning_only_decontextualized"):
                segmentation = "reasoning_only_decontextualized"
            elif base_name.endswith("_reasoning_only_direct"):
                segmentation = "reasoning_only_direct"
            elif base_name.endswith("_decontextualized"):
                segmentation = "decontextualized"
            elif base_name.endswith("_consolidated"):
                segmentation = "consolidated"
            elif base_name.endswith("_direct"):
                segmentation = "direct"
            elif base_name.endswith("_regular"):
                segmentation = "regular"
            # If no specific segmentation found in filename, check for direct pipeline results
            elif "_direct_pipeline_results" in base_name:
                segmentation = "direct"
        
        # Save to sharded_data/gsm8k/alternative_cots_{segmentation}
        output_dir = Path("sharded_data/gsm8k") / f"alternative_cots_{segmentation}"
        print(f"🔍 Detected segmentation style: {segmentation}")
    else:
        output_dir = Path(output_dir)
    
    # Filter examples if requested
    if example_filter:
        filtered_data = [ex for ex in paths_data if example_filter.lower() in ex.get('example_id', '').lower()]
        if filtered_data:
            print(f"\n🔍 Filtering to {len(filtered_data)} examples containing '{example_filter}'")
            paths_data = filtered_data
        else:
            print(f"\n❌ No examples found containing '{example_filter}'")
            return
    
    if preview_only:
        print(f"\n📋 PREVIEW MODE - showing first {max_preview} COTs per example")
        # Just preview, don't save
        for example in paths_data:
            preview_example_cots(example, max_preview)
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"\n💾 Saving alternative COTs to: {output_dir}")
    
    # Save COTs for each example
    results = []
    total_cots_saved = 0
    
    for example in paths_data:
        result = save_example_cots(example, output_dir, max_cots)
        results.append(result)
        
        if 'error' not in result:
            total_cots_saved += result['cots_saved']
            # Check if this example had sampling applied
            example_paths = example.get('total_paths', 0)
            if example_paths > max_cots:
                print(f"✅ {result['example_id']}: {result['cots_saved']} COTs saved (sampled from {example_paths} paths)")
            else:
                print(f"✅ {result['example_id']}: {result['cots_saved']} COTs saved")
        else:
            print(f"❌ {result['example_id']}: {result['error']}")
    
    # Save overall summary
    summary_file = output_dir / "summary.json"
    summary = {
        'source_file': paths_file,
        'total_examples': len(paths_data),
        'successful_examples': len([r for r in results if 'error' not in r]),
        'total_cots_saved': total_cots_saved,
        'output_structure': {
            'description': 'Each example has its own directory with individual COT files and combined file',
            'individual_files': 'cot_001.txt, cot_002.txt, etc.',
            'combined_file': 'all_cots.txt',
            'metadata_file': 'metadata.json'
        },
        'results': results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎯 Summary:")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   ✅ Examples processed: {len([r for r in results if 'error' not in r])}/{len(paths_data)}")
    print(f"   📝 Total COTs saved: {total_cots_saved}")
    print(f"   📋 Summary file: {summary_file}")


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Save alternative reasoning paths as chain-of-thought texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save all alternative COTs to organized directory structure
  python display_reasoning_paths.py results_reasoning_paths.json
  
  # Preview COTs without saving
  python display_reasoning_paths.py results_reasoning_paths.json --preview-only
  
  # Save to specific output directory
  python display_reasoning_paths.py results_reasoning_paths.json --output /path/to/output
  
  # Filter to specific example
  python display_reasoning_paths.py results_reasoning_paths.json --filter train_123
  
  # Preview with more COTs shown per example
  python display_reasoning_paths.py results_reasoning_paths.json --preview-only --max-preview 5
  
  # Save up to 20 COTs per example (instead of default 10)
  python display_reasoning_paths.py results_reasoning_paths.json --max-cots 20

Output Structure:
  sharded_data/gsm8k/alternative_cots_{segmentation}/
  ├── summary.json                    # Overall summary and metadata
  ├── gsm8k_train_123/               # Example directory
  │   ├── cot_001.txt                # Individual COT files
  │   ├── cot_002.txt
  │   ├── ...
  │   ├── all_cots.txt               # All COTs combined
  │   └── metadata.json              # Example metadata
  └── gsm8k_train_456/               # Another example
      └── ...
        """
    )
    
    parser.add_argument("paths_file", help="Reasoning paths JSON file from generate_reasoning_paths.py")
    parser.add_argument("--output", type=str, 
                       help="Output directory (default: sharded_data/gsm8k/alternative_cots_{segmentation})")
    parser.add_argument("--filter", type=str, help="Filter to examples containing this string")
    parser.add_argument("--preview-only", action="store_true",
                       help="Preview COTs without saving them")
    parser.add_argument("--max-preview", type=int, default=3,
                       help="Number of COTs to preview per example (default: 3)")
    parser.add_argument("--max-cots", type=int, default=10,
                       help="Maximum COTs to save per example - randomly samples if more paths exist (default: 10)")
    
    args = parser.parse_args()
    
    save_alternative_cots(
        args.paths_file, 
        output_dir=args.output,
        example_filter=args.filter,
        preview_only=args.preview_only,
        max_preview=args.max_preview,
        max_cots=args.max_cots
    )


if __name__ == "__main__":
    main() 