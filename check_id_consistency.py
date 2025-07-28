#!/usr/bin/env python3
"""
Check ID Consistency

Check that all JSONL files in a directory have the same set of IDs.
"""

import json
import argparse
from pathlib import Path
from typing import Set, Dict, List
from collections import defaultdict


def extract_ids_from_jsonl(file_path: Path) -> Set[str]:
    """
    Extract all unique IDs from a JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Set of unique IDs
    """
    ids = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if 'id' in data:
                        ids.add(data['id'])
                    else:
                        print(f"⚠️  Line {line_num} in {file_path.name} missing 'id' key")
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON on line {line_num} in {file_path.name}: {e}")
                    continue
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return set()
    
    return ids


def check_id_consistency(directory: str) -> Dict[str, Set[str]]:
    """
    Check ID consistency across all JSONL files in a directory.
    
    Args:
        directory: Directory containing JSONL files
        
    Returns:
        Dictionary mapping filename to set of IDs
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"❌ Directory not found: {directory}")
        return {}
    
    # Find all JSONL files
    jsonl_files = list(dir_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"❌ No JSONL files found in {directory}")
        return {}
    
    print(f"📁 Found {len(jsonl_files)} JSONL files in {directory}")
    
    # Extract IDs from each file
    file_ids = {}
    for file_path in jsonl_files:
        print(f"📖 Reading {file_path.name}...")
        ids = extract_ids_from_jsonl(file_path)
        file_ids[file_path.name] = ids
        print(f"   Found {len(ids)} unique IDs")
    
    return file_ids


def analyze_id_consistency(file_ids: Dict[str, Set[str]]):
    """
    Analyze and report on ID consistency.
    
    Args:
        file_ids: Dictionary mapping filename to set of IDs
    """
    if not file_ids:
        return
    
    filenames = list(file_ids.keys())
    if len(filenames) < 2:
        print("ℹ️  Only one JSONL file found - no consistency check needed")
        return
    
    # Get all unique IDs across all files
    all_ids = set()
    for ids in file_ids.values():
        all_ids.update(ids)
    
    print(f"\n📊 ANALYSIS")
    print(f"=" * 50)
    print(f"Total unique IDs across all files: {len(all_ids)}")
    
    # Check if all files have the same set of IDs
    first_file = filenames[0]
    first_ids = file_ids[first_file]
    
    all_consistent = True
    differences = {}
    
    for filename in filenames[1:]:
        current_ids = file_ids[filename]
        
        if current_ids == first_ids:
            print(f"✅ {filename}: Same IDs as {first_file}")
        else:
            all_consistent = False
            missing = first_ids - current_ids
            extra = current_ids - first_ids
            
            differences[filename] = {
                'missing': missing,
                'extra': extra
            }
            
            print(f"❌ {filename}: Different from {first_file}")
            if missing:
                print(f"   Missing IDs: {sorted(missing)}")
            if extra:
                print(f"   Extra IDs: {sorted(extra)}")
    
    if all_consistent:
        print(f"\n🎉 ALL FILES HAVE THE SAME IDS!")
        print(f"   IDs: {sorted(first_ids)}")
    else:
        print(f"\n⚠️  INCONSISTENCIES FOUND")
        print(f"   Reference file: {first_file}")
        print(f"   Reference IDs: {sorted(first_ids)}")
        
        # Show detailed differences
        for filename, diff in differences.items():
            print(f"\n   {filename}:")
            if diff['missing']:
                print(f"     Missing: {sorted(diff['missing'])}")
            if diff['extra']:
                print(f"     Extra: {sorted(diff['extra'])}")


def main():
    """Main function with argument parser."""
    parser = argparse.ArgumentParser(
        description="Check ID consistency across JSONL files in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check consistency in sft_data/gsm8k/10/
  python check_id_consistency.py sft_data/gsm8k/10/
  
  # Check consistency in current directory
  python check_id_consistency.py .
        """
    )
    
    parser.add_argument("directory", help="Directory containing JSONL files to check")
    
    args = parser.parse_args()
    
    print(f"🔍 Checking ID consistency in: {args.directory}")
    
    # Check consistency
    file_ids = check_id_consistency(args.directory)
    
    # Analyze results
    analyze_id_consistency(file_ids)


if __name__ == "__main__":
    main() 