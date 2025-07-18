# Reasoning Paths - Simple Usage

Generate alternative reasoning paths from pipeline results.

## Quick Start

```bash
# 1. Generate paths from pipeline results  
python generate_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results.json

# 2. Analyze the generated paths (optional)
python analyze_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results_reasoning_paths.json
```

## What It Does

**`generate_reasoning_paths.py`**:
- Reads pipeline JSON (nodes, edges from reasoning graphs)
- Builds NetworkX graphs 
- Generates ALL possible reasoning orderings (or samples if too complex)
- Saves paths to `*_reasoning_paths.json` in same directory

**`analyze_reasoning_paths.py`** (optional):
- Analyzes path flexibility patterns
- Creates reports and statistics
- Outputs to `*_analysis/` directory

## Output Format

```json
{
  "example_id": "gsm8k_train_123",
  "complexity_estimate": {
    "complexity": "MEDIUM",
    "constraint_ratio": 1.43,
    "source_nodes": 1,
    "sink_nodes": 1
  },
  "method": "exhaustive",
  "total_paths": 24,
  "paths": [
    {
      "sequence": ["s1", "s2", "s3", "s4"],
      "texts": ["Step 1 text...", "Step 2 text...", ...]
    }
  ]
}
```

## Dependencies

```bash
pip install networkx>=3.0
```