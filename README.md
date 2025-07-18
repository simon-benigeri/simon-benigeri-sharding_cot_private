# Chain-of-Thought Reasoning Pipeline

## 🏗️ Architecture

```
📦 Reasoning Pipeline System
├── 🧠 reasoning_pipeline.py          # Main pipeline with multiple segmentation styles
├── 📋 dataset_loader.py              # Abstract base class for dataset loaders
├── 🔢 gsm8k_loader.py               # GSM8K dataset loader
├── 🔍 hotpot_qa_loader.py           # HotpotQA dataset loader (HuggingFace)
├── 📂 hotpot_qa_local_loader.py     # HotpotQA local JSON loader
├── 🛠️  xlam_function_calling_loader.py # XLAM function calling loader
├── 🔀 generate_reasoning_paths.py    # Generate alternative reasoning paths
├── 📊 analyze_reasoning_paths.py     # Analyze reasoning flexibility patterns
├── 💾 display_reasoning_paths.py     # Save alternative COTs from paths
├── 📁 prompts/
│   └── 📁 gsm8k/
│       ├── generate_cot.txt                    # Generate chain-of-thought
│       ├── segment_cot.txt                     # Regular segmentation
│       ├── segment_cot_consolidated.txt        # Consolidated segmentation
│       ├── segment_cot_decontextualized.txt    # Decontextualized segmentation
│       ├── segment_cot_direct.txt              # Direct answer segmentation
│       ├── create_graph.txt                    # Regular graph creation
│       ├── create_graph_decontextualized.txt   # Decontextualized graph creation
│       └── create_graph_direct.txt             # Direct graph creation
├── 📁 sharded_data/
│   └── 📁 gsm8k/
│       ├── gsm8k_pipeline_results_{style}.json        # Pipeline outputs
│       ├── gsm8k_pipeline_results_{style}_reasoning_paths.json  # Generated paths
│       └── 📁 alternative_cots_{style}/               # Alternative COT texts
└── 📄 requirements.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys
cp env_example.txt .env
# Edit .env and add:
# OPENAI_API_KEY=your_openai_key
# HF_TOKEN=your_huggingface_token (for gated datasets)
```

### 2. Basic Pipeline Usage

```bash
# Regular segmentation with CoT generation
python reasoning_pipeline.py --dataset gsm8k --n_samples 5

# Decontextualized segmentation (self-contained segments)
python reasoning_pipeline.py --dataset gsm8k --n_samples 5 --segmentation decontextualized

# Direct answer segmentation (no CoT generation, segments existing answer)
python reasoning_pipeline.py --dataset gsm8k --n_samples 5 --segmentation direct

# Alternative: Use direct pipeline explicitly
python reasoning_pipeline.py --dataset gsm8k --n_samples 5 --pipeline direct --segmentation decontextualized
```

### 3. Complete Workflow: Generate Alternative Reasoning Paths

```bash
# Step 1: Run pipeline to get reasoning graphs
python reasoning_pipeline.py --dataset gsm8k --n_samples 10 --segmentation decontextualized

# Step 2: Generate alternative reasoning paths (conservative for COTs)
python generate_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized.json --for-cots

# Step 3: Save alternative COTs as text files
python display_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized_reasoning_paths.json

# Step 4: Optional - analyze reasoning patterns
python analyze_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized_reasoning_paths.json
```

## 🎯 Reasoning Approaches

### 1. Regular Segmentation (`--segmentation regular`)
- **Default approach**: Generates CoT, then segments into micro-steps
- **Use case**: Fine-grained reasoning analysis
- **Output**: Many small atomic reasoning steps

### 2. Consolidated Segmentation (`--segmentation consolidated`)
- **Approach**: Generates CoT, then segments into substantial steps
- **Use case**: Higher-level reasoning analysis
- **Output**: Fewer, more substantial reasoning chunks

### 3. Decontextualized Segmentation (`--segmentation decontextualized`)
- **Approach**: Generates CoT, segments into self-contained steps
- **Key feature**: Each segment understandable without previous context
- **Benefits**: Better for reordering, analysis, alternative path generation
- **Rules**: No pronouns, explicit references, standalone statements

### 4. Direct Answer Segmentation (`--segmentation direct`)
- **Approach**: Directly segments existing answer (no CoT generation)
- **Benefits**: Faster, cheaper, analyzes original reasoning
- **Use case**: When you want to analyze provided step-by-step solutions
- **Output**: Decontextualized segments from original GSM8K answers

## 📊 Pipeline Workflows

### Normal Pipeline Flow
```
Question + Answer → Generate CoT → Segment CoT → Create Graph → Generate Paths → Save Alternative COTs
```

### Direct Pipeline Flow  
```
Question + Answer → Segment Answer Directly → Create Graph → Generate Paths → Save Alternative COTs
```

## 🔧 Dataset Support

### GSM8K (Grade School Math 8K)
```bash
python reasoning_pipeline.py --dataset gsm8k --n_samples 10
```

### HotpotQA (Multi-hop Question Answering)
```bash
# Using HuggingFace dataset
python reasoning_pipeline.py --dataset hotpotqa --n_samples 5

# Using local JSON file
python hotpot_qa_local_loader.py  # Load and preview data
```

### XLAM Function Calling
```bash
python xlam_function_calling_loader.py  # Load and preview data
```

## 🔀 Alternative Reasoning Path Generation

The system can generate multiple valid reasoning orderings from dependency graphs:

### Conservative Path Generation (for COTs)
```bash
# Generate fewer paths suitable for COT creation
python generate_reasoning_paths.py results.json --for-cots
```

**Limits:**
- HIGH complexity: 50 paths (vs 500)
- MEDIUM complexity: 100 paths (vs 1000)  
- LOW complexity: 200 paths (vs 10000)

### Full Analysis Mode (default)
```bash
# Generate all possible paths for analysis
python generate_reasoning_paths.py results.json
```

### Save Alternative COTs
```bash
# Save up to 10 alternative COTs per example (default)
python display_reasoning_paths.py reasoning_paths.json

# Save up to 20 COTs per example
python display_reasoning_paths.py reasoning_paths.json --max-cots 20

# Preview without saving
python display_reasoning_paths.py reasoning_paths.json --preview-only
```

**Output Structure:**
```
sharded_data/gsm8k/alternative_cots_decontextualized/
├── summary.json                    # Overall summary
├── gsm8k_train_123/               # Per-example directories
│   ├── cot_001.txt                # Individual alternative COTs
│   ├── cot_002.txt
│   ├── all_cots.txt               # All COTs combined
│   └── metadata.json              # Example metadata
└── gsm8k_train_456/
    └── ...
```

## 📋 Output Formats

### Pipeline Results
```json
{
  "example_id": "gsm8k_train_123",
  "question": "...",
  "answer": "...", 
  "final_answer": "42",
  "cot": "...",
  "segments": [...],
  "required_segments": [...],
  "nodes": [...],
  "edges": [...],
  "success": true,
  "pipeline_type": "direct_answer_segmentation"
}
```

### Reasoning Paths
```json
{
  "example_id": "gsm8k_train_123",
  "complexity_estimate": {
    "complexity": "MEDIUM",
    "constraint_ratio": 1.43,
    "estimated_paths": "10-1,000"
  },
  "method": "exhaustive",
  "total_paths": 24,
  "paths": [
    {
      "sequence": ["s1", "s2", "s3"],
      "texts": ["Step 1 text...", "Step 2 text...", "Step 3 text..."]
    }
  ]
}
```

### Alternative COT Files
```
# Alternative COT 1
# Sequence: s1 → s3 → s2 → s4

Betty picked 16 strawberries
Betty, Matthew, and Natalie have 16 + 36 + 18 = 70 strawberries in total
Matthew picked 16 + 20 = 36 strawberries
Betty, Matthew, and Natalie can make 70/7 = 10 jars of strawberries
```

## 🎛️ Command Line Options

### Pipeline Options
```bash
python reasoning_pipeline.py \
  --dataset gsm8k \
  --n_samples 10 \
  --segmentation decontextualized \
  --pipeline normal \
  --model gpt-4o-mini-2024-07-18 \
  --delay 1.0 \
  --seed 42
```

### Path Generation Options
```bash
python generate_reasoning_paths.py input.json \
  --output custom_output.json \
  --for-cots  # Use conservative limits
```

### COT Saving Options
```bash
python display_reasoning_paths.py paths.json \
  --output custom_dir \
  --max-cots 20 \
  --filter train_123 \
  --preview-only
```

## 🔧 Adding New Datasets

### 1. Create Dataset Loader
```python
from dataset_loader import DatasetLoader

class CustomDatasetLoader(DatasetLoader):
    def load_data(self, num_samples=None):
        # Return list with: example_id, question, answer, final_answer
        pass
    
    def get_problem_text(self, sample):
        return sample['question']
    
    def get_solution(self, sample):
        return sample['answer']
```

### 2. Add to Pipeline
```python
# In reasoning_pipeline.py DATASET_LOADERS
DATASET_LOADERS = {
    'gsm8k': GSM8KLoader,
    'custom': CustomDatasetLoader,
}
```

### 3. Create Prompts
Create `prompts/custom/` directory with required prompt files.

## 🧪 Testing and Examples

### Test Basic Pipeline
```bash
python reasoning_pipeline.py --dataset gsm8k --n_samples 1
```

### Test Alternative Path Generation
```bash
# Generate small test set
python reasoning_pipeline.py --dataset gsm8k --n_samples 3 --segmentation decontextualized

# Generate conservative paths
python generate_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized.json --for-cots

# Save as COTs
python display_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized_reasoning_paths.json --max-cots 5
```

### Preview Dataset
```bash
python gsm8k_loader.py                    # Preview GSM8K
python hotpot_qa_local_loader.py          # Preview local HotpotQA
python xlam_function_calling_loader.py    # Preview XLAM
```

## ⚙️ Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key (required)
- `HF_TOKEN`: HuggingFace token for gated datasets (optional)

### Key Parameters
- **Segmentation styles**: `regular`, `consolidated`, `decontextualized`, `direct`
- **Pipeline types**: `normal` (generates CoT), `direct` (segments existing answer)
- **Path generation**: `--for-cots` for conservative limits
- **COT limits**: `--max-cots N` for manageable alternative sets

## 📁 Key Files

### Core Pipeline
- `reasoning_pipeline.py`: Main reasoning pipeline with multiple approaches
- `generate_reasoning_paths.py`: Generate alternative reasoning orderings
- `display_reasoning_paths.py`: Save alternative COTs as text files
- `analyze_reasoning_paths.py`: Analyze reasoning flexibility patterns

### Dataset Loaders
- `dataset_loader.py`: Abstract base class
- `gsm8k_loader.py`: GSM8K (HuggingFace)
- `hotpot_qa_loader.py`: HotpotQA (HuggingFace) 
- `hotpot_qa_local_loader.py`: HotpotQA (local JSON)
- `xlam_function_calling_loader.py`: XLAM function calling dataset

### Prompts
- `prompts/gsm8k/`: Complete set of reasoning prompts for different styles
- Support for custom dataset prompts in `prompts/{dataset_name}/`
