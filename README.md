# Chain-of-Thought Reasoning & Fine-Tuning Pipeline

## 🏗️ Architecture

```
📦 Complete Reasoning & SFT Pipeline
├── 🧠 reasoning_pipeline.py          # Main pipeline: segments CoT and creates DAGs
├── 📋 dataset_loader.py              # Abstract base class for dataset loaders
├── 🔢 gsm8k_loader.py               # GSM8K dataset loader
├── 🔍 hotpot_qa_loader.py           # HotpotQA dataset loader (HuggingFace)
├── 📂 hotpot_qa_local_loader.py     # HotpotQA local JSON loader
├── 🛠️  xlam_function_calling_loader.py # XLAM function calling loader
├── 🔀 generate_reasoning_paths.py    # Generate alternative reasoning paths
├── 📄 generate_alternative_cots.py   # Generate alternative CoT texts from paths
├── 📊 create_finetuning_dataset.py   # Create SFT datasets in JSONL format
├── 🔧 check_id_consistency.py       # Verify ID consistency across datasets
├── 🚀 hf_sft_train.py               # HuggingFace SFT training script
├── 📝 axolotl_config.yml            # Axolotl configuration for SFT
├── 📊 evaluate_gsm8k.py             # GSM8K evaluation script
├── 📊 analyze_reasoning_paths.py     # Analyze reasoning flexibility patterns
├── 📁 prompts/
│   └── 📁 gsm8k/
│       ├── generate_cot.txt                    # Generate chain-of-thought
│       ├── segment_cot.txt                     # Regular segmentation
│       ├── segment_cot_consolidated.txt        # Consolidated segmentation
│       ├── segment_cot_decontextualized.txt    # Decontextualized segmentation
│       ├── segment_cot_direct.txt              # Direct answer segmentation
│       ├── segment_cot_reasoning_only_decontextualized.txt  # Reasoning-only decontextualized
│       ├── segment_cot_reasoning_only_direct.txt            # Reasoning-only direct
│       ├── create_graph.txt                    # Regular graph creation
│       ├── create_graph_decontextualized.txt   # Decontextualized graph creation
│       ├── create_graph_direct.txt             # Direct graph creation
│       └── extract_outputs.py                  # Utility to extract outputs from LLM responses
├── 📁 sharded_data/
│   └── 📁 gsm8k/
│       ├── gsm8k_pipeline_results_{style}.json        # Pipeline outputs
│       ├── gsm8k_pipeline_results_{style}_reasoning_paths.json  # Generated paths
│       └── 📁 alternative_cots_{style}/               # Alternative CoT texts
├── 📁 sft_data/
│   └── 📁 gsm8k/
│       └── 📁 {n_samples}/
│           ├── original.jsonl                         # Original GSM8K data
│           ├── alternative_cots_consolidated.jsonl    # Consolidated alternative CoTs
│           ├── alternative_cots_decontextualized.jsonl # Decontextualized alternative CoTs
│           ├── alternative_cots_direct.jsonl          # Direct alternative CoTs
│           ├── alternative_cots_reasoning_only_decontextualized.jsonl
│           └── alternative_cots_reasoning_only_direct.jsonl
└── 📄 requirements.txt / requirements_hf.txt
```

## 🚀 Complete Pipeline Workflow

### **End-to-End: From Raw Data to Trained Model**

```bash
# 1️⃣ Generate reasoning graphs and segments
python reasoning_pipeline.py --dataset gsm8k --n_samples 10 --segmentation decontextualized

# 2️⃣ Generate alternative reasoning paths
python generate_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized.json --for-cots

# 3️⃣ Generate alternative CoT texts
python generate_alternative_cots.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized_reasoning_paths.json

# 4️⃣ Create fine-tuning datasets
python create_finetuning_dataset.py --dataset gsm8k --n_samples 10 --type original
python create_finetuning_dataset.py --dataset gsm8k --n_samples 10 --type alternative_cots_decontextualized

# 5️⃣ Verify data consistency (optional)
python check_id_consistency.py sft_data/gsm8k/10

# 6️⃣ Train model (choose one approach)
# Option A: HuggingFace Transformers
python hf_sft_train.py --dataset sft_data/gsm8k/10/original.jsonl --epochs 3 --auto-batch-size

# Option B: Axolotl
axolotl train axolotl_config.yml

# 7️⃣ Evaluate trained model
python evaluate_gsm8k.py --model ./hf_sft_output --num-examples 100
```

## 🎯 Pipeline Components

### **1. Reasoning Pipeline** (`reasoning_pipeline.py`)
- **Purpose**: Segments Chain-of-Thought reasoning and creates dependency graphs (DAGs)
- **Input**: Raw dataset (GSM8K, HotpotQA, etc.)
- **Output**: Segmented reasoning steps with dependency relationships
- **Key Features**: Multiple segmentation styles, graph creation, progress tracking

### **2. Generate Reasoning Paths** (`generate_reasoning_paths.py`) 
- **Purpose**: Generate multiple valid reasoning orderings from dependency graphs
- **Input**: Pipeline results with DAGs
- **Output**: Alternative reasoning sequences respecting dependencies
- **Key Features**: Conservative mode for CoTs, complexity estimation

### **3. Generate Alternative CoTs** (`generate_alternative_cots.py`)
- **Purpose**: Convert reasoning paths into human-readable Chain-of-Thought texts
- **Input**: Reasoning paths with sequences
- **Output**: Alternative CoT texts saved as individual files and JSONL
- **Key Features**: Progress tracking, JSONL format for training

### **4. Create Fine-tuning Dataset** (`create_finetuning_dataset.py`)
- **Purpose**: Prepare training datasets in standardized JSONL format
- **Input**: Original data or alternative CoT directories
- **Output**: Consistent JSONL files for SFT
- **Key Features**: Original & alternative CoT support, consistent formatting

### **5. Training Options**

#### **Option A: HuggingFace SFT** (`hf_sft_train.py`)
- **Purpose**: Direct SFT using HuggingFace Transformers
- **Features**: Auto batch size, MLflow tracking, validation split, system prompts
- **Benefits**: Full control, extensive logging, robustness features

#### **Option B: Axolotl** (`axolotl_config.yml`)
- **Purpose**: Simplified SFT using Axolotl framework
- **Features**: YAML configuration, built-in best practices
- **Benefits**: Easier setup, proven configurations

### **6. Evaluation** (`evaluate_gsm8k.py`)
- **Purpose**: Evaluate trained models on GSM8K test set
- **Features**: Accurate answer extraction, MLflow logging, system prompts
- **Output**: Accuracy metrics, detailed results, per-example analysis

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt        # For reasoning pipeline
pip install -r requirements_hf.txt     # For HuggingFace SFT

# Create .env file with API keys
cp env_example.txt .env
# Edit .env and add:
# OPENAI_API_KEY=your_openai_key
# HF_TOKEN=your_huggingface_token (for gated datasets)
```

### 2. Quick Test Run

```bash
# Test the complete pipeline with a small dataset
python reasoning_pipeline.py --dataset gsm8k --n_samples 3 --segmentation decontextualized
python generate_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized.json --for-cots
python generate_alternative_cots.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized_reasoning_paths.json
python create_finetuning_dataset.py --dataset gsm8k --n_samples 3 --type original
python hf_sft_train.py --dataset sft_data/gsm8k/3/original.jsonl --epochs 1 --max-length 256
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

### 5. Reasoning-Only Variants
- **Purpose**: Focus only on reasoning steps, exclude facts from question
- **Types**: `reasoning_only_decontextualized`, `reasoning_only_direct`
- **Benefits**: Cleaner reasoning chains for training

## 📝 Prompt System

### **Prompt Directory Structure**
```
📁 prompts/
└── 📁 gsm8k/
    ├── generate_cot.txt                           # Generate initial chain-of-thought
    ├── segment_cot.txt                            # Regular segmentation (micro-steps)
    ├── segment_cot_consolidated.txt               # Consolidated segmentation (substantial steps)
    ├── segment_cot_decontextualized.txt           # Decontextualized segmentation (self-contained)
    ├── segment_cot_direct.txt                     # Direct answer segmentation (no CoT generation)
    ├── segment_cot_reasoning_only_decontextualized.txt  # Reasoning-only decontextualized
    ├── segment_cot_reasoning_only_direct.txt            # Reasoning-only direct
    ├── create_graph.txt                           # Regular dependency graph creation
    ├── create_graph_decontextualized.txt          # Decontextualized graph creation
    └── create_graph_direct.txt                    # Direct graph creation
```

### **Prompt Types & Usage**

#### **1. CoT Generation** (`generate_cot.txt`)
- **Purpose**: Generate initial chain-of-thought reasoning from question + answer
- **Input**: Question and final answer
- **Output**: Step-by-step reasoning chain
- **Used by**: Normal pipeline when generating CoT from scratch

#### **2. Segmentation Prompts**
All segmentation prompts take a reasoning chain and break it into discrete steps:

**`segment_cot.txt`** - Regular segmentation
- Creates many small, atomic reasoning steps
- Fine-grained analysis suitable for detailed dependency mapping

**`segment_cot_consolidated.txt`** - Consolidated segmentation  
- Creates fewer, more substantial reasoning chunks
- Higher-level reasoning analysis

**`segment_cot_decontextualized.txt`** - Decontextualized segmentation
- Each segment is self-contained and understandable without context
- No pronouns, explicit references, standalone statements
- **Best for alternative path generation**

**`segment_cot_direct.txt`** - Direct answer segmentation
- Segments existing answer without generating new CoT
- Faster and cheaper for analyzing provided solutions

**`segment_cot_reasoning_only_*.txt`** - Reasoning-only variants
- Focus only on reasoning steps, exclude facts from question
- Cleaner reasoning chains for training

#### **3. Graph Creation Prompts**
Convert segmented steps into dependency graphs (DAGs):

**`create_graph.txt`** - Regular graph creation
- Creates dependency relationships between regular segments

**`create_graph_decontextualized.txt`** - Decontextualized graph creation
- Optimized for self-contained segments
- Better dependency detection for standalone statements

**`create_graph_direct.txt`** - Direct graph creation
- Works with directly segmented answers

### **4. Utility Scripts**

**`extract_outputs.py`** - Output extraction utility
- Parses and extracts structured outputs from LLM responses
- Handles JSON parsing, error recovery, and format validation
- Used internally by the reasoning pipeline for robust response processing

### **How Prompts Are Selected**

The reasoning pipeline automatically selects prompts based on your arguments:

```python
# Example: Decontextualized segmentation
python reasoning_pipeline.py --dataset gsm8k --segmentation decontextualized

# Uses these prompts:
# 1. generate_cot.txt (if --pipeline normal)
# 2. segment_cot_decontextualized.txt
# 3. create_graph_decontextualized.txt
```

```python
# Example: Direct pipeline with reasoning-only
python reasoning_pipeline.py --dataset gsm8k --pipeline direct --segmentation reasoning_only_direct

# Uses these prompts:
# 1. segment_cot_reasoning_only_direct.txt (skips CoT generation)
# 2. create_graph_direct.txt
```

### **Prompt Selection Logic**
```python
# In reasoning_pipeline.py
segmentation_prompt = f"segment_cot{style_suffix}.txt"
graph_prompt = f"create_graph{graph_suffix}.txt"

# Where:
# style_suffix = "" | "_consolidated" | "_decontextualized" | "_direct" | "_reasoning_only_decontextualized" | "_reasoning_only_direct"
# graph_suffix = "" | "_decontextualized" | "_direct"
```

### **Adding Custom Prompts**

To add support for a new dataset:

1. **Create dataset directory:**
   ```bash
   mkdir prompts/my_dataset
   ```

2. **Add required prompt files:**
   ```bash
   # Minimum required prompts
   prompts/my_dataset/generate_cot.txt
   prompts/my_dataset/segment_cot.txt
   prompts/my_dataset/create_graph.txt
   
   # Optional: Add variants for different segmentation styles
   prompts/my_dataset/segment_cot_decontextualized.txt
   prompts/my_dataset/create_graph_decontextualized.txt
   ```

3. **Update dataset loader** to use your dataset name

### **Prompt Engineering Guidelines**

For effective prompts in this system:

1. **Be specific about output format** - especially for segmentation
2. **Include examples** - show desired input/output patterns  
3. **Handle edge cases** - empty inputs, malformed data
4. **Consistency** - use same terminology across related prompts
5. **Error handling** - graceful degradation for unexpected inputs

### **Example Prompt Structure**
```
You are an expert at analyzing mathematical reasoning...

TASK: Segment this chain-of-thought into discrete, self-contained steps.

RULES:
1. Each segment must be understandable without previous context
2. No pronouns (replace with specific nouns)
3. Include explicit numerical values
4. One logical step per segment

INPUT:
{reasoning_chain}

OUTPUT FORMAT:
[List of segments with IDs and texts]
```

## 📊 Fine-Tuning Options

### **HuggingFace SFT Training**

```bash
# Basic training
python hf_sft_train.py --dataset sft_data/gsm8k/100/original.jsonl

# Advanced training with all features
python hf_sft_train.py \
  --dataset sft_data/gsm8k/1000/alternative_cots_decontextualized.jsonl \
  --model allenai/OLMo-2-0425-1B \
  --epochs 3 \
  --lr 2e-5 \
  --auto-batch-size \
  --max-batch-size 32 \
  --enable-mixed-precision \
  --experiment-name "gsm8k_alternative_cots" \
  --system-prompt "You are a math tutor. Show your work step by step."
```

**Key Features:**
- **Auto batch size**: Automatically finds optimal batch size for available GPU 
- **MLflow tracking**: Logs metrics, parameters, and model artifacts
- **System prompts**: Consistent formatting across training and evaluation
- **Validation split**: Automatic 10% validation split
- **Checkpoints**: Resume from checkpoint

### **Axolotl Training**

```bash
# Install Axolotl (optional)
pip install "axolotl[flash-attn,deepspeed] @ git+https://github.com/OpenAccess-AI-Collective/axolotl.git"

# Edit axolotl_config.yml to point to your data
# datasets:
#   - path: sft_data/gsm8k/1000/alternative_cots_decontextualized.jsonl

# Train model
axolotl train axolotl_config.yml
```

## 🔧 Dataset Support

### GSM8K (Grade School Math 8K)
```bash
python reasoning_pipeline.py --dataset gsm8k --n_samples 10
```

### HotpotQA (Multi-hop Question Answering) (TODO)
```bash
# Using HuggingFace dataset
python reasoning_pipeline.py --dataset hotpotqa --n_samples 5

# Using local JSON file
python hotpot_qa_local_loader.py  # Load and preview data
```

### XLAM Function Calling (TODO)
```bash
python xlam_function_calling_loader.py  # Load and preview data
```

## 📋 Output Formats

### **SFT Training Data Format**
```json
{
  "split": "train",
  "id": "gsm8k_train_123",
  "question": "Betty picked 16 strawberries. Matthew picked 20 more strawberries than Betty and Natalie picked 18 strawberries. How many jars of 7 strawberries each can they make?",
  "answer": "Matthew picked 16 + 20 = 36 strawberries.\nBetty, Matthew, and Natalie have 16 + 36 + 18 = 70 strawberries in total.\nBetty, Matthew, and Natalie can make 70/7 = 10 jars of strawberries.\n#### 10",
  "final_answer": "10"
}
```

### **Training Format with System Prompt**
```
You are a helpful assistant that solves math problems step by step. Always end your answer with '#### [number]' where [number] is the final numerical answer.
Question: Betty picked 16 strawberries. Matthew picked 20 more strawberries than Betty and Natalie picked 18 strawberries. How many jars of 7 strawberries each can they make?
Answer: Matthew picked 16 + 20 = 36 strawberries.
Betty, Matthew, and Natalie have 16 + 36 + 18 = 70 strawberries in total.
Betty, Matthew, and Natalie can make 70/7 = 10 jars of strawberries.
#### 10
```

## 🔍 Evaluation & Analysis

### **Model Evaluation**

```bash
# Basic evaluation
python evaluate_gsm8k.py --model ./hf_sft_output

# Advanced evaluation with custom settings
python evaluate_gsm8k.py \
  --model ./hf_sft_output \
  --num-examples 500 \
  --max-new-tokens 512 \
  --experiment-name "gsm8k_eval_decontextualized" \
  --system-prompt "You are a math tutor. Show your work step by step."

# Evaluate Axolotl model
python evaluate_gsm8k.py --model ./outputs/{run_name}
```


### **Alternative CoT Analysis**

```bash
# Analyze reasoning flexibility patterns
python analyze_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized_reasoning_paths.json

# Check dataset consistency
python check_id_consistency.py sft_data/gsm8k/100
```

## 🎛️ Command Line Options

### **Reasoning Pipeline**
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

### **HuggingFace SFT Training**
```bash
python hf_sft_train.py \
  --dataset sft_data/gsm8k/1000/original.jsonl \
  --model allenai/OLMo-2-0425-1B \
  --output-dir ./custom_output \
  --epochs 3 \
  --lr 2e-5 \
  --batch-size 4 \
  --max-length 2048 \
  --auto-batch-size \
  --enable-mixed-precision \
  --system-prompt "Custom system prompt" \
  --experiment-name "my_experiment" \
  --weight-decay 0.01 \
  --gradient-clip 1.0 \
  --lr-scheduler cosine
```

### **Evaluation**
```bash
python evaluate_gsm8k.py \
  --model ./hf_sft_output \
  --max-new-tokens 256 \
  --num-examples 10 \
  --output ./custom_eval_results \
  --experiment-name "gsm8k_evaluation" \
  --system-prompt "Custom evaluation prompt"
```

## 📁 Key Scripts Reference

### **Core Pipeline Scripts**
- `reasoning_pipeline.py`: Main reasoning pipeline to segment COT and perform COT -> DAG
- `generate_reasoning_paths.py`: Generate alternative reasoning orderings from the DAG with topological sort
- `generate_alternative_cots.py`: Convert paths to CoT texts with JSONL output
- `create_finetuning_dataset.py`: Prepare standardized SFT datasets
- `check_id_consistency.py`: Verify dataset ID consistency

### **Fine-Tuning Scripts**
- `hf_sft_train.py`: HuggingFace SFT with advanced features
- `axolotl_config.yml`: Axolotl configuration for simplified SFT
- `evaluate_gsm8k.py`: GSM8K evaluation with robust answer extraction

### **Dataset Loaders**
- `dataset_loader.py`: Abstract base class
- `gsm8k_loader.py`: GSM8K (HuggingFace)
- `hotpot_qa_loader.py`: HotpotQA (HuggingFace) 
- `hotpot_qa_local_loader.py`: HotpotQA (local JSON)
- `xlam_function_calling_loader.py`: XLAM function calling dataset

### **Analysis & Utilities**
- `analyze_reasoning_paths.py`: Analyze reasoning flexibility patterns
- `test_answer_extraction.py`: Test answer extraction functions

### **Prompts & Templates**
- `prompts/gsm8k/generate_cot.txt`: Generate initial chain-of-thought reasoning
- `prompts/gsm8k/segment_cot*.txt`: Various segmentation styles (regular, consolidated, decontextualized, direct, reasoning-only)
- `prompts/gsm8k/create_graph*.txt`: Dependency graph creation for different segmentation types
- `prompts/gsm8k/extract_outputs.py`: Utility for parsing and extracting LLM responses
- `prompts/{dataset}/`: Template structure for adding new datasets

## ⚙️ Configuration

### **Environment Variables**
- `OPENAI_API_KEY`: OpenAI API key (required for reasoning pipeline)
- `HF_TOKEN`: HuggingFace token for gated datasets (optional)

### **System Requirements**
- **CPU training**: Supported (automatic detection)
- **Apple Silicon (MPS)**: Full support with optimizations
- **CUDA GPUs**: Recommended for large-scale training
- **Memory**: Auto batch size selection based on available memory

### **Key Configuration Files**
- `.env`: API keys and environment variables
- `requirements.txt`: Core pipeline dependencies
- `requirements_hf.txt`: HuggingFace SFT dependencies
- `axolotl_config.yml`: Axolotl training configuration

## 🧪 Testing & Debugging

### **Quick Tests**
```bash
# Test reasoning pipeline (1 sample)
python reasoning_pipeline.py --dataset gsm8k --n_samples 1

# Test full pipeline (3 samples)
python reasoning_pipeline.py --dataset gsm8k --n_samples 3 --segmentation decontextualized
python generate_reasoning_paths.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized.json --for-cots
python generate_alternative_cots.py sharded_data/gsm8k/gsm8k_pipeline_results_decontextualized_reasoning_paths.json

# Test dataset creation
python create_finetuning_dataset.py --dataset gsm8k --n_samples 3 --type original

# Test training (short run)
python hf_sft_train.py --dataset sft_data/gsm8k/3/original.jsonl --epochs 1 --max-length 256

# Test evaluation
python evaluate_gsm8k.py --model gpt2 --num-examples 3  # Quick test with base model
```

### **Debugging Tips**
- **MLflow conflicts**: `rm -rf mlruns/` for clean restart
- **Memory issues**: Use `--auto-batch-size` and smaller `--max-length`
- **Data format issues**: Use `check_id_consistency.py` to verify datasets
- **Answer extraction**: Test with `test_answer_extraction.py`

## 🔮 Advanced Usage



### **Multi-Style Training**
```bash
# Create datasets for different reasoning styles
python create_finetuning_dataset.py --dataset gsm8k --n_samples 1000 --type original
python create_finetuning_dataset.py --dataset gsm8k --n_samples 1000 --type alternative_cots_decontextualized
python create_finetuning_dataset.py --dataset gsm8k --n_samples 1000 --type alternative_cots_direct

# Train separate models for comparison
python hf_sft_train.py --dataset sft_data/gsm8k/1000/original.jsonl --experiment-name "baseline"
python hf_sft_train.py --dataset sft_data/gsm8k/1000/alternative_cots_decontextualized.jsonl --experiment-name "decontextualized"
```

### **MLflow Experiment Tracking**
```bash
# View MLflow UI
mlflow ui

# Access at http://localhost:5000 to view:
# - Training metrics (loss, perplexity)
# - Evaluation results (accuracy)
# - Model artifacts
# - Hyperparameters
```

---

**🎯 This README covers the complete pipeline from raw data to trained models. For detailed guides, see:**
- `REASONING_PATHS_USAGE.md`: Alternative reasoning path generation
- `AXOLOTL_SETUP_GUIDE.md`: Axolotl fine-tuning setup
- `TRAINING_EVALUATION_GUIDE.md`: Training and evaluation workflows
