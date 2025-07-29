# Axolotl Setup Guide for GSM8K Fine-Tuning

This guide shows how to use [Axolotl](https://axolotl.ai/) for fine-tuning language models on GSM8K reasoning data.

## 🚀 Installation

### Option 1: Docker (Recommended)
```bash
# Clone Axolotl repository
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

# Build Docker image
docker build -t axolotl .

# Run with GPU support
docker run --gpus all -v $(pwd):/workspace axolotl
```

### Option 2: Conda Environment
```bash
# Clone Axolotl
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

# Create conda environment
conda create -n axolotl python=3.10
conda activate axolotl

# Install dependencies
pip install packaging
pip install -e '.[flash-attn,deepspeed]'

# For Apple Silicon Macs
pip install -e '.[deepspeed]'
```

### Option 3: pip install
```bash
pip install axolotl[flash-attn,deepspeed]
```

## 📁 Project Structure

```
sharding_cot_private/
├── axolotl_config.yml           ← Axolotl configuration
├── axolotl_outputs/             ← Training outputs
└── sft_data/                    ← Your existing data (ready to use!)
    └── gsm8k/
        ├── 10/
        ├── 100/
        └── 1000/
```

## 🎯 Your Data is Already Ready!

Your existing JSONL files work perfectly with Axolotl's instruction tuning format:

```json
{"question": "What is 2+2?", "answer": "Let me think step by step...", "final_answer": "4"}
```

The config automatically maps:
- `question` → `field_instruction`
- `answer` → `field_output`
- Formats as: `"Question: {question}\nAnswer: {answer}"`

### 🤖 System Prompt for Proper Formatting

The config includes a system prompt to ensure consistent answer formatting:

```yaml
system_prompt: "You are a helpful assistant that solves math problems step by step. Always end your answer with '#### [number]' where [number] is the final numerical answer."
```

This ensures the model learns to:
- ✅ Show step-by-step reasoning
- ✅ End with `#### number` format
- ✅ Match the evaluation script expectations

## 🏋️ Training Commands

### Quick Test (10 samples)
```bash
# Train directly on your existing data
axolotl train axolotl_config.yml
```

### Full Training (100+ samples)
```bash
# Update config for larger dataset
cp axolotl_config.yml axolotl_config_100.yml
# Edit datasets path in axolotl_config_100.yml:
#   path: sft_data/gsm8k/100/original.jsonl

# Train
axolotl train axolotl_config_100.yml
```

### Train on Alternative COTs
```bash
# Edit config to use alternative reasoning paths
#   path: sft_data/gsm8k/10/alternative_cots_decontextualized.jsonl

axolotl train axolotl_config.yml
```

### Multi-GPU Training
```bash
# For multiple GPUs
accelerate launch -m axolotl.cli.train axolotl_config.yml

# Or with specific GPU count
accelerate launch --num_processes=2 -m axolotl.cli.train axolotl_config.yml
```

## 📊 Model Configurations

### Small Model (Testing)
```yaml
base_model: HuggingFaceTB/SmolLM-135M
micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 3
```

### Medium Model (Better Performance)
```yaml
base_model: microsoft/DialoGPT-medium
micro_batch_size: 1
gradient_accumulation_steps: 8
num_epochs: 3
```

### Large Model (Best Performance)
```yaml
base_model: allenai/OLMo-2-0425-1B
micro_batch_size: 1
gradient_accumulation_steps: 16
num_epochs: 2
```

## 🔍 Model Evaluation

### Generate Predictions
```bash
# Generate completions
axolotl inference axolotl_config.yml \
    --lora-model-dir="./axolotl_outputs" \
    --prompt="Question: What is 2+2?\nAnswer:"
```

### Use Our Evaluation Script
```bash
# Convert Axolotl model for our eval script
# (You'll need to merge and convert the model first)

# Then evaluate
python evaluate_gsm8k.py \
    --model ./axolotl_outputs/merged_model \
    --num-examples 10
```

## ⚙️ Configuration Options

### Key Parameters to Tune

#### Memory Optimization
```yaml
gradient_checkpointing: true
flash_attention: true
load_in_8bit: true          # For large models
load_in_4bit: true          # For very large models
```

#### Training Settings
```yaml
learning_rate: 2e-5         # Conservative for fine-tuning
weight_decay: 0.01          # Regularization
warmup_steps: 10            # LR warmup
lr_scheduler: cosine        # Learning rate schedule
```

#### Data Settings
```yaml
sequence_len: 512           # Match your data length
sample_packing: false       # Don't pack for QA format
train_on_inputs: false      # Only train on completions
```

### LoRA Configuration (Memory Efficient)
```yaml
adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
```

## 🚀 Quick Start Commands

### 1. Install Axolotl
```bash
pip install axolotl[flash-attn,deepspeed]
```

### 2. Train Immediately
```bash
# Your data is already ready! Just train:
axolotl train axolotl_config.yml
```

### 3. Check Results
```bash
# View training logs
tensorboard --logdir ./axolotl_outputs/

# Test inference
axolotl inference axolotl_config.yml \
    --lora-model-dir="./axolotl_outputs" \
    --prompt="Question: John has 5 apples and gives away 2. How many does he have left?\nAnswer:"
```

## 🔧 Troubleshooting

### Common Issues

#### OOM (Out of Memory)
```yaml
# Reduce batch size
micro_batch_size: 1
gradient_accumulation_steps: 8

# Enable memory optimization
gradient_checkpointing: true
load_in_8bit: true
```

#### Slow Training
```yaml
# Enable optimizations
flash_attention: true
bf16: auto
sample_packing: true  # If your data allows
```

#### Data Format Errors
```bash
# Check your data format (should already be correct)
head -n 1 sft_data/gsm8k/10/original.jsonl
# Should show: {"question": "...", "answer": "...", "final_answer": "..."}
```

## 📈 Monitoring Training

### Weights & Biases (Recommended)
```yaml
wandb_project: gsm8k-finetuning
wandb_entity: your-username
wandb_watch: gradients
wandb_log_model: checkpoint
```

### TensorBoard (Local)
```bash
tensorboard --logdir ./axolotl_outputs/
```

## 🎯 Best Practices

### For GSM8K Reasoning:
1. **Use instruction format** - better for step-by-step reasoning
2. **Don't train on questions** - set `train_on_inputs: false`
3. **Conservative learning rates** - 2e-5 or lower
4. **Monitor for overfitting** - use validation set
5. **Test on different question types** - ensure generalization

### System Prompt Customization:

#### For GSM8K (Default):
```yaml
system_prompt: "You are a helpful assistant that solves math problems step by step. Always end your answer with '#### [number]' where [number] is the final numerical answer."
```

#### For Alternative CoT Datasets:
```yaml
system_prompt: "You are a helpful assistant that provides clear, step-by-step mathematical reasoning. Always conclude with '#### [final_answer]' containing only the numerical result."
```

#### For More Structured Output:
```yaml
system_prompt: "Solve math problems systematically. Show your work clearly and always end with '#### [number]' where [number] is the exact numerical answer without units or extra text."
```

### For Small Datasets (10-100 samples):
1. **More epochs** - 3-5 epochs
2. **Lower learning rate** - 1e-5
3. **Smaller models** - avoid overfitting
4. **Regular evaluation** - check progress frequently

### For Production:
1. **Use larger datasets** - 1000+ samples
2. **Multiple evaluation sets** - different difficulty levels
3. **Systematic hyperparameter search** - learning rate, batch size
4. **Model averaging** - combine multiple checkpoints

## 📚 Resources

- [Axolotl Documentation](https://axolotl.ai/)
- [GitHub Repository](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Discord Community](https://discord.gg/axolotl) - 500+ active members
- [Example Configurations](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples) 