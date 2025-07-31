# Training and Evaluation Guide

This guide shows how to train multiple models and evaluate them systematically.

## 🚀 Training Multiple Models

### Basic Training Commands

```bash
# Train different configurations with descriptive output directories
python hf_sft_train.py \
    --dataset sft_data/gsm8k/10/original.jsonl \
    --model distilgpt2 \
    --output-dir ./models/distilgpt2_original_10samples

python hf_sft_train.py \
    --dataset sft_data/gsm8k/10/alternative_cots_decontextualized.jsonl \
    --model distilgpt2 \
    --output-dir ./models/distilgpt2_decontextualized_10samples
```

### Advanced Training Options

```bash
# Train with auto batch size detection
python hf_sft_train.py \
    --dataset sft_data/gsm8k/100/original.jsonl \
    --model gpt2 \
    --auto-batch-size \
    --epochs 3 \
    --output-dir ./models/gpt2_original_100samples \
    --experiment-name "model_comparison"

# Train on CPU for local testing (smaller models)
python hf_sft_train.py \
    --dataset sft_data/gsm8k/10/original.jsonl \
    --model HuggingFaceTB/SmolLM-135M \
    --batch-size 1 \
    --epochs 3 \
    --output-dir ./models/smollm_original_10samples \
    --experiment-name "local_cpu_test_smollm"

# Train with mixed precision for faster GPU training
python hf_sft_train.py \
    --dataset sft_data/gsm8k/100/alternative_cots_reasoning_only_direct.jsonl \
    --model allenai/OLMo-2-0425-1B \
    --enable-mixed-precision \
    --auto-batch-size \
    --output-dir ./models/olmo_reasoning_direct_100samples \
    --experiment-name "model_comparison"


```

## 🔍 Evaluating Models

### Quick Evaluation (10 examples)

```bash
# Evaluate each trained model
python evaluate_gsm8k.py --model ./models/distilgpt2_original_10samples --num-examples 10
python evaluate_gsm8k.py --model ./models/distilgpt2_decontextualized_10samples --num-examples 10
```

### Full Evaluation (all test examples)

```bash
# Full GSM8K test set evaluation (1,319 examples)
python evaluate_gsm8k.py \
    --model ./models/gpt2_original_100samples \
    --experiment-name "model_comparison" \
    --run-name "gpt2_original_full_eval"

python evaluate_gsm8k.py \
    --model ./models/olmo_reasoning_direct_100samples \
    --experiment-name "model_comparison" \
    --run-name "olmo_reasoning_full_eval"
```

### Custom Evaluation Settings

```bash
# Evaluate with shorter generation length for faster results
python evaluate_gsm8k.py \
    --model ./models/distilgpt2_original_10samples \
    --num-examples 100 \
    --max-length 256 \
    --output custom_eval_results.json
```

## 📁 Recommended Directory Structure

```
models/
├── distilgpt2_original_10samples/
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── config.json
├── distilgpt2_decontextualized_10samples/
├── gpt2_original_100samples/
├── olmo_reasoning_direct_100samples/
└── [model_name]_[dataset_type]_[n_samples]/

sft_data/
├── gsm8k/
│   ├── 10/
│   │   ├── original.jsonl
│   │   ├── alternative_cots_decontextualized.jsonl
│   │   └── alternative_cots_reasoning_only_direct.jsonl
│   └── 100/
└── [other_datasets]/
```

## 📊 MLflow Organization

### Organize by Experiment Type

```bash
# Training experiments
python hf_sft_train.py \
    --dataset sft_data/gsm8k/10/original.jsonl \
    --model distilgpt2 \
    --experiment-name "small_model_experiments" \
    --output-dir ./models/distilgpt2_original_10samples

# Evaluation experiments (use same experiment name)
python evaluate_gsm8k.py \
    --model ./models/distilgpt2_original_10samples \
    --experiment-name "small_model_experiments" \
    --run-name "eval_distilgpt2_original"
```

### Compare Different Approaches

```bash
# Compare original vs alternative COTs
python hf_sft_train.py \
    --dataset sft_data/gsm8k/100/original.jsonl \
    --experiment-name "cot_comparison" \
    --output-dir ./models/comparison_original_100

python hf_sft_train.py \
    --dataset sft_data/gsm8k/100/alternative_cots_decontextualized.jsonl \
    --experiment-name "cot_comparison" \
    --output-dir ./models/comparison_decontextualized_100

# Evaluate both
python evaluate_gsm8k.py --model ./models/comparison_original_100 --experiment-name "cot_comparison"
python evaluate_gsm8k.py --model ./models/comparison_decontextualized_100 --experiment-name "cot_comparison"
```

## 🛠️ Utility Commands

### List Available Models
```bash
# Find all trained models
find ./models -name "pytorch_model.bin" -o -name "model.safetensors"

# List model directories
ls -la models/
```

### Check MLflow Results
```bash
# Start MLflow UI
mlflow ui

# Open browser to: http://localhost:5000
# Compare experiments, runs, and metrics
```

### Clean Up Old Models
```bash
# Remove specific model
rm -rf ./models/old_model_name

# Remove all models (careful!)
rm -rf ./models/*
```

## 🎯 Quick Start Workflow

1. **Train a model**:
   ```bash
   python hf_sft_train.py --dataset sft_data/gsm8k/10/original.jsonl --model distilgpt2 --output-dir ./models/my_first_model
   ```

2. **Quick evaluation**:
   ```bash
   python evaluate_gsm8k.py --model ./models/my_first_model --num-examples 10
   ```

3. **Check results**:
   ```bash
   mlflow ui
   cat gsm8k_eval_results.json
   ```

4. **Scale up**:
   ```bash
   python hf_sft_train.py --dataset sft_data/gsm8k/100/original.jsonl --model gpt2 --auto-batch-size --output-dir ./models/scaled_model
   python evaluate_gsm8k.py --model ./models/scaled_model --num-examples 100
   ```

## 📈 Performance Tips

- **Use `--auto-batch-size`** to maximize GPU utilization
- **Start with small datasets** (10-100 examples) for quick iteration
- **Use descriptive output directories** for easy model management
- **Group related experiments** in MLflow for easy comparison
- **Evaluate on subsets first** before full test set evaluation 