# Vision-Language Model Training with SLURM

Fast training orchestrator for Qwen3-VL and InternVL3.5 models with W&B logging.

## Quick Start

```bash
# List models
python train_mllm.py --list_models

# Generate config + SLURM script
python train_mllm.py --model qwen3_vl_2b --dataset OVIS

# Submit to SLURM
python train_mllm.py --model qwen3_vl_2b --dataset OVIS --submit
```

## Usage Examples

```bash
# Preview config without saving
python train_mllm.py --model qwen3_vl_2b --dataset OVIS --preview

# Single GPU training (1-4B models)
python train_mllm.py --model qwen3_vl_2b --dataset OVIS --finetuning_type full

# Multi-GPU training (8B models)
python train_mllm.py --model qwen3_vl_8b --dataset OVIS --num_gpus 2 --submit

# LoRA fine-tuning (memory efficient)
python train_mllm.py --model qwen3_vl_8b --dataset OVIS --finetuning_type lora

# QLoRA (minimal memory, 4-bit)
python train_mllm.py --model qwen3_vl_8b --dataset OVIS --finetuning_type qlora

# Specific dataset types
python train_mllm.py --model qwen3_vl_2b --dataset OVIS --dataset_type single_image

# Multiple dataset types
python train_mllm.py --model qwen3_vl_2b --dataset OVIS \
  --dataset_type single_image multi_image_single_turn
```

## Models

| Model | Size | GPUs | Tier |
|-------|------|------|------|
| qwen3_vl_2b | 2B | 1 | tiny |
| qwen3_vl_4b | 4B | 1 | tiny |
| qwen3_vl_8b | 8B | 1-2 | small |
| qwen3_vl_30b_a3b | 30B | 2 | large |
| qwen3_vl_235b_a22b | 235B | 4 | xlarge |
| internvl3_1b | 1B | 1 | tiny |
| internvl3_4b | 4B | 1 | tiny |
| internvl3_8b | 8B | 1-2 | small |
| internvl3_20b | 20B | 2 | medium |
| internvl3_30b_a3b | 30B | 2 | large |

## Datasets

- `OVIS` - Object VQA/Scene Understanding
- `LVVis` - Large-scale Vision dataset
- `Youtube-VIS-2021/2022` - Video Instance Segmentation

Dataset types: `single_image`, `multi_image_single_turn`, `multi_image_multi_turn`

## Finetuning Types

- `full` - Train entire model (best quality, high memory)
- `lora` - LoRA adaptation (medium memory, fast)
- `qlora` - 4-bit quantized LoRA (minimal memory)

## Output

Generated files in:
- `training_configs/` - YAML configs for LLaMA-Factory
- `sbatch_scripts/` - SLURM submission scripts

## Features

✅ Automatic hyperparameter scaling by model size  
✅ Multi-GPU support with DeepSpeed  
✅ W&B logging configured  
✅ Held-out validation datasets  
✅ SLURM job generation and submission  
✅ All three finetuning methods supported  

## Tips

1. Start small: test with 2B model first
2. Use `--preview` to check config before running
3. Use `--submit` to run directly on SLURM
4. Check logs: `tail -f /usr/stud/falu/code/vis/logs/train-*.out`
5. Monitor jobs: `squeue -u $USER`

