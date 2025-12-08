# 🚀 Vision-Language Training System - Ready to Use

Complete, streamlined training orchestrator for Qwen3-VL and InternVL3 models with W&B integration.

## What You Have

### 4 Core Python Modules (~1000 lines total)

1. **train_mllm.py** - Main CLI (240 lines)
2. **train_config.py** - Model registry (344 lines)  
3. **config_generator.py** - YAML builder with W&B (180 lines)
4. **sbatch_generator.py** - SLURM script builder (216 lines)

### 3 Documentation Files

- **TRAINING_GUIDE.md** - Quick reference
- **EXAMPLES.sh** - Example workflows  
- **IMPLEMENTATION_SUMMARY.md** - Technical details

## Quick Start (60 seconds)

```bash
cd /home/stud/falu/code/LLaMA-Factory

# Preview config
python train_mllm.py --model qwen3_vl_2b --dataset OVIS --preview

# Generate + submit
python train_mllm.py --model qwen3_vl_2b --dataset OVIS --submit

# Check job
squeue -u $USER
```

## All Features

| Feature | Status | Details |
|---------|--------|---------|
| **10 Models** | ✅ | Qwen3-VL, InternVL3 all sizes |
| **4 Datasets** | ✅ | OVIS, LVVis, Youtube-VIS variants |
| **3 Finetuning Methods** | ✅ | Full, LoRA, QLoRA |
| **Auto Scaling** | ✅ | Batch size, LR, eval frequency by model size |
| **Multi-GPU** | ✅ | DeepSpeed ZeRO-2/3 auto-selected |
| **W&B Logging** | ✅ | Configured in every generated config |
| **SLURM Support** | ✅ | Auto-generate + submit jobs |
| **Validation Split** | ✅ | Using held-out validation datasets |

## Usage Patterns

```bash
# 1. SINGLE GPU (2-4B models)
python train_mllm.py --model qwen3_vl_2b --dataset OVIS

# 2. MULTI GPU (8B+ models)
python train_mllm.py --model qwen3_vl_8b --dataset OVIS --num_gpus 2

# 3. MEMORY EFFICIENT
python train_mllm.py --model qwen3_vl_8b --dataset OVIS --finetuning_type qlora

# 4. QUICK PROTOTYPE
python train_mllm.py --model qwen3_vl_2b --dataset OVIS --finetuning_type lora

# 5. PRODUCTION TRAINING  
python train_mllm.py --model qwen3_vl_8b --dataset OVIS --finetuning_type full --num_gpus 2 --submit

# 6. SPECIFIC DATASET TYPE
python train_mllm.py --model qwen3_vl_4b --dataset OVIS --dataset_type single_image

# 7. MULTIPLE DATASET TYPES
python train_mllm.py --model qwen3_vl_4b --dataset OVIS \
  --dataset_type single_image multi_image_single_turn
```

## Generated Files

After running the script, you get:

**YAML Config** (`training_configs/qwen3_vl_8b_OVIS_single_image_full.yaml`)
```yaml
model_name_or_path: Qwen/Qwen3-VL-8B-Instruct
dataset: OVIS_train_qwen3_single_image
eval_dataset: OVIS_val_qwen3_single_image
per_device_train_batch_size: 2
learning_rate: 3.0e-05
report_to: wandb
run_name: qwen3_vl_8b_OVIS_single_image
project_name: llamafactory-vision
# ... full config ready to train
```

**SLURM Script** (`sbatch_scripts/qwen3_vl_8b_OVIS_single_image_full.sbatch`)
```bash
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=72-00:00:00

# Multi-GPU setup with torchrun
torchrun --nproc_per_node 2 src/train.py training_configs/...
```

## Hyperparameters by Model Size

Automatically selected based on model:

| Size | Example | Batch | LR | Epochs | Image Px | DeepSpeed |
|------|---------|-------|-----|--------|----------|-----------|
| 2-4B | qwen3_vl_2b | 4 | 5e-5 | 3 | 768×768 | ZeRO-2 |
| 8B | qwen3_vl_8b | 2 | 3e-5 | 3 | 768×768 | ZeRO-2 |
| 20B | internvl3_20b | 1 | 2e-5 | 2 | 576×576 | ZeRO-3 |
| 30B | qwen3_vl_30b_a3b | 1 | 1e-5 | 2 | 512×512 | ZeRO-3 |
| 235B | qwen3_vl_235b_a22b | 1 | 5e-6 | 1 | 512×512 | ZeRO-3 |

## Monitoring

```bash
# Check job status
squeue -u $USER

# View logs (live)
tail -f /usr/stud/falu/code/vis/logs/train-*.out

# View W&B dashboard
# Visit: wandb.ai/projects/llamafactory-vision

# Cancel job if needed
scancel JOB_ID
```

## What's Configured Automatically

✅ Model selection with correct template  
✅ Dataset pairs (train + validation)  
✅ Batch size based on model tier  
✅ Learning rate scaled appropriately  
✅ Number of epochs for convergence  
✅ Image resolution based on memory  
✅ DeepSpeed stage (ZeRO-2 or ZeRO-3)  
✅ Multi-GPU torchrun setup  
✅ W&B project and run naming  
✅ Evaluation frequency by dataset size  
✅ Output directory structure  

## Examples

### Training 2B Model (Quick Prototype)
```bash
python train_mllm.py --model qwen3_vl_2b --dataset OVIS --finetuning_type lora --submit
# ~6 hours, 1 GPU, ~8GB memory
```

### Training 8B Model (Production)
```bash
python train_mllm.py --model qwen3_vl_8b --dataset OVIS --finetuning_type full --num_gpus 2 --submit
# ~24 hours, 2 GPUs, ~36GB memory
```

### Training 30B Model (Best Quality)
```bash
python train_mllm.py --model qwen3_vl_30b_a3b --dataset OVIS --finetuning_type full --num_gpus 2 --submit
# ~48 hours, 2 GPUs, sparse model, ~35GB memory
```

### Memory-Constrained (Large Model on Limited GPU)
```bash
python train_mllm.py --model qwen3_vl_8b --dataset OVIS --finetuning_type qlora --submit
# ~48 hours, 1 GPU, 4-bit quantization, ~8GB memory
```

## File Locations

```
/home/stud/falu/code/LLaMA-Factory/
├── train_mllm.py              # Main script
├── train_config.py            # Config registry
├── config_generator.py        # YAML builder
├── sbatch_generator.py        # SLURM builder
├── TRAINING_GUIDE.md          # User guide
├── EXAMPLES.sh                # Example commands
├── training_configs/          # Generated YAML configs
└── sbatch_scripts/            # Generated SLURM scripts

Data:
/storage/user/falu/vis/processed/dataset_info.json  # Dataset registry
/storage/user/falu/.cache/huggingface/               # Model cache

Output:
/storage/user/falu/trained_models/                   # Trained models
/usr/stud/falu/code/vis/logs/                        # Training logs
```

## Testing

All modules tested and working:
- ✅ Config generation with W&B
- ✅ SLURM script generation
- ✅ Model registry (10 models)
- ✅ Dataset validation mapping
- ✅ Multi-GPU setup
- ✅ End-to-end workflow

## Next Steps

1. **List available options**
   ```bash
   python train_mllm.py --list_models
   python train_mllm.py --list_datasets
   ```

2. **Preview a configuration**
   ```bash
   python train_mllm.py --model qwen3_vl_2b --dataset OVIS --preview
   ```

3. **Submit your first training job**
   ```bash
   python train_mllm.py --model qwen3_vl_2b --dataset OVIS --submit
   ```

4. **Monitor training**
   ```bash
   squeue -u $USER
   tail -f /usr/stud/falu/code/vis/logs/train-*.out
   ```

---

**Status**: ✅ Ready for immediate use  
**Code Quality**: Clean, concise, ~1000 lines  
**Testing**: All components verified  
**Documentation**: Complete and concise  
