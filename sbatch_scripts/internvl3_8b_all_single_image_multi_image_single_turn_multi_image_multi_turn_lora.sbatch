#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=node[15-20]
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-12:12:00
#SBATCH --job-name=train-internvl3_8b_all_single_image_multi_image_single_t
#SBATCH --output=/usr/stud/falu/code/LLaMA-Factory/logs/train-%j.out
#SBATCH --error=/usr/stud/falu/code/LLaMA-Factory/logs/train-%j.out

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

set -e  # Exit on error

echo "Starting training job: train-internvl3_8b_all_single_image_multi_image_single_t"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Allocated GPUs: ${SLURM_GPUS}"
echo "Host: $(hostname)"
echo "================================"

# Activate virtual environment
source /home/stud/falu/code/LLaMA-Factory/.venv/bin/activate

# Set Hugging Face cache
export HF_HOME=/storage/user/falu/.cache/huggingface

# ============================================================================
# DISTRIBUTED TRAINING SETUP
# ============================================================================

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

cd /home/stud/falu/code/LLaMA-Factory

echo "Running training with config: training_configs/internvl3_8b_all_single_image_multi_image_single_turn_multi_image_multi_turn_lora.yaml"
echo "================================"

python src/train.py training_configs/internvl3_8b_all_single_image_multi_image_single_turn_multi_image_multi_turn_lora.yaml

TRAIN_EXIT_CODE=$?

# ============================================================================
# JOB COMPLETION
# ============================================================================

if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed with exit code ${TRAIN_EXIT_CODE}"
    exit ${TRAIN_EXIT_CODE}
fi

echo "Job finished at: $(date)"
