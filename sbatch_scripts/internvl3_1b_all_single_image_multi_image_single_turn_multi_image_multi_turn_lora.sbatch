#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=node[15-20]
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-12:12:00
#SBATCH --job-name=train-internvl3_1b_all_single_image_multi_image_single_t
#SBATCH --output=/usr/stud/falu/code/LLaMA-Factory/logs/train-%j.out
#SBATCH --error=/usr/stud/falu/code/LLaMA-Factory/logs/train-%j.out

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

set -e  # Exit on error

echo "Starting training job: train-internvl3_1b_all_single_image_multi_image_single_t"
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

# Multi-GPU setup with torchrun
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=${SLURM_NODELIST%%,*}
export MASTER_PORT=29500

echo "Multi-GPU Training Configuration:"
echo "  GPUs per node: ${NPROC_PER_NODE}"
echo "  Total nodes: ${NNODES}"
echo "  Master address: ${MASTER_ADDR}"
echo "  Master port: ${MASTER_PORT}"

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

cd /home/stud/falu/code/LLaMA-Factory

echo "Running training with config: training_configs/internvl3_1b_all_single_image_multi_image_single_turn_multi_image_multi_turn_lora.yaml"
echo "================================"

torchrun \
  --nproc_per_node ${NPROC_PER_NODE} \
  --nnodes ${NNODES} \
  --node_rank ${NODE_RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  src/train.py training_configs/internvl3_1b_all_single_image_multi_image_single_turn_multi_image_multi_turn_lora.yaml

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
