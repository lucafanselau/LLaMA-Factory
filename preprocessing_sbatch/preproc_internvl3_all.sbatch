#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --nodelist=node[15-20]
#SBATCH --time=0-12:00:00
#SBATCH --job-name=preproc-internvl3-all
#SBATCH --output=/usr/stud/falu/code/LLaMA-Factory/logs/preproc-%j.out
#SBATCH --error=/usr/stud/falu/code/LLaMA-Factory/logs/preproc-%j.out

set -e

echo "Preprocessing job: preproc-internvl3-all"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "================================"

source /home/stud/falu/code/LLaMA-Factory/.venv/bin/activate
export HF_HOME=/storage/user/falu/.cache/huggingface

cd /home/stud/falu/code/LLaMA-Factory

echo "Running preprocessing with config: preprocessing_configs/preproc_internvl3_all.yaml"
python src/train.py preprocessing_configs/preproc_internvl3_all.yaml

echo "Preprocessing completed at: $(date)"
