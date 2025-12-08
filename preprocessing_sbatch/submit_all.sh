#!/bin/bash
# Submit all preprocessing jobs
sbatch preprocessing_sbatch/preproc_qwen3_all.sbatch
sbatch preprocessing_sbatch/preproc_internvl3_all.sbatch
