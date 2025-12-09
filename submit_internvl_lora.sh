#!/bin/bash
# Submit all InternVL models for LoRA training

MODELS=(
    "internvl3_1b"
    "internvl3_4b"
    "internvl3_8b"
    "internvl3_20b"
    "internvl3_30b_a3b"
)

for model in "${MODELS[@]}"; do
    echo "Submitting $model with LoRA..."
    python train_mllm.py --model "$model" --finetuning_type lora --submit
    echo ""
done

echo "All jobs submitted!"

