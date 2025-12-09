"""Model registry and dynamic hyperparameter computation for multimodal VLM training."""

from typing import Dict, List, Optional, Tuple, Any

# ============================================================================
# MODEL REGISTRY
# ============================================================================
# Each model has memory estimates for dynamic config calculation.
# - params_b: Total parameters in billions
# - active_params_b: Active parameters (differs for MoE models)
# - min_gpus: Minimum GPUs required (1 for most, more for very large)

MODEL_REGISTRY = {
    # Qwen3-VL Models
    "qwen3_vl_2b": {
        "model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
        "template": "qwen3_vl_nothink",
        "family": "qwen3",
        "params_b": 2.0,
        "active_params_b": 2.0,
        "min_gpus": 1,
    },
    "qwen3_vl_4b": {
        "model_name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
        "template": "qwen3_vl_nothink",
        "family": "qwen3",
        "params_b": 4.0,
        "active_params_b": 4.0,
        "min_gpus": 1,
    },
    "qwen3_vl_8b": {
        "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
        "template": "qwen3_vl_nothink",
        "family": "qwen3",
        "params_b": 8.0,
        "active_params_b": 8.0,
        "min_gpus": 1,
    },
    "qwen3_vl_30b_a3b": {
        "model_name_or_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "template": "qwen3_vl_nothink",
        "family": "qwen3",
        "params_b": 30.0,
        "active_params_b": 3.0,  # MoE: only ~3B active
        "min_gpus": 1,
    },
    "qwen3_vl_235b_a22b": {
        "model_name_or_path": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "template": "qwen3_vl_nothink",
        "family": "qwen3",
        "params_b": 235.0,
        "active_params_b": 22.0,  # MoE: ~22B active
        "min_gpus": 4,
    },
    # InternVL3.5 Models
    "internvl3_1b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-1B-HF",
        "template": "intern_vl",
        "family": "internvl3",
        "params_b": 1.0,
        "active_params_b": 1.0,
        "min_gpus": 1,
    },
    "internvl3_4b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-4B-HF",
        "template": "intern_vl",
        "family": "internvl3",
        "params_b": 4.0,
        "active_params_b": 4.0,
        "min_gpus": 1,
    },
    "internvl3_8b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-8B-HF",
        "template": "intern_vl",
        "family": "internvl3",
        "params_b": 8.0,
        "active_params_b": 8.0,
        "min_gpus": 1,
    },
    "internvl3_20b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-20B-HF",
        "template": "intern_vl",
        "family": "internvl3",
        "params_b": 20.0,
        "active_params_b": 20.0,
        "min_gpus": 1,
    },
    "internvl3_30b_a3b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-30B-A3B-HF",
        "template": "intern_vl",
        "family": "internvl3",
        "params_b": 30.0,
        "active_params_b": 3.0,  # MoE
        "min_gpus": 1,
    },
}

# ============================================================================
# DATASET REGISTRY & VALIDATION MAPPING
# ============================================================================

DATASET_VALIDATION_MAP = {
    # OVIS datasets
    "OVIS_train_qwen3_single_image": "OVIS_val_qwen3_single_image",
    "OVIS_train_qwen3_multi_image_single_turn": "OVIS_val_qwen3_multi_image_single_turn",
    "OVIS_train_qwen3_multi_image_multi_turn": "OVIS_val_qwen3_multi_image_multi_turn",
    "OVIS_train_internvl3_single_image": "OVIS_val_internvl3_single_image",
    "OVIS_train_internvl3_multi_image_single_turn": "OVIS_val_internvl3_multi_image_single_turn",
    "OVIS_train_internvl3_multi_image_multi_turn": "OVIS_val_internvl3_multi_image_multi_turn",
    # LVVis datasets
    "LVVis_train_qwen3_single_image": "LVVis_val_qwen3_single_image",
    "LVVis_train_qwen3_multi_image_single_turn": "LVVis_val_qwen3_multi_image_single_turn",
    "LVVis_train_qwen3_multi_image_multi_turn": "LVVis_val_qwen3_multi_image_multi_turn",
    "LVVis_train_internvl3_single_image": "LVVis_val_internvl3_single_image",
    "LVVis_train_internvl3_multi_image_single_turn": "LVVis_val_internvl3_multi_image_single_turn",
    "LVVis_train_internvl3_multi_image_multi_turn": "LVVis_val_internvl3_multi_image_multi_turn",
    # Youtube-VIS-2021 datasets
    "Youtube-VIS-2021_train_qwen3_single_image": "Youtube-VIS-2021_val_qwen3_single_image",
    "Youtube-VIS-2021_train_qwen3_multi_image_single_turn": "Youtube-VIS-2021_val_qwen3_multi_image_single_turn",
    "Youtube-VIS-2021_train_qwen3_multi_image_multi_turn": "Youtube-VIS-2021_val_qwen3_multi_image_multi_turn",
    "Youtube-VIS-2021_train_internvl3_single_image": "Youtube-VIS-2021_val_internvl3_single_image",
    "Youtube-VIS-2021_train_internvl3_multi_image_single_turn": "Youtube-VIS-2021_val_internvl3_multi_image_single_turn",
    "Youtube-VIS-2021_train_internvl3_multi_image_multi_turn": "Youtube-VIS-2021_val_internvl3_multi_image_multi_turn",
    # Youtube-VIS-2022 datasets
    "Youtube-VIS-2022_train_qwen3_single_image": "Youtube-VIS-2022_val_qwen3_single_image",
    "Youtube-VIS-2022_train_qwen3_multi_image_single_turn": "Youtube-VIS-2022_val_qwen3_multi_image_single_turn",
    "Youtube-VIS-2022_train_qwen3_multi_image_multi_turn": "Youtube-VIS-2022_val_qwen3_multi_image_multi_turn",
    "Youtube-VIS-2022_train_internvl3_single_image": "Youtube-VIS-2022_val_internvl3_single_image",
    "Youtube-VIS-2022_train_internvl3_multi_image_single_turn": "Youtube-VIS-2022_val_internvl3_multi_image_single_turn",
    "Youtube-VIS-2022_train_internvl3_multi_image_multi_turn": "Youtube-VIS-2022_val_internvl3_multi_image_multi_turn",
}

# ============================================================================
# FINETUNING TYPE CONFIGURATIONS
# ============================================================================

FINETUNING_CONFIGS = {
    "full": {
        "finetuning_type": "full",
        "freeze_vision_tower": False,
        "freeze_multi_modal_projector": False,
        "freeze_language_model": False,
        "quantization_bit": None,
    },
    "lora": {
        "finetuning_type": "lora",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "lora_target": "q_proj,v_proj",
        "quantization_bit": None,
    },
    "qlora": {
        "finetuning_type": "lora",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "quantization_bit": 4,
        "quantization_method": "bitsandbytes",
    },
}

# DeepSpeed config paths
DEEPSPEED_CONFIGS = {
    "z2": "examples/deepspeed/ds_z2_config.json",
    "z3": "examples/deepspeed/ds_z3_config.json",
}

# ============================================================================
# DYNAMIC HYPERPARAMETER COMPUTATION
# ============================================================================


def compute_optimal_config(
    model_key: str,
    finetuning_type: str,
    num_gpus: int,
    gpu_vram_gb: int = 48,
    target_effective_batch: int = 32,
    total_train_samples: Optional[int] = None,
    num_epochs: int = 6,
) -> Dict[str, Any]:
    """
    Dynamically compute optimal training configuration for maximum throughput.

    Principles for maximizing it/s:
    1. Maximize batch_size to fully utilize GPU memory
    2. Minimize gradient accumulation (adds iteration overhead)
    3. Use DeepSpeed only when necessary (adds communication overhead)
    4. For multi-GPU without memory pressure, prefer DDP over DeepSpeed

    Memory estimation (bf16 training):
    - Full: weights (2B/param) + optimizer (8B/param for AdamW fp32) + grads (2B/param) = ~6GB/B
    - LoRA: weights (2B/param) + negligible LoRA overhead = ~2.2GB/B
    - QLoRA: 4-bit weights (~0.5B/param) + LoRA = ~0.8GB/B

    Args:
        total_train_samples: If provided, computes absolute eval/save steps
                            instead of fractional values.
        num_epochs: Number of training epochs.
    """
    model = MODEL_REGISTRY[model_key]
    params_b = model["params_b"]
    active_params_b = model["active_params_b"]

    # Memory per billion params based on finetuning type
    # VLMs need extra headroom due to vision encoder activations
    if finetuning_type == "full":
        # Full: weights + AdamW optimizer states + gradients + activation overhead
        # For MoE: all params for storage, active params for optimizer/gradients
        gb_per_b_storage = 2.0  # bf16 weights
        gb_per_b_training = 6.0  # optimizer + gradients + activation buffers
        model_mem_gb = params_b * gb_per_b_storage + active_params_b * gb_per_b_training
    elif finetuning_type == "lora":
        # LoRA: frozen bf16 weights + tiny LoRA overhead
        model_mem_gb = params_b * 2.5
    else:  # qlora
        # QLoRA: 4-bit weights + LoRA
        model_mem_gb = params_b * 1.0

    # Per-sample memory for VLMs is HIGH due to:
    # - Image patches expand to many tokens (up to 12 patches * 256 tokens = 3072 tokens)
    # - Attention activation scales O(seq_len^2)
    # - Multi-image samples multiply this further
    mem_per_sample_gb = 4.0  # Conservative for VLM training

    usable_vram = gpu_vram_gb * 0.88  # More conservative (88% utilization)
    total_vram = usable_vram * num_gpus

    # Determine DeepSpeed necessity
    if finetuning_type == "qlora":
        deepspeed_config = None
        effective_model_mem = model_mem_gb
    elif model_mem_gb > usable_vram:
        if model_mem_gb > total_vram * 0.85:
            deepspeed_config = DEEPSPEED_CONFIGS["z3"]
            effective_model_mem = model_mem_gb / num_gpus
        else:
            deepspeed_config = DEEPSPEED_CONFIGS["z2"]
            # ZeRO-2: model stays, optimizer/grads sharded
            weights_mem = params_b * 2
            trainable_mem = (model_mem_gb - weights_mem) / num_gpus
            effective_model_mem = weights_mem + trainable_mem
    elif num_gpus > 1 and params_b > 15 and finetuning_type == "full":
        # Large model full finetuning: ZeRO-2 helps with optimizer memory
        deepspeed_config = DEEPSPEED_CONFIGS["z2"]
        weights_mem = params_b * 2
        trainable_mem = (model_mem_gb - weights_mem) / num_gpus
        effective_model_mem = weights_mem + trainable_mem
    else:
        deepspeed_config = None
        effective_model_mem = model_mem_gb

    # Calculate batch size
    available_for_batch = usable_vram - effective_model_mem
    max_batch_size = max(1, int(available_for_batch / mem_per_sample_gb))
    batch_size = min(max_batch_size, 8)

    # Gradient accumulation
    current_effective = batch_size * num_gpus
    grad_accum = (
        max(1, target_effective_batch // current_effective)
        if current_effective < target_effective_batch
        else 1
    )

    # Learning rate with sqrt scaling
    effective_batch = batch_size * num_gpus * grad_accum
    lr = 2e-5 * (effective_batch / 32) ** 0.5
    lr = max(5e-6, min(lr, 1e-4))

    dataloader_workers = min(8, max(4, batch_size * 2))
    preproc_workers = min(64, max(16, num_gpus * 16))

    # Compute eval/save steps - absolute if dataset size known, else fractional
    if total_train_samples is not None:
        steps_per_epoch = max(1, total_train_samples // effective_batch)
        total_steps = steps_per_epoch * num_epochs

        # Target ~3-4 evals per epoch for good feedback, minimum 100 steps
        evals_per_epoch = 8
        eval_steps = max(100, steps_per_epoch // evals_per_epoch)

        # Save checkpoints less frequently (~2 per epoch)
        save_steps = max(eval_steps * 2, steps_per_epoch // 2)
    else:
        # Fallback to fractional (legacy behavior)
        if params_b < 5:
            eval_steps, save_steps = 0.2, 0.5
        elif params_b < 15:
            eval_steps, save_steps = 0.3, 0.6
        else:
            eval_steps, save_steps = 0.5, 1.0
        steps_per_epoch = None
        total_steps = None

    return {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": lr,
        "deepspeed_config": deepspeed_config,
        "dataloader_num_workers": dataloader_workers,
        "preprocessing_num_workers": preproc_workers,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "num_epochs": num_epochs,
        "effective_batch_size": effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_vram_per_gpu": int(
            effective_model_mem + batch_size * mem_per_sample_gb
        ),
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_model_config(model_key: str) -> Dict:
    """Get configuration for a specific model."""
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Model '{model_key}' not found. Available: {available}")
    return MODEL_REGISTRY[model_key]


def get_finetuning_config(finetuning_type: str) -> Dict:
    """Get configuration for finetuning type."""
    if finetuning_type not in FINETUNING_CONFIGS:
        available = ", ".join(sorted(FINETUNING_CONFIGS.keys()))
        raise ValueError(
            f"Finetuning type '{finetuning_type}' not found. Available: {available}"
        )
    return FINETUNING_CONFIGS[finetuning_type].copy()


def build_dataset_pairs(
    dataset_name: str,
    model_family: str,
    dataset_types: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Build train and validation dataset pairs."""
    if dataset_types is None:
        dataset_types = [
            "single_image",
            "multi_image_single_turn",
            "multi_image_multi_turn",
        ]

    train_datasets, val_datasets = [], []
    for dtype in dataset_types:
        train_key = f"{dataset_name}_train_{model_family}_{dtype}"
        val_key = f"{dataset_name}_val_{model_family}_{dtype}"
        if train_key not in DATASET_VALIDATION_MAP:
            raise ValueError(f"Dataset '{train_key}' not found in registry")
        train_datasets.append(train_key)
        val_datasets.append(val_key)

    return train_datasets, val_datasets


def list_available_models() -> str:
    """Return formatted list of available models."""
    lines = ["Available models:"]
    for model_key, cfg in sorted(MODEL_REGISTRY.items()):
        params = cfg["params_b"]
        active = cfg["active_params_b"]
        moe_str = f" (MoE, {active}B active)" if active != params else ""
        lines.append(
            f"  {model_key:25} | {params:5.1f}B{moe_str:20} | {cfg['model_name_or_path']}"
        )
    return "\n".join(lines)


def list_available_datasets() -> str:
    """Return formatted list of available datasets."""
    datasets = set()
    for key in DATASET_VALIDATION_MAP.keys():
        parts = key.split("_")
        if parts[0] == "Youtube":
            dataset_name = "_".join(parts[:3])
        else:
            dataset_name = parts[0]
        datasets.add(dataset_name)
    return "Available datasets:\n  " + "\n  ".join(sorted(datasets))
