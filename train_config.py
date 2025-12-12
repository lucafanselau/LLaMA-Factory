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
        "lora_alpha": 17,
        "lora_dropout": 0.0,
        "lora_target": "all",
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
    safety_margin: float = 0.75,
    enable_gradient_checkpointing: bool = True,
) -> Dict[str, Any]:
    """
    Dynamically compute optimal training configuration with improved memory estimation.

    Principles for maximizing it/s:
    1. Maximize batch_size to fully utilize GPU memory
    2. Minimize gradient accumulation (adds iteration overhead)
    3. Use DeepSpeed only when necessary (adds communication overhead)
    4. For multi-GPU without memory pressure, prefer DDP over DeepSpeed

    Memory estimation (bf16 training):
    - Full: weights (2B/param) + optimizer (8B/param) + grads (2B/param) + activations
    - LoRA: weights (2B/param) + negligible LoRA overhead + activations
    - QLoRA: 4-bit weights (~0.5B/param) + LoRA + activations

    VLM-specific: Activation memory is 30-50% of model size due to long sequences
    from image patches. Gradient checkpointing reduces this by ~60%.

    Args:
        total_train_samples: If provided, computes absolute eval/save steps
        num_epochs: Number of training epochs
        safety_margin: Fraction of available memory to use (0.6-0.8 recommended)
        enable_gradient_checkpointing: Reduces activation memory by ~60%
    """
    model = MODEL_REGISTRY[model_key]
    params_b = model["params_b"]
    active_params_b = model["active_params_b"]

    # Improved memory estimation with separate components
    if finetuning_type == "full":
        # Base model memory (weights in bf16)
        weights_mem = params_b * 2.0

        # Optimizer states (8-bit AdamW: 2 states * 1 byte each for active params)
        # Using 8-bit optimizer saves ~6GB per billion params vs fp32 Adam
        optimizer_mem = active_params_b * 2.0

        # Gradients (bf16 for active params)
        gradient_mem = active_params_b * 2.0

        # Vision model activations (MUCH higher for VLMs)
        # Image patches create 100s-1000s of tokens, attention is O(seq_len^2)
        activation_overhead = params_b * 0.4

        if enable_gradient_checkpointing:
            # Gradient checkpointing reduces activation memory by ~60%
            activation_overhead *= 0.4

        model_mem_gb = weights_mem + optimizer_mem + gradient_mem + activation_overhead

    elif finetuning_type == "lora":
        # Frozen weights (bf16) + small LoRA adapters + reduced activations
        # Still need forward pass activations for long VLM sequences
        weights_mem = params_b * 2.0
        lora_mem = 0.5  # LoRA adapters are tiny
        activation_overhead = (
            params_b * 0.2 if not enable_gradient_checkpointing else params_b * 0.12
        )
        model_mem_gb = weights_mem + lora_mem + activation_overhead

    else:  # qlora
        # 4-bit quantized weights + LoRA + dequantization buffers
        # Note: 4-bit needs dequant buffers for computation, not as light as it seems
        weights_mem = params_b * 0.5
        lora_mem = 0.5
        dequant_buffers = params_b * 0.3  # Temporary fp16 buffers for computation
        activation_overhead = (
            params_b * 0.15 if not enable_gradient_checkpointing else params_b * 0.1
        )
        model_mem_gb = weights_mem + lora_mem + dequant_buffers + activation_overhead

    # Per-sample memory: VLM batches are HEAVY
    # Multi-image samples can have 10K+ tokens with attention O(n^2)
    # LoRA/QLoRA save some memory but VLM sequences are still massive
    # With 8-bit optimizer + reduced resolution, we can be slightly less conservative
    if finetuning_type == "full":
        if enable_gradient_checkpointing:
            mem_per_sample_gb = 2.0  # With 8-bit Adam + lower res + checkpointing
        else:
            mem_per_sample_gb = 4.5  # Conservative for VLM without checkpointing
    elif finetuning_type == "lora":
        # LoRA: frozen backbone saves gradient memory but forward activations still large
        if enable_gradient_checkpointing:
            mem_per_sample_gb = 1.8  # Still need activations for long sequences
        else:
            mem_per_sample_gb = 3.2
    else:  # qlora
        # QLoRA: 4-bit has dequantization overhead + VLM attention is memory-intensive
        if enable_gradient_checkpointing:
            mem_per_sample_gb = 2.0  # 4-bit dequant adds overhead during forward pass
        else:
            mem_per_sample_gb = 3.5

    # More conservative VRAM usage - leave room for CUDA overhead
    usable_vram = gpu_vram_gb * 0.85  # 85% max utilization
    total_vram = usable_vram * num_gpus

    # Determine DeepSpeed necessity
    if finetuning_type == "qlora":
        deepspeed_config = None
        effective_model_mem = model_mem_gb
    elif model_mem_gb > usable_vram:
        if model_mem_gb > total_vram * 0.8:  # Need ZeRO-3
            deepspeed_config = DEEPSPEED_CONFIGS["z2"]
            effective_model_mem = model_mem_gb / num_gpus
        else:  # ZeRO-2 sufficient
            deepspeed_config = DEEPSPEED_CONFIGS["z2"]
            weights_mem = params_b * 2
            trainable_mem = (model_mem_gb - weights_mem) / num_gpus
            effective_model_mem = weights_mem + trainable_mem
    elif num_gpus > 1 and finetuning_type == "full":
        # Always use ZeRO-2 for multi-GPU full finetuning to distribute optimizer states
        deepspeed_config = DEEPSPEED_CONFIGS["z2"]
        weights_mem = params_b * 2
        trainable_mem = (model_mem_gb - weights_mem) / num_gpus
        effective_model_mem = weights_mem + trainable_mem
    else:
        deepspeed_config = None
        effective_model_mem = model_mem_gb

    # Smarter batch size calculation with safety margin
    available_for_batch = usable_vram - effective_model_mem
    safe_batch_mem = available_for_batch * safety_margin
    max_batch_size = max(1, int(safe_batch_mem / mem_per_sample_gb))

    # Cap based on model size and finetuning type to avoid OOM spikes
    # VLM sequences are long (image patches), so caps are conservative even for LoRA/QLoRA
    if finetuning_type == "full":
        # Full finetuning is memory-intensive, be conservative
        if params_b >= 20:
            batch_size = min(max_batch_size, 2)  # Very conservative for large models
        elif params_b >= 8:
            batch_size = min(max_batch_size, 4)
        else:
            batch_size = min(max_batch_size, 6)
    elif finetuning_type == "lora":
        # LoRA is lighter but VLM attention is still heavy, especially at high-res
        if params_b >= 20:
            batch_size = min(max_batch_size, 4)
        elif params_b >= 8:
            batch_size = min(max_batch_size, 6)
        else:
            batch_size = min(max_batch_size, 8)
    else:  # qlora
        # QLoRA: 4-bit dequant overhead means similar memory to LoRA for VLMs
        if params_b >= 20:
            batch_size = min(max_batch_size, 5)
        elif params_b >= 8:
            batch_size = min(max_batch_size, 8)
        else:
            batch_size = min(max_batch_size, 10)

    batch_size = max(1, batch_size)

    # Gradient accumulation
    current_effective = batch_size * num_gpus
    grad_accum = (
        max(1, target_effective_batch // current_effective)
        if current_effective < target_effective_batch
        else 1
    )

    # Learning rate with sqrt scaling
    effective_batch = batch_size * num_gpus * grad_accum
    # Use higher base LR for LoRA (empirically better for adapter training)
    if finetuning_type in ["lora", "qlora"]:
        lr = 6e-5 * (effective_batch / 32) ** 0.5
    else:
        lr = 6e-6 * (effective_batch / 32) ** 0.5
    lr = max(5e-7, min(lr, 1e-4))

    dataloader_workers = min(8, max(4, batch_size * 2))
    preproc_workers = min(64, max(16, num_gpus * 16))

    # Compute eval/save steps - absolute if dataset size known, else fractional
    if total_train_samples is not None:
        steps_per_epoch = max(1, total_train_samples // effective_batch) * num_gpus
        total_steps = steps_per_epoch * num_epochs

        # With fast sampled validation, we can eval much more frequently
        # Target ~15-20 evals per epoch for rapid feedback, minimum 50 steps
        evals_per_epoch = 6
        eval_steps = max(50, steps_per_epoch // evals_per_epoch)

        # save at every eval for lora, qlora, otherwise save at every 4 evals
        save_steps = (
            eval_steps if finetuning_type in ["lora", "qlora"] else eval_steps * 4
        )
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
        "gradient_checkpointing": enable_gradient_checkpointing,
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


def generate_run_name(
    model_key: str,
    finetuning_type: str,
    dataset_name: str,
    dataset_types: List[str],
    num_gpus: int,
    gpu_vram_gb: int,
    effective_batch_size: int,
    description: Optional[str] = None,
) -> str:
    """
    Generate concise, informative run name for WandB and SLURM.

    Format with description: {model}-{desc}-{ft}-{dataset}-{gpus}x{vram}GB-b{batch}
    Format without: {model}-{ft}-{dataset}-{gpus}x{vram}GB-b{batch}

    Examples:
        - internvl3_1b-baseline-full-all-1x48GB-b32
        - qwen3_vl_8b-lora-OVIS-4x48GB-b64
        - internvl3_1b-full-all-si-1x48GB-b32  (single_image only)
    """
    # Dataset suffix if not using all types
    all_types = ["single_image", "multi_image_single_turn", "multi_image_multi_turn"]
    if set(dataset_types) == set(all_types):
        ds_suffix = ""
    elif len(dataset_types) == 1:
        type_map = {
            "single_image": "si",
            "multi_image_single_turn": "mist",
            "multi_image_multi_turn": "mimt",
        }
        ds_suffix = f"-{type_map.get(dataset_types[0], dataset_types[0])}"
    else:
        # Multiple but not all types
        type_map = {
            "single_image": "si",
            "multi_image_single_turn": "mist",
            "multi_image_multi_turn": "mimt",
        }
        abbrevs = [type_map.get(dt, dt[:4]) for dt in dataset_types]
        ds_suffix = f"-{'+'.join(abbrevs)}"

    # Shorten finetuning type
    ft_map = {"full": "full", "lora": "lora", "qlora": "qlora"}
    ft_short = ft_map.get(finetuning_type, finetuning_type)

    # Build name components
    parts = [model_key]

    if description:
        parts.append(description)

    parts.extend(
        [
            ft_short,
            f"{dataset_name}{ds_suffix}",
            f"{num_gpus}x{gpu_vram_gb}GB",
            f"b{effective_batch_size}",
        ]
    )

    return "-".join(parts)
