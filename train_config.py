"""Model registry and hyperparameter configs for multimodal vision-language training."""

from typing import Dict, List, Optional, Tuple

MODEL_REGISTRY = {
    # Qwen3-VL Models
    "qwen3_vl_2b": {
        "model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
        "template": "qwen3_vl_nothink",
        "tier": "tiny",
        "max_gpus": 4,
        "family": "qwen3",
    },
    "qwen3_vl_4b": {
        "model_name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
        "template": "qwen3_vl_nothink",
        "tier": "tiny",
        "max_gpus": 4,
        "family": "qwen3",
    },
    "qwen3_vl_8b": {
        "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
        "template": "qwen3_vl_nothink",
        "tier": "medium",
        "max_gpus": 4,
        "family": "qwen3",
    },
    "qwen3_vl_30b_a3b": {
        "model_name_or_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "template": "qwen3_vl_nothink",
        "tier": "large",
        "max_gpus": 4,
        "family": "qwen3",
    },
    "qwen3_vl_235b_a22b": {
        "model_name_or_path": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "template": "qwen3_vl_nothink",
        "tier": "xlarge",
        "max_gpus": 4,
        "family": "qwen3",
    },
    # InternVL3.5 Models
    "internvl3_1b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-1B-HF",
        "template": "intern_vl",
        "tier": "tiny",
        "max_gpus": 4,
        "family": "internvl3",
    },
    "internvl3_4b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-4B-HF",
        "template": "intern_vl",
        "tier": "tiny",
        "max_gpus": 4,
        "family": "internvl3",
    },
    "internvl3_8b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-8B-HF",
        "template": "intern_vl",
        "tier": "medium",
        "max_gpus": 4,
        "family": "internvl3",
    },
    "internvl3_20b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-20B-HF",
        "template": "intern_vl",
        "tier": "medium",
        "max_gpus": 4,
        "family": "internvl3",
    },
    "internvl3_30b_a3b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-30B-A3B-HF",
        "template": "intern_vl",
        "tier": "large",
        "max_gpus": 4,
        "family": "internvl3",
    },
}

# ============================================================================
# DATASET REGISTRY & VALIDATION MAPPING
# ============================================================================

# Auto-map training datasets to their corresponding validation sets
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
# HYPERPARAMETER TIERS
# ============================================================================

HYPERPARAMETER_TIERS = {
    "tiny": {
        # 1-4B models, 1 GPU, optimized for speed
        "batch_size": 2,
        "grad_accum": 8,
        "image_max_pixels": 196608,  # ~443x443 (reduced for faster tokenization)
        "image_min_pixels": 512,
        "video_max_pixels": 16384,  # Reduced from 65536
        "video_min_pixels": 256,
        "max_samples": None,  # Use all available data
        "learning_rate": 5e-5,
        "num_epochs": 6,
        "eval_steps": 50,
        "save_steps": 100,
        "deepspeed_stage": 2,  # ZeRO-2 sufficient
        "recommended_gpus": 1,
        "preprocessing_num_workers": 48,
        "dataloader_num_workers": 4,
        "pin_memory": True,
        "overwrite_cache": False,
        "crop_to_patches": False,  # Disable dynamic patching for InternVL
    },
    "small": {
        # 8B models, 1-2 GPUs
        "batch_size": 2,
        "grad_accum": 8,
        "image_max_pixels": 196608,  # ~443x443 (reduced for faster tokenization)
        "image_min_pixels": 512,
        "video_max_pixels": 16384,
        "video_min_pixels": 256,
        "max_samples": 50000,
        "learning_rate": 3e-5,
        "num_epochs": 3,
        "eval_steps": 100,
        "save_steps": 200,
        "deepspeed_stage": 2,
        "recommended_gpus": 1,
        "preprocessing_num_workers": 48,
        "dataloader_num_workers": 4,
        "pin_memory": True,
        "overwrite_cache": False,
        "crop_to_patches": False,
    },
    "medium": {
        # 20B models, 2 GPUs
        "batch_size": 2,
        "grad_accum": 16,
        "image_max_pixels": 196608,  # ~443x443 (reduced for faster tokenization)
        "image_min_pixels": 512,
        "video_max_pixels": 16384,
        "video_min_pixels": 256,
        "max_samples": None,
        "learning_rate": 2e-5,
        "num_epochs": 6,
        "eval_steps": 150,
        "save_steps": 300,
        "deepspeed_stage": 3,
        "recommended_gpus": 2,
        "preprocessing_num_workers": 64,
        "dataloader_num_workers": 4,
        "pin_memory": True,
        "overwrite_cache": False,
        "crop_to_patches": False,
    },
    "large": {
        # 30B sparse models, 2 GPUs
        "batch_size": 1,
        "grad_accum": 16,
        "image_max_pixels": 196608,  # ~443x443 (reduced for faster tokenization)
        "image_min_pixels": 512,
        "video_max_pixels": 16384,
        "video_min_pixels": 256,
        "max_samples": 20000,
        "learning_rate": 1e-5,
        "num_epochs": 2,
        "eval_steps": 200,
        "save_steps": 400,
        "deepspeed_stage": 3,
        "recommended_gpus": 2,
        "preprocessing_num_workers": 64,
        "dataloader_num_workers": 4,
        "pin_memory": True,
        "overwrite_cache": False,
        "crop_to_patches": False,
    },
    "xlarge": {
        # 235B sparse models, 4 GPUs
        "batch_size": 1,
        "grad_accum": 32,
        "image_max_pixels": 196608,  # ~443x443 (reduced for faster tokenization)
        "image_min_pixels": 512,
        "video_max_pixels": 16384,
        "video_min_pixels": 256,
        "max_samples": 15000,
        "learning_rate": 5e-6,
        "num_epochs": 1,
        "eval_steps": 300,
        "save_steps": 500,
        "deepspeed_stage": 3,
        "recommended_gpus": 4,
        "preprocessing_num_workers": 64,
        "dataloader_num_workers": 4,
        "pin_memory": True,
        "overwrite_cache": False,
        "crop_to_patches": False,
    },
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
        "use_deepspeed": True,
        "requires_deepspeed": True,
        "lora_config": None,
        "quantization_bit": None,
        "description": "Train all model parameters (most memory, best quality)",
    },
    "lora": {
        "finetuning_type": "lora",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,  # Changed: freeze projector for speed
        "freeze_language_model": False,
        "use_deepspeed": False,
        "requires_deepspeed": False,
        "lora_rank": 8,  # Reduced from 16
        "lora_alpha": 16,  # Reduced from 32
        "lora_dropout": 0.1,
        "lora_target": "q_proj,v_proj",  # Target only Q,V for efficiency
        "quantization_bit": None,
        "description": "LoRA on language model (optimized for speed)",
    },
    "qlora": {
        "finetuning_type": "lora",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "use_deepspeed": False,
        "requires_deepspeed": False,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "quantization_bit": 4,
        "quantization_method": "bitsandbytes",
        "description": "Quantized LoRA (least memory, ~80% reduction)",
    },
}

# ============================================================================
# DEEPSPEED CONFIGURATIONS
# ============================================================================

DEEPSPEED_CONFIGS = {
    "z2_1gpu": "examples/deepspeed/ds_z2_config.json",
    "z2_multi": "examples/deepspeed/ds_z2_config.json",
    "z3_multi": "examples/deepspeed/ds_z3_config.json",
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


def get_hyperparams(tier: str) -> Dict:
    """Get hyperparameters for a tier."""
    if tier not in HYPERPARAMETER_TIERS:
        available = ", ".join(sorted(HYPERPARAMETER_TIERS.keys()))
        raise ValueError(f"Tier '{tier}' not found. Available: {available}")
    return HYPERPARAMETER_TIERS[tier].copy()


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
    """
    Build train and validation dataset pairs based on dataset name, model family, and types.

    Args:
        dataset_name: e.g., "OVIS", "LVVis", "Youtube-VIS-2021", "Youtube-VIS-2022"
        model_family: e.g., "qwen3_vl", "internvl3"
        dataset_types: e.g., ["single_image"], ["single_image", "multi_image_single_turn"], or None for all

    Returns:
        Tuple of (train_datasets, val_datasets) lists
    """
    if dataset_types is None:
        dataset_types = [
            "single_image",
            "multi_image_single_turn",
            "multi_image_multi_turn",
        ]

    train_datasets = []
    val_datasets = []

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
    for model_key, config in sorted(MODEL_REGISTRY.items()):
        model_path = config["model_name_or_path"]
        tier = config["tier"]
        max_gpus = config["max_gpus"]
        lines.append(f"  {model_key:25} | {tier:8} | {max_gpus} GPU(s) | {model_path}")
    return "\n".join(lines)


def list_available_datasets() -> str:
    """Return formatted list of available datasets."""
    datasets = set()
    for key in DATASET_VALIDATION_MAP.keys():
        # Extract dataset name (e.g., "OVIS" from "OVIS_train_qwen3_single_image")
        parts = key.split("_")
        if parts[0] == "Youtube":
            dataset_name = "_".join(parts[:3])  # "Youtube-VIS-2021"
        else:
            dataset_name = parts[0]
        datasets.add(dataset_name)

    return "Available datasets:\n  " + "\n  ".join(sorted(datasets))
