"""YAML configuration generator for multimodal training."""

from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from train_config import (
    get_model_config,
    get_hyperparams,
    get_finetuning_config,
    build_dataset_pairs,
    DEEPSPEED_CONFIGS,
)


class ConfigGenerator:
    """Generate YAML training configurations."""

    def __init__(
        self,
        model_key: str,
        dataset_name: str,
        finetuning_type: str = "full",
        dataset_types: Optional[List[str]] = None,
        num_gpus: int = 1,
        dataset_dir: str = "/storage/user/falu/vis/processed",
        output_base: str = "/storage/user/falu/trained_models",
        hf_cache: str = "/storage/user/falu/.cache/huggingface",
    ):
        """Initialize config generator."""
        self.model_key = model_key
        self.dataset_name = dataset_name
        self.finetuning_type = finetuning_type
        self.dataset_types = dataset_types or [
            "single_image",
            "multi_image_single_turn",
            "multi_image_multi_turn",
        ]
        self.num_gpus = num_gpus
        self.dataset_dir = dataset_dir
        self.output_base = output_base
        self.hf_cache = hf_cache

        # Get configurations
        self.model_config = get_model_config(model_key)
        self.tier = self.model_config["tier"]
        self.hyperparams = get_hyperparams(self.tier)
        self.finetune_config = get_finetuning_config(finetuning_type)

        # Validate GPU count
        max_gpus = self.model_config["max_gpus"]
        if num_gpus > max_gpus:
            raise ValueError(
                f"Model {model_key} supports max {max_gpus} GPUs, "
                f"but {num_gpus} requested"
            )

        # Adjust DeepSpeed for GPU count
        if num_gpus == 1 and not self.finetune_config.get("requires_deepspeed", False):
            self.deepspeed_config = None
        elif num_gpus == 1:
            self.deepspeed_config = DEEPSPEED_CONFIGS["z2_1gpu"]
        elif num_gpus <= 2:
            self.deepspeed_config = (
                DEEPSPEED_CONFIGS["z2_multi"]
                if self.finetune_config.get("finetuning_type") != "full"
                else DEEPSPEED_CONFIGS["z2_multi"]
            )
        else:
            self.deepspeed_config = DEEPSPEED_CONFIGS["z3_multi"]

    def generate(self) -> Dict[str, Any]:
        """Generate complete training configuration."""
        # Build dataset pairs
        if self.dataset_name == "all":
            # Get all available datasets for this model family
            from train_config import DATASET_VALIDATION_MAP

            model_family = self.model_config["family"]
            all_train_datasets = []
            all_val_datasets = []

            for train_key, val_key in DATASET_VALIDATION_MAP.items():
                if f"_{model_family}_" in train_key:
                    all_train_datasets.append(train_key)
                    all_val_datasets.append(val_key)

            train_datasets = all_train_datasets
            val_datasets = all_val_datasets
            dataset_str = "all"
        else:
            train_datasets, val_datasets = build_dataset_pairs(
                self.dataset_name,
                self.model_config["family"],
                self.dataset_types,
            )
            dataset_str = f"{self.dataset_name}_{'_'.join(self.dataset_types)}"

        # Effective batch size calculation
        per_device_batch_size = self.hyperparams["batch_size"]
        grad_accum = self.hyperparams["grad_accum"]
        effective_batch_size = per_device_batch_size * grad_accum * self.num_gpus

        # Output directory
        output_dir = (
            Path(self.output_base) / self.model_key / self.finetuning_type / dataset_str
        )

        config = {
            # ====== MODEL ======
            "model_name_or_path": self.model_config["model_name_or_path"],
            "template": self.model_config["template"],
            "trust_remote_code": True,
            # ====== TRAINING STAGE & TYPE ======
            "stage": "sft",
            "do_train": True,
            "finetuning_type": self.finetune_config["finetuning_type"],
            # ====== MULTIMODAL SETTINGS ======
            "freeze_vision_tower": self.finetune_config["freeze_vision_tower"],
            "freeze_multi_modal_projector": self.finetune_config[
                "freeze_multi_modal_projector"
            ],
            "freeze_language_model": self.finetune_config["freeze_language_model"],
            "image_max_pixels": self.hyperparams["image_max_pixels"],
            "image_min_pixels": self.hyperparams.get("image_min_pixels", 32 * 32),
            "video_max_pixels": self.hyperparams.get("video_max_pixels", 256 * 256),
            "video_min_pixels": self.hyperparams.get("video_min_pixels", 16 * 16),
            # ====== DATASET ======
            "dataset": ",".join(train_datasets),
            "eval_dataset": ",".join(val_datasets),
            "eval_on_each_dataset": False,
            "dataset_dir": self.dataset_dir,
            "media_dir": str(Path(self.dataset_dir).parent),
            "cutoff_len": 8192,
            "max_samples": self.hyperparams["max_samples"],
            "overwrite_cache": self.hyperparams.get("overwrite_cache", True),
            "preprocessing_num_workers": self.hyperparams.get(
                "preprocessing_num_workers", 16
            ),
            "dataloader_num_workers": self.hyperparams.get("dataloader_num_workers", 4),
            # ====== TRAINING HYPERPARAMETERS ======
            "output_dir": str(output_dir),
            "per_device_train_batch_size": per_device_batch_size,
            "per_device_eval_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": self.hyperparams["learning_rate"],
            "num_train_epochs": self.hyperparams["num_epochs"],
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": True,
            "ddp_timeout": 180000000,
            "max_grad_norm": 1.0,
            # ====== EVALUATION ======
            "eval_strategy": "steps",
            "eval_steps": 0.5,  # self.hyperparams["eval_steps"],
            "save_strategy": "steps",
            "save_steps": self.hyperparams["save_steps"],
            "logging_steps": 10,
            "log_level": "info",
            # ====== OUTPUT ======
            "output_dir": str(output_dir),
            "save_only_model": False,
            "overwrite_output_dir": True,
            "plot_loss": True,
            "report_to": "wandb",
            "run_name": f"{self.model_key}_{self.dataset_name}_{'_'.join(self.dataset_types)}",
            # ====== OPTIMIZATION ======
            "optim": "adamw_torch",
            "resume_from_checkpoint": None,
        }

        # Add LoRA config if needed
        if self.finetune_config["finetuning_type"] == "lora":
            config.update(
                {
                    "lora_rank": self.finetune_config["lora_rank"],
                    "lora_alpha": self.finetune_config["lora_alpha"],
                    "lora_dropout": self.finetune_config["lora_dropout"],
                    "lora_target": self.finetune_config.get("lora_target", "all"),
                    "create_new_adapter": True,
                }
            )

        # Add quantization if needed
        if self.finetune_config["quantization_bit"] is not None:
            config.update(
                {
                    "quantization_bit": self.finetune_config["quantization_bit"],
                    "quantization_method": self.finetune_config.get(
                        "quantization_method", "bitsandbytes"
                    ),
                }
            )

        # Add DeepSpeed if needed
        if self.deepspeed_config and self.finetune_config.get("use_deepspeed", False):
            config["deepspeed"] = self.deepspeed_config

        return config

    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        config = self.generate()

        # Custom YAML representer for None to use "null"
        def represent_none(self, _):
            return self.represent_scalar("tag:yaml.org,2002:null", "null")

        yaml.add_representer(type(None), represent_none)

        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
        return yaml_str

    def save(self, path: str) -> None:
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_yaml())
        print(f"✓ Config saved to: {path}")

    def print_config(self) -> None:
        """Pretty print configuration."""
        config = self.generate()
        print("\n" + "=" * 70)
        print(f"TRAINING CONFIGURATION")
        print("=" * 70)
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
        print("=" * 70)
