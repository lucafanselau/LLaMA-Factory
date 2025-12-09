"""YAML configuration generator for multimodal training with dynamic hyperparameters."""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml
from train_config import (
    get_model_config,
    get_finetuning_config,
    build_dataset_pairs,
    compute_optimal_config,
    generate_run_name,
    DATASET_VALIDATION_MAP,
)


class ConfigGenerator:
    """Generate YAML training configurations with dynamic hyperparameter optimization."""

    def __init__(
        self,
        model_key: str,
        dataset_name: str,
        finetuning_type: str = "full",
        dataset_types: Optional[List[str]] = None,
        num_gpus: int = 1,
        gpu_vram_gb: int = 48,
        dataset_dir: str = "/storage/user/falu/vis/processed",
        output_base: str = "/storage/user/falu/trained_models",
        hf_cache: str = "/storage/user/falu/.cache/huggingface",
        use_tokenized_cache: bool = True,
        use_sampled_validation: bool = True,
        num_epochs: int = 6,
        safety_margin: float = 0.75,
        enable_gradient_checkpointing: bool = True,
        description: Optional[str] = None,
    ):
        self.model_key = model_key
        self.dataset_name = dataset_name
        self.finetuning_type = finetuning_type
        self.dataset_types = dataset_types or [
            "single_image",
            "multi_image_single_turn",
            "multi_image_multi_turn",
        ]
        self.num_gpus = num_gpus
        self.gpu_vram_gb = gpu_vram_gb
        self.dataset_dir = dataset_dir
        self.output_base = output_base
        self.hf_cache = hf_cache
        self.use_tokenized_cache = use_tokenized_cache
        self.use_sampled_validation = use_sampled_validation
        self.num_epochs = num_epochs
        self.safety_margin = safety_margin
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.description = description

        # Get model and finetuning configurations
        self.model_config = get_model_config(model_key)
        self.finetune_config = get_finetuning_config(finetuning_type)

        # Validate GPU count
        min_gpus = self.model_config["min_gpus"]
        if num_gpus < min_gpus:
            raise ValueError(f"Model {model_key} requires at least {min_gpus} GPUs")

        # Initial optimal config (will be recomputed with dataset size)
        self.optimal = None
        self._dataset_info = None

    def _load_dataset_info(self) -> Dict[str, Any]:
        """Load dataset_info.json."""
        if self._dataset_info is not None:
            return self._dataset_info

        dataset_info_path = Path(self.dataset_dir) / "dataset_info.json"
        if not dataset_info_path.exists():
            self._dataset_info = {}
            return self._dataset_info

        with open(dataset_info_path) as f:
            self._dataset_info = json.load(f)
        return self._dataset_info

    def _estimate_dataset_size(self, dataset_name: str) -> int:
        """Estimate number of samples by counting JSONL lines."""
        dataset_info = self._load_dataset_info()

        if dataset_name not in dataset_info:
            return 10000  # Conservative default

        file_name = dataset_info[dataset_name].get("file_name", "")
        file_path = Path(self.dataset_dir) / file_name

        if file_path.exists():
            try:
                with open(file_path) as f:
                    return sum(1 for _ in f)
            except Exception:
                return 10000

        return 10000

    def _get_total_train_samples(self, train_datasets: List[str]) -> int:
        """Sum samples across all training datasets."""
        total = sum(self._estimate_dataset_size(ds) for ds in train_datasets)
        return max(total, 1000)  # Minimum to avoid division issues

    def _get_tokenized_path(self, dataset_str: str) -> str:
        """Get tokenized cache path based on model family and dataset."""
        family = self.model_config["family"]

        # Use sampled validation cache for faster eval
        if self.use_sampled_validation and dataset_str == "all":
            cache_name = "all_val_sampled"
        else:
            cache_name = dataset_str

        return str(Path(self.dataset_dir) / "tokenized" / family / cache_name)

    def _build_datasets(self) -> Tuple[List[str], List[str], str]:
        """Build train and validation dataset lists."""
        if self.dataset_name == "all":
            model_family = self.model_config["family"]
            train_datasets, val_datasets = [], []
            for train_key, val_key in DATASET_VALIDATION_MAP.items():
                if f"_{model_family}_" in train_key:
                    train_datasets.append(train_key)
                    val_datasets.append(val_key)
            dataset_str = "all"
        else:
            train_datasets, val_datasets = build_dataset_pairs(
                self.dataset_name,
                self.model_config["family"],
                self.dataset_types,
            )
            dataset_str = f"{self.dataset_name}_{'_'.join(self.dataset_types)}"

        return train_datasets, val_datasets, dataset_str

    def generate(self) -> Dict[str, Any]:
        """Generate complete training configuration."""
        train_datasets, val_datasets, dataset_str = self._build_datasets()

        # Estimate total training samples for smart eval scheduling
        total_train_samples = self._get_total_train_samples(train_datasets)

        # Compute optimal hyperparameters with dataset size awareness
        self.optimal = compute_optimal_config(
            model_key=self.model_key,
            finetuning_type=self.finetuning_type,
            num_gpus=self.num_gpus,
            gpu_vram_gb=self.gpu_vram_gb,
            total_train_samples=total_train_samples,
            num_epochs=self.num_epochs,
            safety_margin=self.safety_margin,
            enable_gradient_checkpointing=self.enable_gradient_checkpointing,
        )

        # Generate concise run name for output directory
        run_name = generate_run_name(
            model_key=self.model_key,
            finetuning_type=self.finetuning_type,
            dataset_name=self.dataset_name,
            dataset_types=self.dataset_types,
            num_gpus=self.num_gpus,
            gpu_vram_gb=self.gpu_vram_gb,
            effective_batch_size=self.optimal["effective_batch_size"],
            description=self.description,
        )

        output_dir = Path(self.output_base) / run_name

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
            "image_max_pixels": 196608,
            "image_min_pixels": 512,
            "video_max_pixels": 16384,
            "video_min_pixels": 256,
            "crop_to_patches": False,
            # ====== DATASET ======
            "dataset": ",".join(train_datasets),
            "eval_dataset": ",".join(val_datasets),
            "eval_on_each_dataset": True,
            "dataset_dir": self.dataset_dir,
            "media_dir": str(Path(self.dataset_dir).parent),
            "cutoff_len": 8192,
            "max_samples": None,
            "overwrite_cache": False,
            "use_fast_tokenizer": True,
            "preprocessing_num_workers": self.optimal["preprocessing_num_workers"],
            "dataloader_num_workers": self.optimal["dataloader_num_workers"],
            "tokenized_path": (
                self._get_tokenized_path(dataset_str)
                if self.use_tokenized_cache
                else None
            ),
            # ====== TRAINING HYPERPARAMETERS (dynamically computed) ======
            "output_dir": str(output_dir),
            "per_device_train_batch_size": self.optimal["batch_size"],
            "per_device_eval_batch_size": self.optimal["batch_size"],
            "gradient_accumulation_steps": self.optimal["grad_accum"],
            "learning_rate": self.optimal["learning_rate"],
            "num_train_epochs": self.optimal["num_epochs"],
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": True,
            "ddp_timeout": 180000000,
            "max_grad_norm": 1.0,
            # ====== EVALUATION ======
            "eval_strategy": "steps",
            "eval_steps": self.optimal["eval_steps"],
            "save_strategy": "steps",
            "save_steps": self.optimal["save_steps"],
            "logging_steps": 10,
            "log_level": "info",
            # ====== GROUNDING METRICS ======
            "predict_with_generate": True,
            "compute_grounding_iou": True,
            "grounding_model_family": self.model_config["family"],
            # ====== OUTPUT ======
            "save_only_model": False,
            "overwrite_output_dir": True,
            "plot_loss": True,
            "report_to": "wandb",
            "run_name": run_name,
            # ====== OPTIMIZATION ======
            "optim": "adamw_torch",
            "gradient_checkpointing": self.optimal["gradient_checkpointing"],
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
        if self.finetune_config.get("quantization_bit") is not None:
            config.update(
                {
                    "quantization_bit": self.finetune_config["quantization_bit"],
                    "quantization_method": self.finetune_config.get(
                        "quantization_method", "bitsandbytes"
                    ),
                }
            )

        # Add DeepSpeed if computed necessary
        if self.optimal["deepspeed_config"]:
            config["deepspeed"] = self.optimal["deepspeed_config"]

        return config

    def to_yaml(self, config: Optional[Dict[str, Any]] = None) -> str:
        """Convert config to YAML string."""
        if config is None:
            config = self.generate()

        def represent_none(dumper, _):
            return dumper.represent_scalar("tag:yaml.org,2002:null", "null")

        yaml.add_representer(type(None), represent_none)
        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def save(self, path: str) -> None:
        """Save config and update dataset_info.json with fast eval entries."""
        config = self.generate()

        # Save YAML config
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_yaml(config))
        print(f"✓ Config saved to: {path}")

    def print_config(self) -> None:
        """Pretty print configuration with computed settings."""
        config = self.generate()
        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Model: {self.model_key} ({self.model_config['params_b']}B params)")
        print(f"GPUs: {self.num_gpus}x {self.gpu_vram_gb}GB")
        print(f"Effective batch size: {self.optimal['effective_batch_size']}")
        print(
            f"  = {self.optimal['batch_size']} batch × {self.num_gpus} GPUs × {self.optimal['grad_accum']} accum"
        )

        if self.optimal.get("steps_per_epoch"):
            print(f"Steps/epoch: {self.optimal['steps_per_epoch']}")
            print(f"Total steps: {self.optimal['total_steps']}")
            print(f"Eval every: {self.optimal['eval_steps']} steps")
            print(f"Save every: {self.optimal['save_steps']} steps")
            evals_per_epoch = (
                self.optimal["steps_per_epoch"] // self.optimal["eval_steps"]
            )
            print(f"  (~{evals_per_epoch} evals per epoch)")

        # Show validation cache info
        val_cache = (
            "all_val_sampled (~1.5k samples)"
            if self.use_sampled_validation
            else "all (full ~18k samples)"
        )
        print(f"Validation cache: {val_cache}")

        ds = (
            "ZeRO-" + self.optimal["deepspeed_config"].split("_z")[1].split("_")[0]
            if self.optimal["deepspeed_config"]
            else "None (DDP)"
        )
        print(f"DeepSpeed: {ds}")
        print("-" * 70)
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
        print("=" * 70)

    def generate_full_eval_config(self, checkpoint_path: str) -> Dict[str, Any]:
        """Generate config for full evaluation after training (no sample limits)."""
        train_datasets, val_datasets, dataset_str = self._build_datasets()

        return {
            "stage": "sft",
            "do_eval": True,
            "do_train": False,
            "model_name_or_path": checkpoint_path,
            "template": self.model_config["template"],
            "trust_remote_code": True,
            "finetuning_type": self.finetune_config["finetuning_type"],
            "eval_dataset": ",".join(val_datasets),  # Full datasets, not _fast
            "eval_on_each_dataset": True,
            "dataset_dir": self.dataset_dir,
            "media_dir": str(Path(self.dataset_dir).parent),
            "cutoff_len": 8192,
            "per_device_eval_batch_size": (
                self.optimal["batch_size"] if self.optimal else 8
            ),
            "preprocessing_num_workers": 16,
            "bf16": True,
            "report_to": "none",
            "output_dir": str(Path(checkpoint_path).parent / "full_eval"),
        }

    def save_full_eval_config(
        self, checkpoint_path: str, eval_config_path: str
    ) -> None:
        """Save a separate config for full evaluation."""
        config = self.generate_full_eval_config(checkpoint_path)
        Path(eval_config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(eval_config_path, "w") as f:
            f.write(self.to_yaml(config))
        print(f"✓ Full eval config saved to: {eval_config_path}")
