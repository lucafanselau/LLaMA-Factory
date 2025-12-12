#!/usr/bin/env python3
"""Vision-language model training orchestrator with SLURM support."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
from config_generator import ConfigGenerator
from sbatch_generator import SbatchGenerator
from train_config import (
    MODEL_REGISTRY,
    DATASET_VALIDATION_MAP,
    FINETUNING_CONFIGS,
    list_available_models,
    list_available_datasets,
)


class TrainingOrchestrator:
    """Orchestrates training job creation and submission."""

    def __init__(
        self,
        model: str,
        dataset: str,
        finetuning_type: str = "full",
        dataset_types: Optional[List[str]] = None,
        num_gpus: int = 1,
        gpu_vram_gb: int = 48,
        dataset_dir: str = "/storage/user/falu/vis/processed_llamafactory",
        output_base: str = "/storage/user/falu/trained_models",
        hf_cache: str = "/storage/user/falu/.cache/huggingface",
        config_output: str = "autoconfig/configs",
        sbatch_output: str = "autoconfig/sbatch",
        use_tokenized_cache: bool = True,
        num_epochs: int = 6,
        safety_margin: float = 0.75,
        enable_gradient_checkpointing: bool = True,
        description: Optional[str] = None,
        optimizer: Optional[str] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.finetuning_type = finetuning_type
        self.dataset_types = dataset_types
        self.num_gpus = num_gpus
        self.gpu_vram_gb = gpu_vram_gb
        self.dataset_dir = dataset_dir
        self.output_base = output_base
        self.hf_cache = hf_cache
        self.config_output = Path(config_output)
        self.sbatch_output = Path(sbatch_output)
        self.use_tokenized_cache = use_tokenized_cache
        self.num_epochs = num_epochs
        self.safety_margin = safety_margin
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.description = description
        self.optimizer = optimizer

        self.config_output.mkdir(parents=True, exist_ok=True)
        self.sbatch_output.mkdir(parents=True, exist_ok=True)

    def validate_inputs(self) -> None:
        """Validate all input parameters."""
        if self.model not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{self.model}' not found.\n{list_available_models()}"
            )

        if self.finetuning_type not in FINETUNING_CONFIGS:
            available = ", ".join(sorted(FINETUNING_CONFIGS.keys()))
            raise ValueError(
                f"Finetuning type '{self.finetuning_type}' not found. Available: {available}"
            )

        model_config = MODEL_REGISTRY[self.model]
        min_gpus = model_config["min_gpus"]
        if self.num_gpus < min_gpus:
            raise ValueError(f"Model {self.model} requires at least {min_gpus} GPUs")

        if self.dataset != "all":
            datasets = self._get_available_datasets()
            if self.dataset not in datasets:
                available = ", ".join(sorted(datasets))
                raise ValueError(
                    f"Dataset '{self.dataset}' not found. Available: {available}"
                )

    @staticmethod
    def _get_available_datasets() -> set:
        """Get set of available dataset names."""
        datasets = set()
        for key in DATASET_VALIDATION_MAP.keys():
            parts = key.split("_")
            if parts[0] == "Youtube":
                dataset_name = "_".join(parts[:3])
            else:
                dataset_name = parts[0]
            datasets.add(dataset_name)
        return datasets

    def generate_config(self) -> ConfigGenerator:
        """Generate training configuration."""
        return ConfigGenerator(
            model_key=self.model,
            dataset_name=self.dataset,
            finetuning_type=self.finetuning_type,
            dataset_types=self.dataset_types,
            num_gpus=self.num_gpus,
            gpu_vram_gb=self.gpu_vram_gb,
            dataset_dir=self.dataset_dir,
            output_base=self.output_base,
            hf_cache=self.hf_cache,
            use_tokenized_cache=self.use_tokenized_cache,
            num_epochs=self.num_epochs,
            safety_margin=self.safety_margin,
            enable_gradient_checkpointing=self.enable_gradient_checkpointing,
            description=self.description,
            optimizer=self.optimizer,
        )

    def save_config(self, config_gen: ConfigGenerator) -> str:
        """Save configuration to YAML file."""
        # Generate config first to populate optimal hyperparameters
        config = config_gen.generate()

        # Use the run_name from the generated config as filename
        run_name = config["run_name"]
        config_name = f"{run_name}.yaml"
        config_path = self.config_output / config_name

        # Save using the already-generated config
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            f.write(config_gen.to_yaml(config))

        print(f"✓ Config saved to: {config_path}")
        return str(config_path)

    def create_sbatch(self, config_path: str) -> str:
        """Create SLURM sbatch script."""
        config_stem = Path(config_path).stem
        sbatch_name = f"{config_stem}.sbatch"
        sbatch_path = self.sbatch_output / sbatch_name

        # SLURM job name has length limits, but use full name for log files
        job_name = config_stem[:50]  # SLURM has limits on job name length
        model_config = MODEL_REGISTRY.get(self.model, {})

        sbatch_gen = SbatchGenerator(
            config_path=config_path,
            num_gpus=self.num_gpus,
            job_name=job_name,
            output_dir="/usr/stud/falu/code/LLaMA-Factory/logs",
            hf_cache=self.hf_cache,
            gpu_vram_gb=self.gpu_vram_gb,
            model_params_b=model_config.get("params_b", 2.0),
            log_file_name=config_stem,  # Use full run name for log files
        )
        sbatch_gen.save(str(sbatch_path))
        return str(sbatch_path)


def main():
    parser = argparse.ArgumentParser(description="VLM training with SLURM")

    parser.add_argument("--model", type=str, help="Model to train (e.g., qwen3_vl_2b)")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name")
    parser.add_argument(
        "--dataset_type",
        type=str,
        nargs="+",
        help="Dataset types (single_image, multi_image_single_turn, multi_image_multi_turn)",
    )
    parser.add_argument(
        "--finetuning_type",
        type=str,
        default="full",
        choices=["full", "lora", "qlora"],
        help="Finetuning method",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--gpu_vram", type=int, default=48, help="GPU VRAM in GB")
    parser.add_argument("--sbatch", action="store_true", help="Generate SLURM script")
    parser.add_argument("--submit", action="store_true", help="Submit SLURM job")
    parser.add_argument("--preview", action="store_true", help="Preview config")
    parser.add_argument(
        "--list_models", action="store_true", help="List available models"
    )
    parser.add_argument("--list_datasets", action="store_true", help="List datasets")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/storage/user/falu/vis/processed_llamafactory",
    )
    parser.add_argument(
        "--output_base", type=str, default="/storage/user/falu/trained_models"
    )
    parser.add_argument("--config_output", type=str, default="autoconfig/configs")
    parser.add_argument("--sbatch_output", type=str, default="autoconfig/sbatch")
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable tokenized dataset caching"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=6, help="Number of training epochs"
    )
    parser.add_argument(
        "--safety_margin",
        type=float,
        default=0.75,
        help="Memory safety margin (0.6-0.8 recommended, lower=more conservative)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more memory but slightly faster)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional descriptive identifier for the run (e.g., 'baseline', 'test_lr', 'v2')",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adamw_torch", "adamw_bnb_8bit", "adamw_8bit", "adafactor"],
        help="Override optimizer (default: auto-select based on finetuning type)",
    )

    args = parser.parse_args()

    if args.list_models:
        print(list_available_models())
        sys.exit(0)
    if args.list_datasets:
        print(list_available_datasets())
        sys.exit(0)

    if not args.model:
        parser.print_help()
        print("\n❌ Error: --model is required")
        sys.exit(1)

    try:
        orchestrator = TrainingOrchestrator(
            model=args.model,
            dataset=args.dataset,
            finetuning_type=args.finetuning_type,
            dataset_types=args.dataset_type,
            num_gpus=args.num_gpus,
            gpu_vram_gb=args.gpu_vram,
            dataset_dir=args.dataset_dir,
            output_base=args.output_base,
            config_output=args.config_output,
            sbatch_output=args.sbatch_output,
            use_tokenized_cache=not args.no_cache,
            num_epochs=args.num_epochs,
            safety_margin=args.safety_margin,
            enable_gradient_checkpointing=not args.no_gradient_checkpointing,
            description=args.description,
            optimizer=args.optimizer,
        )
        orchestrator.validate_inputs()

        config_gen = orchestrator.generate_config()

        if args.preview:
            config_gen.print_config()
            sys.exit(0)

        config_path = orchestrator.save_config(config_gen)

        if args.sbatch or args.submit:
            sbatch_path = orchestrator.create_sbatch(config_path)
            print(f"Config:  {config_path}")
            print(f"SLURM:   {sbatch_path}")
            print(f"\nSubmit with: sbatch {sbatch_path}")

            if args.submit:
                result = subprocess.run(
                    ["sbatch", sbatch_path], capture_output=True, text=True
                )
                print(f"\n{result.stdout}")
                if result.returncode != 0:
                    print(result.stderr)
                    sys.exit(1)
        else:
            sbatch_path = orchestrator.create_sbatch(config_path)
            print(f"Config:  {config_path}")
            print(f"SLURM:   {sbatch_path}")

        print("\n✓ Training job ready!")

    except ValueError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
