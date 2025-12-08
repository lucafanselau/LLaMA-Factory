#!/usr/bin/env python3
"""Vision-language model training orchestrator with SLURM support."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
from config_generator import ConfigGenerator
from sbatch_generator import SbatchGenerator
from train_config import (
    MODEL_REGISTRY,
    list_available_models,
    list_available_datasets,
    FINETUNING_CONFIGS,
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
        dataset_dir: str = "/storage/user/falu/vis/processed",
        output_base: str = "/storage/user/falu/trained_models",
        hf_cache: str = "/storage/user/falu/.cache/huggingface",
        config_output: str = "training_configs",
        sbatch_output: str = "sbatch_scripts",
        use_tokenized_cache: bool = True,
    ):
        self.model = model
        self.dataset = dataset
        self.finetuning_type = finetuning_type
        self.dataset_types = dataset_types
        self.num_gpus = num_gpus
        self.dataset_dir = dataset_dir
        self.output_base = output_base
        self.hf_cache = hf_cache
        self.config_output = Path(config_output)
        self.sbatch_output = Path(sbatch_output)
        self.use_tokenized_cache = use_tokenized_cache

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
        if self.num_gpus > model_config["max_gpus"]:
            raise ValueError(
                f"Model {self.model} supports max {model_config['max_gpus']} GPUs, but {self.num_gpus} requested"
            )

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
        from train_config import DATASET_VALIDATION_MAP

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
            dataset_dir=self.dataset_dir,
            output_base=self.output_base,
            hf_cache=self.hf_cache,
            use_tokenized_cache=self.use_tokenized_cache,
        )

    def save_config(self, config_gen: ConfigGenerator) -> str:
        """Save configuration to YAML file."""
        dataset_str = "_".join(config_gen.dataset_types)
        config_name = (
            f"{self.model}_{self.dataset}_{dataset_str}_{self.finetuning_type}.yaml"
        )
        config_path = self.config_output / config_name

        config_gen.save(str(config_path))
        return str(config_path)

    def create_sbatch(self, config_path: str) -> str:
        """Create SLURM sbatch script with smart resource allocation."""
        config_stem = Path(config_path).stem
        sbatch_name = f"{config_stem}.sbatch"
        sbatch_path = self.sbatch_output / sbatch_name

        job_name = f"train-{config_stem[:50]}"

        # Get model tier for smart resource allocation
        model_config = MODEL_REGISTRY.get(self.model, {})
        model_tier = model_config.get("tier", "small")

        # Get hyperparams to extract preprocessing workers and batch size
        from train_config import get_hyperparams

        hyperparams = get_hyperparams(model_tier)
        preprocessing_num_workers = hyperparams.get("preprocessing_num_workers", 8)
        batch_size = hyperparams.get("batch_size", 2)

        sbatch_gen = SbatchGenerator(
            config_path=config_path,
            num_gpus=self.num_gpus,
            job_name=job_name,
            output_dir="/usr/stud/falu/code/LLaMA-Factory/logs",
            hf_cache=self.hf_cache,
            model_tier=model_tier,
            preprocessing_num_workers=preprocessing_num_workers,
            batch_size=batch_size,
        )
        sbatch_gen.save(str(sbatch_path))
        return str(sbatch_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Vision-language model training with SLURM"
    )

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
    parser.add_argument("--sbatch", action="store_true", help="Generate SLURM script")
    parser.add_argument("--submit", action="store_true", help="Submit SLURM job")
    parser.add_argument("--preview", action="store_true", help="Preview config")
    parser.add_argument(
        "--list_models", action="store_true", help="List available models"
    )
    parser.add_argument("--list_datasets", action="store_true", help="List datasets")
    parser.add_argument(
        "--dataset_dir", type=str, default="/storage/user/falu/vis/processed"
    )
    parser.add_argument(
        "--output_base", type=str, default="/storage/user/falu/trained_models"
    )
    parser.add_argument("--config_output", type=str, default="training_configs")
    parser.add_argument("--sbatch_output", type=str, default="sbatch_scripts")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable tokenized dataset caching (reprocess each run)",
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
            dataset_dir=args.dataset_dir,
            output_base=args.output_base,
            config_output=args.config_output,
            sbatch_output=args.sbatch_output,
            use_tokenized_cache=not args.no_cache,
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
