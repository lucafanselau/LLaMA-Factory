#!/usr/bin/env python3
"""Interactive TUI for dispatching VLM training jobs."""

import subprocess
import sys

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from config_generator import ConfigGenerator
from train_config import (
    DATASET_VALIDATION_MAP,
    FINETUNING_CONFIGS,
    MODEL_REGISTRY,
)

console = Console()

# Custom style for questionary
custom_style = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:cyan bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
        ("separator", "fg:gray"),
        ("instruction", "fg:gray italic"),
    ]
)


def get_available_datasets() -> list[str]:
    """Extract unique dataset names."""
    datasets = set()
    for key in DATASET_VALIDATION_MAP.keys():
        parts = key.split("_")
        if parts[0] == "Youtube":
            datasets.add("_".join(parts[:3]))
        else:
            datasets.add(parts[0])
    return sorted(datasets)


def preview_config(config_gen: ConfigGenerator) -> None:
    """Display config preview in a rich panel."""
    config = config_gen.generate()
    yaml_str = config_gen.to_yaml(config)

    # Summary table
    table = Table(title="Training Summary", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    table.add_row(
        "Model", f"{config_gen.model_key} ({config_gen.model_config['params_b']}B)"
    )
    table.add_row("Finetuning", config_gen.finetuning_type)
    table.add_row("GPUs", f"{config_gen.num_gpus}x {config_gen.gpu_vram_gb}GB")
    table.add_row("Batch Size", str(config_gen.optimal["effective_batch_size"]))

    if config_gen.optimal.get("steps_per_epoch"):
        table.add_row("Steps/Epoch", str(config_gen.optimal["steps_per_epoch"]))
        table.add_row("Eval Every", f"{config_gen.optimal['eval_steps']} steps")
        evals = (
            config_gen.optimal["steps_per_epoch"] // config_gen.optimal["eval_steps"]
        )
        table.add_row("Evals/Epoch", f"~{evals}")

    if config_gen._fast_eval_info:
        table.add_row(
            "Fast Eval",
            f"{len(config_gen._fast_eval_info)} datasets @ {config_gen.eval_samples_per_dataset} samples",
        )

    ds = (
        "ZeRO-" + config_gen.optimal["deepspeed_config"].split("_z")[1].split("_")[0]
        if config_gen.optimal["deepspeed_config"]
        else "DDP"
    )
    table.add_row("DeepSpeed", ds)

    console.print()
    console.print(table)
    console.print()

    # YAML config
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Generated Config", border_style="cyan"))


def main_menu() -> None:
    """Main interactive menu."""
    console.print(
        Panel.fit(
            "[bold cyan]VLM Training Job Dispatcher[/bold cyan]\n"
            "[dim]Interactive training configuration and submission[/dim]",
            border_style="cyan",
        )
    )

    while True:
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "🚀 Create & Submit Training Job",
                "👀 Preview Configuration",
                "📋 List Available Models",
                "📂 List Available Datasets",
                "❌ Exit",
            ],
            style=custom_style,
        ).ask()

        if action is None or "Exit" in action:
            console.print("[dim]Goodbye![/dim]")
            break
        elif "List Available Models" in action:
            show_models()
        elif "List Available Datasets" in action:
            show_datasets()
        elif "Preview" in action or "Submit" in action:
            config = configure_job()
            if config:
                if "Submit" in action:
                    submit_job(config)
                else:
                    preview_only(config)


def show_models() -> None:
    """Display available models in a table."""
    table = Table(title="Available Models")
    table.add_column("Key", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Family", style="yellow")
    table.add_column("HuggingFace Path", style="dim")

    for key, cfg in sorted(MODEL_REGISTRY.items()):
        size = f"{cfg['params_b']}B"
        if cfg["active_params_b"] != cfg["params_b"]:
            size += f" (MoE, {cfg['active_params_b']}B active)"
        table.add_row(key, size, cfg["family"], cfg["model_name_or_path"])

    console.print()
    console.print(table)
    console.print()


def show_datasets() -> None:
    """Display available datasets."""
    datasets = get_available_datasets()
    console.print()
    console.print(
        Panel("\n".join(f"  • {d}" for d in datasets), title="Available Datasets")
    )
    console.print()


def configure_job() -> dict | None:
    """Interactive job configuration."""
    console.print()

    # Model selection
    model_choices = [
        f"{key} ({cfg['params_b']}B)" for key, cfg in sorted(MODEL_REGISTRY.items())
    ]
    model_answer = questionary.select(
        "Select model:",
        choices=model_choices,
        style=custom_style,
    ).ask()

    if model_answer is None:
        return None
    model = model_answer.split(" (")[0]

    # Dataset selection
    datasets = ["all"] + get_available_datasets()
    dataset = questionary.select(
        "Select dataset:",
        choices=datasets,
        default="all",
        style=custom_style,
    ).ask()

    if dataset is None:
        return None

    # Dataset types
    dataset_types = questionary.checkbox(
        "Select dataset types:",
        choices=[
            questionary.Choice("single_image", checked=True),
            questionary.Choice("multi_image_single_turn", checked=True),
            questionary.Choice("multi_image_multi_turn", checked=True),
        ],
        style=custom_style,
    ).ask()

    if not dataset_types:
        dataset_types = [
            "single_image",
            "multi_image_single_turn",
            "multi_image_multi_turn",
        ]

    # Finetuning type
    finetuning_type = questionary.select(
        "Select finetuning type:",
        choices=list(FINETUNING_CONFIGS.keys()),
        default="full",
        style=custom_style,
    ).ask()

    if finetuning_type is None:
        return None

    # GPU configuration
    num_gpus = questionary.text(
        "Number of GPUs:",
        default="1",
        validate=lambda x: x.isdigit() and int(x) > 0,
        style=custom_style,
    ).ask()

    if num_gpus is None:
        return None

    gpu_vram = questionary.select(
        "GPU VRAM (GB):",
        choices=["24", "40", "48", "80"],
        default="48",
        style=custom_style,
    ).ask()

    if gpu_vram is None:
        return None

    # Training parameters
    num_epochs = questionary.text(
        "Number of epochs:",
        default="6",
        validate=lambda x: x.isdigit() and int(x) > 0,
        style=custom_style,
    ).ask()

    if num_epochs is None:
        return None

    eval_samples = questionary.text(
        "Samples per eval dataset (for fast validation):",
        default="500",
        validate=lambda x: x.isdigit() and int(x) > 0,
        style=custom_style,
    ).ask()

    if eval_samples is None:
        return None

    return {
        "model": model,
        "dataset": dataset,
        "dataset_types": dataset_types,
        "finetuning_type": finetuning_type,
        "num_gpus": int(num_gpus),
        "gpu_vram": int(gpu_vram),
        "num_epochs": int(num_epochs),
        "eval_samples": int(eval_samples),
    }


def create_config_generator(config: dict) -> ConfigGenerator:
    """Create ConfigGenerator from config dict."""
    return ConfigGenerator(
        model_key=config["model"],
        dataset_name=config["dataset"],
        finetuning_type=config["finetuning_type"],
        dataset_types=config["dataset_types"],
        num_gpus=config["num_gpus"],
        gpu_vram_gb=config["gpu_vram"],
        num_epochs=config["num_epochs"],
        eval_samples_per_dataset=config["eval_samples"],
    )


def preview_only(config: dict) -> None:
    """Preview configuration without submitting."""
    config_gen = create_config_generator(config)
    preview_config(config_gen)

    questionary.press_any_key_to_continue(
        "Press any key to return to menu...",
        style=custom_style,
    ).ask()


def submit_job(config: dict) -> None:
    """Submit training job after confirmation."""
    config_gen = create_config_generator(config)
    preview_config(config_gen)

    # Confirm submission
    confirm = questionary.confirm(
        "Submit this job?",
        default=False,
        style=custom_style,
    ).ask()

    if not confirm:
        console.print("[yellow]Job cancelled.[/yellow]")
        return

    # Build command
    cmd = [
        sys.executable,
        "train_mllm.py",
        "--model",
        config["model"],
        "--dataset",
        config["dataset"],
        "--finetuning_type",
        config["finetuning_type"],
        "--num_gpus",
        str(config["num_gpus"]),
        "--gpu_vram",
        str(config["gpu_vram"]),
        "--num_epochs",
        str(config["num_epochs"]),
        "--eval_samples",
        str(config["eval_samples"]),
        "--sbatch",
    ]

    if config["dataset_types"]:
        cmd.extend(["--dataset_type"] + config["dataset_types"])

    # Ask whether to submit or just generate
    submit_action = questionary.select(
        "Action:",
        choices=[
            "Generate config & sbatch only",
            "Generate and submit to SLURM",
        ],
        style=custom_style,
    ).ask()

    if submit_action is None:
        return

    if "submit to SLURM" in submit_action:
        cmd.append("--submit")

    console.print()
    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    console.print()

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        console.print()
        console.print("[bold green]✓ Job dispatched successfully![/bold green]")
    else:
        console.print()
        console.print("[bold red]✗ Job dispatch failed[/bold red]")

    questionary.press_any_key_to_continue(
        "Press any key to return to menu...",
        style=custom_style,
    ).ask()


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(0)
