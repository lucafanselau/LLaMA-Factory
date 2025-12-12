#!/usr/bin/env python3
"""Interactive TUI for merging LoRA adapters with base models."""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from merge_lora import MergeConfigGenerator, MergeOrchestrator
from train_config import MODEL_REGISTRY

console = Console()

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


def get_checkpoint_info(checkpoint_path: Path, orchestrator: MergeOrchestrator) -> dict:
    """Get detailed info about a checkpoint."""
    run_path = checkpoint_path.parent
    run_name = run_path.name

    # Extract step
    match = re.match(r"checkpoint-(\d+)", checkpoint_path.name)
    step = int(match.group(1)) if match else 0

    # Check if merged
    is_merged = orchestrator.is_already_merged(checkpoint_path)

    # Try to get eval_loss from trainer_state.json
    eval_loss = None
    trainer_state_path = run_path / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            with open(trainer_state_path) as f:
                state = json.load(f)
            for entry in state.get("log_history", []):
                if entry.get("step") == step and "eval_loss" in entry:
                    eval_loss = entry["eval_loss"]
                    break
        except (json.JSONDecodeError, KeyError):
            pass

    return {
        "path": checkpoint_path,
        "run_name": run_name,
        "step": step,
        "eval_loss": eval_loss,
        "is_merged": is_merged,
    }


def preview_merge(config_gen: MergeConfigGenerator) -> None:
    """Display merge config preview."""
    config = config_gen.generate()
    yaml_str = config_gen.to_yaml(config)

    table = Table(title="Merge Summary", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Base Model", config_gen.model_config["model_name_or_path"])
    table.add_row("Model Key", config_gen.model_key)
    table.add_row("Template", config_gen.model_config["template"])
    table.add_row("Checkpoint", f"step {config_gen.checkpoint_step}")
    table.add_row("Export Name", config_gen.export_name)
    table.add_row("Export Dir", str(config_gen.output_base / config_gen.export_name))

    console.print()
    console.print(table)
    console.print()

    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Generated Merge Config", border_style="cyan"))


def show_runs(orchestrator: MergeOrchestrator) -> None:
    """Display all LoRA/QLoRA runs in a table."""
    runs = orchestrator.find_lora_runs()

    if not runs:
        console.print("[yellow]No LoRA/QLoRA runs found.[/yellow]")
        return

    table = Table(title=f"LoRA/QLoRA Runs ({len(runs)} found)")
    table.add_column("Run Name", style="cyan")
    table.add_column("Checkpoints", style="green", justify="right")
    table.add_column("Best Step", style="yellow", justify="right")
    table.add_column("Best Loss", style="magenta", justify="right")
    table.add_column("Merged", style="dim", justify="center")

    for run in runs:
        checkpoints = orchestrator.find_checkpoints(run)
        best = orchestrator.find_best_checkpoint(run)

        best_step = "-"
        best_loss = "-"
        merged = "-"

        if best:
            info = get_checkpoint_info(best, orchestrator)
            best_step = str(info["step"])
            if info["eval_loss"]:
                best_loss = f"{info['eval_loss']:.4f}"
            merged = "✓" if info["is_merged"] else ""

        table.add_row(
            run.name,
            str(len(checkpoints)),
            best_step,
            best_loss,
            merged,
        )

    console.print()
    console.print(table)
    console.print()


def select_runs(orchestrator: MergeOrchestrator) -> list[Path]:
    """Interactive run selection."""
    runs = orchestrator.find_lora_runs()

    if not runs:
        console.print("[yellow]No LoRA/QLoRA runs found.[/yellow]")
        return []

    choices = []
    for run in runs:
        checkpoints = orchestrator.find_checkpoints(run)
        best = orchestrator.find_best_checkpoint(run)

        label = run.name
        if best:
            info = get_checkpoint_info(best, orchestrator)
            if info["is_merged"]:
                label += " [merged]"
            if info["eval_loss"]:
                label += f" (loss: {info['eval_loss']:.4f})"

        choices.append(questionary.Choice(label, value=run))

    selected = questionary.checkbox(
        "Select runs to merge:",
        choices=choices,
        style=custom_style,
    ).ask()

    return selected or []


def select_checkpoint_mode() -> str:
    """Select which checkpoint to use."""
    mode = questionary.select(
        "Which checkpoint to merge?",
        choices=[
            questionary.Choice("Best (lowest eval_loss)", "best"),
            questionary.Choice("Latest (highest step)", "latest"),
            questionary.Choice("All checkpoints", "all"),
            questionary.Choice("Select specific checkpoint", "select"),
        ],
        style=custom_style,
    ).ask()

    return mode or "best"


def select_specific_checkpoint(
    run_path: Path, orchestrator: MergeOrchestrator
) -> Optional[Path]:
    """Select a specific checkpoint from a run."""
    checkpoints = orchestrator.find_checkpoints(run_path)

    if not checkpoints:
        console.print("[yellow]No checkpoints found in this run.[/yellow]")
        return None

    choices = []
    for cp in checkpoints:
        info = get_checkpoint_info(cp, orchestrator)
        label = f"checkpoint-{info['step']}"
        if info["eval_loss"]:
            label += f" (loss: {info['eval_loss']:.4f})"
        if info["is_merged"]:
            label += " [merged]"
        choices.append(questionary.Choice(label, value=cp))

    selected = questionary.select(
        "Select checkpoint:",
        choices=choices,
        style=custom_style,
    ).ask()

    return selected


def process_merges(
    checkpoints: list[Path],
    orchestrator: MergeOrchestrator,
    submit: bool = False,
    force: bool = False,
) -> None:
    """Process multiple merge operations (one SLURM job per checkpoint)."""
    console.print()
    console.print(f"[bold]Processing {len(checkpoints)} checkpoint(s)...[/bold]")
    console.print()

    results = {"success": 0, "skipped": 0, "error": 0}

    for cp in checkpoints:
        console.print(f"[cyan]🔄 {cp.parent.name}/{cp.name}[/cyan]")

        if not force and orchestrator.is_already_merged(cp):
            console.print("   [dim]⏭ Already merged, skipping[/dim]")
            results["skipped"] += 1
            continue

        try:
            config_path, config_gen = orchestrator.generate_merge_config(cp)
            sbatch_path = orchestrator.generate_sbatch(config_path, config_gen)

            console.print(f"   [green]✓ Config: {config_path}[/green]")
            console.print(f"   [green]✓ SLURM:  {sbatch_path}[/green]")

            if submit:
                result = subprocess.run(
                    ["sbatch", sbatch_path],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    console.print(f"   [green]✓ {result.stdout.strip()}[/green]")
                    results["success"] += 1
                else:
                    console.print(f"   [red]✗ {result.stderr}[/red]")
                    results["error"] += 1
            else:
                results["success"] += 1

        except ValueError as e:
            console.print(f"   [red]✗ Error: {e}[/red]")
            results["error"] += 1

        console.print()

    # Summary
    console.print("[bold]Summary:[/bold]")
    console.print(f"  ✓ Success: {results['success']}")
    console.print(f"  ⏭ Skipped: {results['skipped']}")
    console.print(f"  ✗ Errors:  {results['error']}")


def process_batch_merges(
    runs_with_checkpoints: list[tuple[Path, list[Path]]],
    orchestrator: MergeOrchestrator,
    submit: bool = False,
    force: bool = False,
) -> None:
    """Process merge operations with batch mode (one SLURM job per run)."""
    console.print()
    total_checkpoints = sum(len(cps) for _, cps in runs_with_checkpoints)
    console.print(
        f"[bold]Batch processing {total_checkpoints} checkpoints "
        f"across {len(runs_with_checkpoints)} run(s)...[/bold]"
    )
    console.print()

    results = {"runs_success": 0, "runs_error": 0, "checkpoints": 0}

    for run_path, checkpoints in runs_with_checkpoints:
        console.print(
            f"[cyan]📦 {run_path.name} ({len(checkpoints)} checkpoints)[/cyan]"
        )

        try:
            sbatch_path = orchestrator.merge_run_batch(
                run_path,
                checkpoints,
                submit=submit,
                force=force,
            )

            if sbatch_path:
                results["runs_success"] += 1
                results["checkpoints"] += len(checkpoints)
            else:
                console.print("   [dim]⏭ All checkpoints already merged[/dim]")

        except ValueError as e:
            console.print(f"   [red]✗ Error: {e}[/red]")
            results["runs_error"] += 1

        console.print()

    # Summary
    console.print("[bold]Summary:[/bold]")
    console.print(f"  📦 Batch jobs created: {results['runs_success']}")
    console.print(f"  📋 Total checkpoints: {results['checkpoints']}")
    console.print(f"  ✗ Errors: {results['runs_error']}")


def main_menu() -> None:
    """Main interactive menu."""
    console.print(
        Panel.fit(
            "[bold cyan]LoRA Merge Tool[/bold cyan]\n"
            "[dim]Merge LoRA adapters with base models for vLLM inference[/dim]",
            border_style="cyan",
        )
    )

    orchestrator = MergeOrchestrator()

    while True:
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "📋 List LoRA/QLoRA Runs",
                "🔀 Merge Single Run",
                "🔀 Batch Merge Multiple Runs",
                "👀 Preview Merge Config",
                "❌ Exit",
            ],
            style=custom_style,
        ).ask()

        if action is None or "Exit" in action:
            console.print("[dim]Goodbye![/dim]")
            break

        elif "List" in action:
            show_runs(orchestrator)
            questionary.press_any_key_to_continue(
                "Press any key to continue...",
                style=custom_style,
            ).ask()

        elif "Single Run" in action:
            merge_single_run(orchestrator)

        elif "Batch Merge" in action:
            batch_merge(orchestrator)

        elif "Preview" in action:
            preview_single(orchestrator)


def merge_single_run(orchestrator: MergeOrchestrator) -> None:
    """Merge a single run interactively."""
    runs = orchestrator.find_lora_runs()

    if not runs:
        console.print("[yellow]No LoRA/QLoRA runs found.[/yellow]")
        return

    # Select run
    choices = [questionary.Choice(run.name, value=run) for run in runs]
    run = questionary.select(
        "Select run to merge:",
        choices=choices,
        style=custom_style,
    ).ask()

    if run is None:
        return

    # Select checkpoint mode
    mode = select_checkpoint_mode()

    checkpoints = []
    if mode == "best":
        cp = orchestrator.find_best_checkpoint(run)
        if cp:
            checkpoints = [cp]
    elif mode == "latest":
        cp = orchestrator.find_latest_checkpoint(run)
        if cp:
            checkpoints = [cp]
    elif mode == "all":
        checkpoints = orchestrator.find_checkpoints(run)
    elif mode == "select":
        cp = select_specific_checkpoint(run, orchestrator)
        if cp:
            checkpoints = [cp]

    if not checkpoints:
        console.print("[yellow]No checkpoints found.[/yellow]")
        return

    # Show preview
    console.print()
    for cp in checkpoints:
        info = get_checkpoint_info(cp, orchestrator)
        merged_str = "[merged]" if info["is_merged"] else ""
        loss_str = f"loss: {info['eval_loss']:.4f}" if info["eval_loss"] else ""
        console.print(f"  • checkpoint-{info['step']} {loss_str} {merged_str}")
    console.print()

    # For "all" mode with multiple checkpoints, offer batch processing
    use_batch = False
    if mode == "all" and len(checkpoints) > 1:
        use_batch = questionary.confirm(
            f"Create single SLURM job for all {len(checkpoints)} checkpoints? (recommended)",
            default=True,
            style=custom_style,
        ).ask()
        if use_batch is None:
            return

    # Confirm and select action
    action = questionary.select(
        "Action:",
        choices=[
            "Generate configs & SLURM scripts only",
            "Generate and submit to SLURM",
            "Cancel",
        ],
        style=custom_style,
    ).ask()

    if action is None or "Cancel" in action:
        return

    submit = "submit" in action.lower()
    force = False

    # Check for already merged
    merged_count = sum(1 for cp in checkpoints if orchestrator.is_already_merged(cp))
    if merged_count > 0:
        force = questionary.confirm(
            f"{merged_count} checkpoint(s) already merged. Re-merge them?",
            default=False,
            style=custom_style,
        ).ask()

    if use_batch:
        process_batch_merges(
            [(run, checkpoints)], orchestrator, submit=submit, force=force or False
        )
    else:
        process_merges(checkpoints, orchestrator, submit=submit, force=force or False)

    questionary.press_any_key_to_continue(
        "Press any key to continue...",
        style=custom_style,
    ).ask()


def batch_merge(orchestrator: MergeOrchestrator) -> None:
    """Batch merge multiple runs."""
    selected_runs = select_runs(orchestrator)

    if not selected_runs:
        return

    # Select checkpoint mode
    mode = select_checkpoint_mode()
    if mode == "select":
        console.print(
            "[yellow]Cannot use 'select specific' for batch merge. Using 'best' instead.[/yellow]"
        )
        mode = "best"

    # Collect checkpoints (organized by run for batch mode)
    runs_with_checkpoints = []  # For batch mode: [(run, [checkpoints]), ...]
    all_checkpoints = []  # For non-batch mode

    for run in selected_runs:
        if mode == "best":
            cp = orchestrator.find_best_checkpoint(run)
            if cp:
                all_checkpoints.append(cp)
                runs_with_checkpoints.append((run, [cp]))
        elif mode == "latest":
            cp = orchestrator.find_latest_checkpoint(run)
            if cp:
                all_checkpoints.append(cp)
                runs_with_checkpoints.append((run, [cp]))
        elif mode == "all":
            checkpoints = orchestrator.find_checkpoints(run)
            if checkpoints:
                all_checkpoints.extend(checkpoints)
                runs_with_checkpoints.append((run, checkpoints))

    if not all_checkpoints:
        console.print("[yellow]No checkpoints found.[/yellow]")
        return

    console.print()
    console.print(
        f"[bold]Found {len(all_checkpoints)} checkpoint(s) across {len(runs_with_checkpoints)} run(s)[/bold]"
    )
    console.print()

    # For "all" mode, offer batch processing (one SLURM job per run)
    use_batch = False
    if mode == "all" and len(all_checkpoints) > len(runs_with_checkpoints):
        use_batch = questionary.confirm(
            "Create one SLURM job per run? (recommended for 'all checkpoints' mode)",
            default=True,
            style=custom_style,
        ).ask()
        if use_batch is None:
            return

    # Confirm and select action
    action = questionary.select(
        "Action:",
        choices=[
            "Generate configs & SLURM scripts only",
            "Generate and submit all to SLURM",
            "Cancel",
        ],
        style=custom_style,
    ).ask()

    if action is None or "Cancel" in action:
        return

    submit = "submit" in action.lower()
    force = False

    # Check for already merged
    merged_count = sum(
        1 for cp in all_checkpoints if orchestrator.is_already_merged(cp)
    )
    if merged_count > 0:
        force = questionary.confirm(
            f"{merged_count} checkpoint(s) already merged. Re-merge them?",
            default=False,
            style=custom_style,
        ).ask()

    if use_batch:
        process_batch_merges(
            runs_with_checkpoints, orchestrator, submit=submit, force=force or False
        )
    else:
        process_merges(
            all_checkpoints, orchestrator, submit=submit, force=force or False
        )

    questionary.press_any_key_to_continue(
        "Press any key to continue...",
        style=custom_style,
    ).ask()


def preview_single(orchestrator: MergeOrchestrator) -> None:
    """Preview merge config for a single checkpoint."""
    runs = orchestrator.find_lora_runs()

    if not runs:
        console.print("[yellow]No LoRA/QLoRA runs found.[/yellow]")
        return

    # Select run
    choices = [questionary.Choice(run.name, value=run) for run in runs]
    run = questionary.select(
        "Select run:",
        choices=choices,
        style=custom_style,
    ).ask()

    if run is None:
        return

    # Select checkpoint
    cp = select_specific_checkpoint(run, orchestrator)
    if cp is None:
        return

    try:
        config_gen = MergeConfigGenerator(
            str(cp),
            output_base=str(orchestrator.merged_models_dir),
        )
        preview_merge(config_gen)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")

    questionary.press_any_key_to_continue(
        "Press any key to continue...",
        style=custom_style,
    ).ask()


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(0)
