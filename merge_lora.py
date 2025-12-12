#!/usr/bin/env python3
"""LoRA adapter merger for LlamaFactory trained models."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

import yaml

from train_config import MODEL_REGISTRY


class MergeConfigGenerator:
    """Generate merge configs for LoRA adapters."""

    def __init__(
        self,
        adapter_path: str,
        output_base: str = "/storage/user/falu/merged_models",
        hf_cache: str = "/storage/user/falu/.cache/huggingface",
        export_size: int = 5,
        export_device: str = "cpu",
    ):
        self.adapter_path = Path(adapter_path)
        self.output_base = Path(output_base)
        self.hf_cache = hf_cache
        self.export_size = export_size
        self.export_device = export_device

        # Parse adapter path to extract metadata
        self.run_name, self.checkpoint_step = self._parse_adapter_path()
        self.model_key, self.finetuning_type, self.description = self._parse_run_name()

        # Get model config from registry
        if self.model_key not in MODEL_REGISTRY:
            raise ValueError(f"Model '{self.model_key}' not found in MODEL_REGISTRY")
        self.model_config = MODEL_REGISTRY[self.model_key]

        # Generate export directory name
        self.export_name = self._generate_export_name()

    def _parse_adapter_path(self) -> Tuple[str, int]:
        """Extract run name and checkpoint step from adapter path."""
        checkpoint_dir = self.adapter_path.name
        run_name = self.adapter_path.parent.name

        match = re.match(r"checkpoint-(\d+)", checkpoint_dir)
        if not match:
            raise ValueError(f"Invalid checkpoint directory: {checkpoint_dir}")
        step = int(match.group(1))

        return run_name, step

    def _parse_run_name(self) -> Tuple[str, str, Optional[str]]:
        """Parse run name to extract model key and finetuning type.

        Format: {model_key}-{description}-{ft_type}-{dataset}-{gpus}x{vram}GB-b{batch}
        or:     {model_key}-{ft_type}-{dataset}-{gpus}x{vram}GB-b{batch}
        """
        parts = self.run_name.split("-")

        # Find finetuning type position
        ft_idx = None
        for i, part in enumerate(parts):
            if part in ("lora", "qlora", "full"):
                ft_idx = i
                break

        if ft_idx is None:
            raise ValueError(
                f"Could not find finetuning type in run name: {self.run_name}"
            )

        finetuning_type = parts[ft_idx]

        # Model key is everything before description/ft_type
        # Need to find where model key ends - it's a known key in MODEL_REGISTRY
        model_key = None
        for i in range(ft_idx, 0, -1):
            candidate = "-".join(parts[:i])
            # Handle underscore variants (model keys use underscores)
            candidate_underscore = candidate.replace("-", "_")
            if candidate_underscore in MODEL_REGISTRY:
                model_key = candidate_underscore
                description = "-".join(parts[i:ft_idx]) if i < ft_idx else None
                break

        if model_key is None:
            # Try common patterns
            for known_key in MODEL_REGISTRY.keys():
                key_parts = known_key.replace("_", "-")
                if self.run_name.startswith(key_parts):
                    model_key = known_key
                    # Find where model key ends in the run name
                    key_len = len(key_parts.split("-"))
                    description = (
                        "-".join(parts[key_len:ft_idx]) if key_len < ft_idx else None
                    )
                    break

        if model_key is None:
            raise ValueError(
                f"Could not determine model key from run name: {self.run_name}"
            )

        # Clean up description
        if description == "":
            description = None

        return model_key, finetuning_type, description

    def _generate_export_name(self) -> str:
        """Generate a concise export directory name."""
        # Format: {model_key}-{description}-{ft_type}-ckpt{step}
        parts = [self.model_key.replace("_", "-")]
        if self.description:
            parts.append(self.description)
        parts.append(self.finetuning_type)
        parts.append(f"ckpt{self.checkpoint_step}")
        return "-".join(parts)

    def generate(self) -> Dict[str, Any]:
        """Generate merge config dict."""
        return {
            "model_name_or_path": self.model_config["model_name_or_path"],
            "adapter_name_or_path": str(self.adapter_path),
            "template": self.model_config["template"],
            "finetuning_type": "lora",  # Always "lora" for merging, even for qlora
            "trust_remote_code": True,
            "export_dir": str(self.output_base / self.export_name),
            "export_size": self.export_size,
            "export_device": self.export_device,
            "export_legacy_format": False,
        }

    def to_yaml(self, config: Optional[Dict[str, Any]] = None) -> str:
        """Convert config to YAML string."""
        if config is None:
            config = self.generate()

        header = (
            "### Note: DO NOT use quantized model or quantization_bit when merging\n\n"
        )
        return header + yaml.dump(config, default_flow_style=False, sort_keys=False)

    def save(self, path: str) -> None:
        """Save config to YAML file."""
        config = self.generate()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_yaml(config))
        print(f"✓ Merge config saved to: {path}")


class MergeSbatchGenerator:
    """Generate SLURM sbatch scripts for LoRA merging (CPU-only)."""

    def __init__(
        self,
        config_path: str,
        job_name: str,
        model_params_b: float = 2.0,
        output_dir: str = "/usr/stud/falu/code/LLaMA-Factory/logs",
        hf_cache: str = "/storage/user/falu/.cache/huggingface",
        time_hours: int = 4,
        cpus: int = 8,
        node_list: str = "node[15-20]",
    ):
        self.config_path = config_path
        self.job_name = job_name[:50]  # SLURM limit
        self.output_dir = output_dir
        self.hf_cache = hf_cache
        self.time_hours = time_hours
        self.cpus = cpus
        self.node_list = node_list

        # Memory estimation: ~2x model size for loading + merging
        # bf16 = 2 bytes per param, need base + adapter + merged copy
        self.memory_gb = max(32, int(model_params_b * 2 * 3) + 16)

    def generate(self) -> str:
        """Generate sbatch script for CPU-only merge."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        log_file = f"{self.output_dir}/merge-{self.job_name}-%j.out"

        hours = self.time_hours
        time_str = f"0-{hours:02d}:00:00"

        script = dedent(
            f"""\
            #!/bin/bash -l
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --nodelist={self.node_list}
            #SBATCH --cpus-per-task={self.cpus}
            #SBATCH --mem={self.memory_gb}G
            #SBATCH --time={time_str}
            #SBATCH --job-name=merge-{self.job_name}
            #SBATCH --output={log_file}
            #SBATCH --error={log_file}

            # ============================================================================
            # CPU-ONLY LORA MERGE JOB
            # ============================================================================

            set -e

            echo "Job: merge-{self.job_name}"
            echo "Job ID: ${{SLURM_JOB_ID}}"
            echo "Host: $(hostname)"
            echo "Memory: {self.memory_gb}GB"
            echo "================================"

            source /home/stud/falu/code/LLaMA-Factory/.venv/bin/activate
            export HF_HOME={self.hf_cache}

            cd /home/stud/falu/code/LLaMA-Factory
            echo "Config: {self.config_path}"
            echo "================================"

            llamafactory-cli export {self.config_path}

            if [ $? -eq 0 ]; then
                echo "✓ Merge completed successfully"
            else
                echo "✗ Merge failed"
                exit 1
            fi

            echo "Finished: $(date)"
        """
        )

        return script

    def save(self, path: str) -> None:
        """Save sbatch script to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.generate())
        Path(path).chmod(0o755)
        print(f"✓ SLURM script saved to: {path}")


class MergeBatchSbatchGenerator:
    """Generate SLURM sbatch scripts for batch LoRA merging (multiple checkpoints, single job)."""

    def __init__(
        self,
        config_paths: List[str],
        job_name: str,
        model_params_b: float = 2.0,
        output_dir: str = "/usr/stud/falu/code/LLaMA-Factory/logs",
        hf_cache: str = "/storage/user/falu/.cache/huggingface",
        time_hours_per_checkpoint: float = 1.0,
        cpus: int = 8,
        node_list: str = "node[15-20]",
    ):
        self.config_paths = config_paths
        self.job_name = job_name[:50]  # SLURM limit
        self.output_dir = output_dir
        self.hf_cache = hf_cache
        self.cpus = cpus
        self.node_list = node_list

        # Memory estimation: ~2x model size for loading + merging
        self.memory_gb = max(32, int(model_params_b * 2 * 3) + 16)

        # Time estimation: base + per checkpoint
        total_hours = max(2, int(len(config_paths) * time_hours_per_checkpoint) + 1)
        self.time_hours = min(total_hours, 48)  # Cap at 48 hours

    def generate(self) -> str:
        """Generate sbatch script for batch CPU-only merge."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        log_file = f"{self.output_dir}/merge-batch-{self.job_name}-%j.out"

        days = self.time_hours // 24
        hours = self.time_hours % 24
        time_str = f"{days}-{hours:02d}:00:00"

        # Build the merge commands
        merge_commands = []
        for i, config_path in enumerate(self.config_paths, 1):
            merge_commands.append(
                f'echo ""\n'
                f'echo "[{i}/{len(self.config_paths)}] Merging: {config_path}"\n'
                f'echo "================================"\n'
                f"llamafactory-cli export {config_path}\n"
                f"if [ $? -eq 0 ]; then\n"
                f'    echo "✓ Checkpoint {i} merged successfully"\n'
                f"else\n"
                f'    echo "✗ Checkpoint {i} failed"\n'
                f"    FAILED=$((FAILED + 1))\n"
                f"fi"
            )

        merge_section = "\n\n".join(merge_commands)

        script = f"""\
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist={self.node_list}
#SBATCH --cpus-per-task={self.cpus}
#SBATCH --mem={self.memory_gb}G
#SBATCH --time={time_str}
#SBATCH --job-name=merge-{self.job_name}
#SBATCH --output={log_file}
#SBATCH --error={log_file}

# ============================================================================
# CPU-ONLY BATCH LORA MERGE JOB
# ============================================================================

echo "Job: merge-batch-{self.job_name}"
echo "Job ID: ${{SLURM_JOB_ID}}"
echo "Host: $(hostname)"
echo "Memory: {self.memory_gb}GB"
echo "Checkpoints to merge: {len(self.config_paths)}"
echo "================================"

source /home/stud/falu/code/LLaMA-Factory/.venv/bin/activate
export HF_HOME={self.hf_cache}

cd /home/stud/falu/code/LLaMA-Factory

FAILED=0

{merge_section}

echo ""
echo "================================"
if [ $FAILED -eq 0 ]; then
    echo "✓ All {len(self.config_paths)} checkpoints merged successfully"
else
    echo "✗ $FAILED/{len(self.config_paths)} checkpoints failed"
    exit 1
fi

echo "Finished: $(date)"
"""

        return script

    def save(self, path: str) -> None:
        """Save sbatch script to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.generate())
        Path(path).chmod(0o755)
        print(f"✓ Batch SLURM script saved to: {path}")


class MergeOrchestrator:
    """Orchestrate LoRA merge operations."""

    def __init__(
        self,
        trained_models_dir: str = "/storage/user/falu/trained_models",
        merged_models_dir: str = "/storage/user/falu/merged_models",
        config_output: str = "autoconfig/merge_configs",
        sbatch_output: str = "autoconfig/merge_sbatch",
        hf_cache: str = "/storage/user/falu/.cache/huggingface",
    ):
        self.trained_models_dir = Path(trained_models_dir)
        self.merged_models_dir = Path(merged_models_dir)
        self.config_output = Path(config_output)
        self.sbatch_output = Path(sbatch_output)
        self.hf_cache = hf_cache

        self.config_output.mkdir(parents=True, exist_ok=True)
        self.sbatch_output.mkdir(parents=True, exist_ok=True)

    def find_lora_runs(self) -> List[Path]:
        """Find all LoRA/QLoRA training runs."""
        runs = []
        for run_dir in sorted(self.trained_models_dir.iterdir()):
            if run_dir.is_dir() and (
                "-lora-" in run_dir.name or "-qlora-" in run_dir.name
            ):
                runs.append(run_dir)
        return runs

    def find_checkpoints(self, run_path: Path) -> List[Path]:
        """Find all checkpoints in a run directory."""
        checkpoints = []
        for item in sorted(run_path.iterdir()):
            if item.is_dir() and item.name.startswith("checkpoint-"):
                # Verify it has adapter files
                if (item / "adapter_model.safetensors").exists() or (
                    item / "adapter_model.bin"
                ).exists():
                    checkpoints.append(item)
        return checkpoints

    def find_best_checkpoint(self, run_path: Path) -> Optional[Path]:
        """Find checkpoint with lowest eval_loss from trainer_state.json."""
        checkpoints = self.find_checkpoints(run_path)
        if not checkpoints:
            return None

        # Try to find best by eval_loss
        trainer_state_path = run_path / "trainer_state.json"
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path) as f:
                    state = json.load(f)

                # Find entries with eval_loss
                eval_entries = [
                    h
                    for h in state.get("log_history", [])
                    if "eval_loss" in h and "step" in h
                ]

                if eval_entries:
                    best = min(eval_entries, key=lambda x: x["eval_loss"])
                    best_step = best["step"]
                    best_checkpoint = run_path / f"checkpoint-{best_step}"
                    if best_checkpoint.exists():
                        return best_checkpoint
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback to latest checkpoint
        return self.find_latest_checkpoint(run_path)

    def find_latest_checkpoint(self, run_path: Path) -> Optional[Path]:
        """Find the latest checkpoint by step number."""
        checkpoints = self.find_checkpoints(run_path)
        if not checkpoints:
            return None

        # Sort by step number
        def get_step(cp: Path) -> int:
            match = re.match(r"checkpoint-(\d+)", cp.name)
            return int(match.group(1)) if match else 0

        return max(checkpoints, key=get_step)

    def is_already_merged(self, adapter_path: Path) -> bool:
        """Check if this checkpoint has already been merged."""
        try:
            config_gen = MergeConfigGenerator(
                str(adapter_path), output_base=str(self.merged_models_dir)
            )
            export_path = self.merged_models_dir / config_gen.export_name
            # Check if merged model exists with model files
            if export_path.exists():
                has_model = any(
                    (export_path / f).exists()
                    for f in [
                        "model.safetensors",
                        "model-00001-of-00001.safetensors",
                        "pytorch_model.bin",
                        "model.safetensors.index.json",
                    ]
                )
                return has_model
        except ValueError:
            pass
        return False

    def generate_merge_config(
        self, adapter_path: Path
    ) -> Tuple[str, MergeConfigGenerator]:
        """Generate and save merge config for an adapter."""
        config_gen = MergeConfigGenerator(
            str(adapter_path),
            output_base=str(self.merged_models_dir),
            hf_cache=self.hf_cache,
        )

        config_name = f"{config_gen.export_name}.yaml"
        config_path = self.config_output / config_name
        config_gen.save(str(config_path))

        return str(config_path), config_gen

    def generate_sbatch(
        self, config_path: str, config_gen: MergeConfigGenerator
    ) -> str:
        """Generate SLURM sbatch script for merge."""
        sbatch_name = f"{config_gen.export_name}.sbatch"
        sbatch_path = self.sbatch_output / sbatch_name

        sbatch_gen = MergeSbatchGenerator(
            config_path=config_path,
            job_name=config_gen.export_name,
            model_params_b=config_gen.model_config.get("params_b", 2.0),
            hf_cache=self.hf_cache,
        )
        sbatch_gen.save(str(sbatch_path))

        return str(sbatch_path)

    def merge_checkpoint(
        self,
        adapter_path: Path,
        generate_sbatch: bool = False,
        submit: bool = False,
        force: bool = False,
    ) -> Optional[str]:
        """Process a single checkpoint for merging."""
        if not force and self.is_already_merged(adapter_path):
            print(f"⏭ Already merged: {adapter_path.name}")
            return None

        config_path, config_gen = self.generate_merge_config(adapter_path)
        print(
            f"  Model: {config_gen.model_key} ({config_gen.model_config['params_b']}B)"
        )
        print(f"  Export: {config_gen.export_name}")

        if generate_sbatch or submit:
            sbatch_path = self.generate_sbatch(config_path, config_gen)
            print(f"  Config: {config_path}")
            print(f"  SLURM:  {sbatch_path}")

            if submit:
                result = subprocess.run(
                    ["sbatch", sbatch_path],
                    capture_output=True,
                    text=True,
                )
                print(f"  {result.stdout.strip()}")
                if result.returncode != 0:
                    print(f"  Error: {result.stderr}")
            return sbatch_path
        else:
            print(f"  Config: {config_path}")
            return config_path

    def merge_run_batch(
        self,
        run_path: Path,
        checkpoints: List[Path],
        submit: bool = False,
        force: bool = False,
    ) -> Optional[str]:
        """Process all checkpoints from a run as a single batch SLURM job."""
        if not checkpoints:
            print(f"⏭ No checkpoints found in {run_path.name}")
            return None

        # Filter out already merged checkpoints unless force
        if not force:
            checkpoints = [cp for cp in checkpoints if not self.is_already_merged(cp)]
            if not checkpoints:
                print(f"⏭ All checkpoints already merged in {run_path.name}")
                return None

        # Generate configs for all checkpoints
        config_paths = []
        model_params_b = 2.0
        run_name = run_path.name

        print(f"📦 Generating configs for {len(checkpoints)} checkpoints...")
        for cp in checkpoints:
            try:
                config_path, config_gen = self.generate_merge_config(cp)
                config_paths.append(config_path)
                model_params_b = config_gen.model_config.get("params_b", 2.0)
                print(f"   ✓ {cp.name} -> {config_gen.export_name}")
            except ValueError as e:
                print(f"   ✗ {cp.name}: {e}")

        if not config_paths:
            print("❌ No configs generated")
            return None

        # Generate batch sbatch script
        # Use run name (shortened) for job name
        job_name = run_name[:40] + "-batch"
        sbatch_name = f"{run_name}-batch.sbatch"
        sbatch_path = self.sbatch_output / sbatch_name

        batch_gen = MergeBatchSbatchGenerator(
            config_paths=config_paths,
            job_name=job_name,
            model_params_b=model_params_b,
            hf_cache=self.hf_cache,
        )
        batch_gen.save(str(sbatch_path))

        print(f"\n📋 Batch job: {len(config_paths)} checkpoints")
        print(f"   SLURM: {sbatch_path}")

        if submit:
            result = subprocess.run(
                ["sbatch", str(sbatch_path)],
                capture_output=True,
                text=True,
            )
            print(f"   {result.stdout.strip()}")
            if result.returncode != 0:
                print(f"   Error: {result.stderr}")

        return str(sbatch_path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with base models for vLLM inference"
    )

    # Input selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to specific checkpoint directory",
    )
    group.add_argument(
        "--run",
        type=str,
        help="Run name to process (e.g., internvl3_1b-tech-lora-all-1x48GB-b32)",
    )
    group.add_argument(
        "--scan",
        action="store_true",
        help="Scan and process all LoRA/QLoRA runs",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List all LoRA/QLoRA runs and their checkpoints",
    )

    # Checkpoint selection mode
    parser.add_argument(
        "--best",
        action="store_true",
        help="Select checkpoint with lowest eval_loss",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Select latest checkpoint by step number",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Process all checkpoints in run",
    )

    # Output options
    parser.add_argument(
        "--sbatch",
        action="store_true",
        help="Generate SLURM sbatch script",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit SLURM job",
    )
    parser.add_argument(
        "--batch-per-run",
        action="store_true",
        help="Create single SLURM job per run when using --all-checkpoints (default: True)",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Create separate SLURM job for each checkpoint (disables batching)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force merge even if already merged",
    )

    # Paths
    parser.add_argument(
        "--trained-models-dir",
        type=str,
        default="/storage/user/falu/trained_models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--merged-models-dir",
        type=str,
        default="/storage/user/falu/merged_models",
        help="Directory for merged model output",
    )

    args = parser.parse_args()

    orchestrator = MergeOrchestrator(
        trained_models_dir=args.trained_models_dir,
        merged_models_dir=args.merged_models_dir,
    )

    # List mode
    if args.list:
        runs = orchestrator.find_lora_runs()
        if not runs:
            print("No LoRA/QLoRA runs found.")
            return

        print(f"\nFound {len(runs)} LoRA/QLoRA runs:\n")
        for run in runs:
            checkpoints = orchestrator.find_checkpoints(run)
            best = orchestrator.find_best_checkpoint(run)
            latest = orchestrator.find_latest_checkpoint(run)

            print(f"📁 {run.name}")
            print(f"   Checkpoints: {len(checkpoints)}")
            if checkpoints:
                steps = [
                    int(re.match(r"checkpoint-(\d+)", c.name).group(1))
                    for c in checkpoints
                ]
                print(f"   Steps: {', '.join(map(str, sorted(steps)))}")
            if best:
                merged = "✓ merged" if orchestrator.is_already_merged(best) else ""
                print(f"   Best: {best.name} {merged}")
            if latest and latest != best:
                merged = "✓ merged" if orchestrator.is_already_merged(latest) else ""
                print(f"   Latest: {latest.name} {merged}")
            print()
        return

    # Determine if we should use batch processing
    use_batch = (
        args.all_checkpoints and (args.sbatch or args.submit) and not args.no_batch
    )

    # Determine checkpoints to process
    checkpoints_to_process = []
    runs_to_batch = []  # For batch processing: list of (run_path, checkpoints)

    if args.checkpoint:
        checkpoints_to_process = [Path(args.checkpoint)]

    elif args.run:
        run_path = orchestrator.trained_models_dir / args.run
        if not run_path.exists():
            print(f"❌ Run not found: {args.run}")
            sys.exit(1)

        if args.all_checkpoints:
            checkpoints = orchestrator.find_checkpoints(run_path)
            if use_batch:
                runs_to_batch = [(run_path, checkpoints)]
            else:
                checkpoints_to_process = checkpoints
        elif args.latest:
            cp = orchestrator.find_latest_checkpoint(run_path)
            if cp:
                checkpoints_to_process = [cp]
        else:  # default to best
            cp = orchestrator.find_best_checkpoint(run_path)
            if cp:
                checkpoints_to_process = [cp]

    elif args.scan:
        runs = orchestrator.find_lora_runs()
        for run in runs:
            if args.all_checkpoints:
                checkpoints = orchestrator.find_checkpoints(run)
                if use_batch:
                    if checkpoints:
                        runs_to_batch.append((run, checkpoints))
                else:
                    checkpoints_to_process.extend(checkpoints)
            elif args.latest:
                cp = orchestrator.find_latest_checkpoint(run)
                if cp:
                    checkpoints_to_process.append(cp)
            else:  # default to best
                cp = orchestrator.find_best_checkpoint(run)
                if cp:
                    checkpoints_to_process.append(cp)

    else:
        parser.print_help()
        print("\n❌ Please specify --checkpoint, --run, --scan, or --list")
        sys.exit(1)

    # Process using batch mode (one SLURM job per run)
    if use_batch and runs_to_batch:
        total_checkpoints = sum(len(cps) for _, cps in runs_to_batch)
        print(
            f"\n📦 Batch processing {total_checkpoints} checkpoints across {len(runs_to_batch)} run(s)...\n"
        )

        for run_path, checkpoints in runs_to_batch:
            print(f"\n🔄 {run_path.name} ({len(checkpoints)} checkpoints)")
            try:
                orchestrator.merge_run_batch(
                    run_path,
                    checkpoints,
                    submit=args.submit,
                    force=args.force,
                )
            except ValueError as e:
                print(f"   ❌ Error: {e}")

        print("\n✓ Done!")
        return

    # Regular processing (one SLURM job per checkpoint)
    if not checkpoints_to_process:
        print("❌ No checkpoints found to process")
        sys.exit(1)

    print(f"\nProcessing {len(checkpoints_to_process)} checkpoint(s)...\n")

    for cp in checkpoints_to_process:
        print(f"🔄 {cp.parent.name}/{cp.name}")
        try:
            orchestrator.merge_checkpoint(
                cp,
                generate_sbatch=args.sbatch or args.submit,
                submit=args.submit,
                force=args.force,
            )
        except ValueError as e:
            print(f"   ❌ Error: {e}")
        print()

    print("✓ Done!")


if __name__ == "__main__":
    main()
