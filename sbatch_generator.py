"""SLURM sbatch script generator for distributed VLM training."""

from typing import Optional
from pathlib import Path
import textwrap


class SbatchGenerator:
    """Generate SLURM sbatch training scripts with dynamic resource allocation."""

    def __init__(
        self,
        config_path: str,
        num_gpus: int = 1,
        num_nodes: int = 1,
        job_name: Optional[str] = None,
        time_hours: Optional[int] = None,
        gpu_vram_gb: int = 48,
        cpus_per_gpu: int = 8,
        node_list: Optional[str] = None,
        output_dir: str = "/usr/stud/falu/code/LLaMA-Factory/logs",
        hf_cache: str = "/storage/user/falu/.cache/huggingface",
        model_params_b: float = 2.0,
        log_file_name: Optional[str] = None,
    ):
        """
        Initialize sbatch generator.

        Args:
            config_path: Path to YAML config file
            num_gpus: Number of GPUs to allocate
            num_nodes: Number of compute nodes
            job_name: SLURM job name (may be truncated for SLURM limits)
            time_hours: Maximum runtime (auto-calculated if None)
            gpu_vram_gb: GPU VRAM in GB (for resource request)
            cpus_per_gpu: CPU cores per GPU
            node_list: Specific nodes (e.g., "node[15-20]")
            output_dir: Directory for SLURM logs
            hf_cache: Hugging Face cache directory
            model_params_b: Model size in billions (for time estimation)
            log_file_name: Full run name for log file (defaults to job_name if None)
        """
        self.config_path = config_path
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.node_list = node_list or "node[15-20]"
        self.output_dir = output_dir
        self.hf_cache = hf_cache
        self.gpu_vram_gb = gpu_vram_gb
        self.model_params_b = model_params_b

        if job_name is None:
            config_stem = Path(config_path).stem
            job_name = f"train-{config_stem}"
        self.job_name = job_name
        self.log_file_name = log_file_name or job_name

        self.time_hours = time_hours or self._estimate_time()
        self.cpus_per_gpu = cpus_per_gpu
        self.total_cpus = cpus_per_gpu * num_gpus
        self.total_memory = (
            gpu_vram_gb * num_gpus
        )  # Request GPU VRAM as system memory too

    def _estimate_time(self) -> int:
        """Estimate job time based on model size."""
        # Rough heuristic: larger models need more time
        if self.model_params_b < 5:
            return 48  # 2 days
        elif self.model_params_b < 15:
            return 72  # 3 days
        elif self.model_params_b < 50:
            return 96  # 4 days
        else:
            return 120  # 5 days

    def generate(self) -> str:
        """Generate the sbatch script content."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # Use full run name in log file for better identification
        log_file = f"{self.output_dir}/{self.log_file_name}-%j.out"

        days = self.time_hours // 24
        hours = self.time_hours % 24
        time_str = f"{days}-{hours:02d}:00:00"

        script = textwrap.dedent(
            f"""\
            #!/bin/bash -l
            #SBATCH --nodes={self.num_nodes}
            #SBATCH --ntasks-per-node=1
            #SBATCH --nodelist={self.node_list}
            #SBATCH --gpus-per-node={self.num_gpus}
            #SBATCH --cpus-per-task={self.total_cpus}
            #SBATCH --mem={self.total_memory}G
            #SBATCH --time={time_str}
            #SBATCH --job-name={self.job_name}
            #SBATCH --output={log_file}
            #SBATCH --error={log_file}
            
            # ============================================================================
            # ENVIRONMENT SETUP
            # ============================================================================
            
            set -e
            
            echo "Job: {self.job_name}"
            echo "Job ID: ${{SLURM_JOB_ID}}"
            echo "GPUs: {self.num_gpus}"
            echo "Host: $(hostname)"
            echo "================================"
            
            source /home/stud/falu/code/LLaMA-Factory/.venv/bin/activate
            export HF_HOME={self.hf_cache}
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        """
        )

        if self.num_gpus > 1:
            script += textwrap.dedent(
                f"""\
                
                # Multi-GPU setup
                export NPROC_PER_NODE={self.num_gpus}
                export NNODES={self.num_nodes}
                export NODE_RANK=0
                export MASTER_ADDR=${{SLURM_NODELIST%%,*}}
                export MASTER_PORT=29501
                
                echo "GPUs/node: ${{NPROC_PER_NODE}}, Nodes: ${{NNODES}}"
            """
            )

        script += textwrap.dedent(
            f"""\
            
            # ============================================================================
            # TRAINING
            # ============================================================================
            
            cd /home/stud/falu/code/LLaMA-Factory
            echo "Config: {self.config_path}"
            echo "================================"
        """
        )

        if self.num_gpus > 1:
            script += textwrap.dedent(
                f"""\
                
                torchrun \\
                  --nproc_per_node ${{NPROC_PER_NODE}} \\
                  --nnodes ${{NNODES}} \\
                  --node_rank ${{NODE_RANK}} \\
                  --master_addr ${{MASTER_ADDR}} \\
                  --master_port ${{MASTER_PORT}} \\
                  src/train.py {self.config_path}
            """
            )
        else:
            script += f"\npython src/train.py {self.config_path}\n"

        script += textwrap.dedent(
            """\
            
            TRAIN_EXIT_CODE=$?
            
            if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
                echo "✓ Training completed"
            else
                echo "✗ Training failed: ${TRAIN_EXIT_CODE}"
                exit ${TRAIN_EXIT_CODE}
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

    def get_submit_command(self, sbatch_path: str) -> str:
        """Get the command to submit the job."""
        return f"sbatch {sbatch_path}"

    def print_script(self) -> None:
        """Print the sbatch script."""
        print("\n" + "=" * 70)
        print("SBATCH SCRIPT")
        print("=" * 70)
        print(self.generate())
        print("=" * 70 + "\n")
