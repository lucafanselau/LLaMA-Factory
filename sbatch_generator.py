"""
SLURM sbatch script generator for distributed training on GPU nodes.
Creates batch job submissions with proper resource allocation and environment setup.
"""

from typing import Optional
from pathlib import Path
import textwrap


class SbatchGenerator:
    """Generate SLURM sbatch training scripts."""

    def __init__(
        self,
        config_path: str,
        num_gpus: int = 1,
        num_nodes: int = 1,
        job_name: Optional[str] = None,
        time_hours: Optional[int] = None,
        mem_per_gpu: Optional[int] = None,
        cpus_per_gpu: Optional[int] = None,
        node_list: Optional[str] = None,
        output_dir: str = "/usr/stud/falu/code/LLaMA-Factory/logs",
        hf_cache: str = "/storage/user/falu/.cache/huggingface",
        model_tier: Optional[str] = None,
        preprocessing_num_workers: int = 8,
        batch_size: int = 2,
    ):
        """
        Initialize sbatch generator with smart resource allocation.

        Args:
            config_path: Path to YAML config file for training
            num_gpus: Number of GPUs to allocate
            num_nodes: Number of compute nodes (default 1)
            job_name: SLURM job name (auto-generated if not provided)
            time_hours: Maximum job runtime (auto-calculated if None)
            mem_per_gpu: GPU memory in GB (auto-calculated if None)
            cpus_per_gpu: CPU cores per GPU (auto-calculated if None)
            node_list: Specific node names (e.g., "node[15-20]")
            output_dir: Directory for SLURM log files
            hf_cache: Hugging Face cache directory
            model_tier: Model tier (tiny/small/medium/large/xlarge) for smart defaults
            preprocessing_num_workers: Number of data preprocessing workers (affects CPU allocation)
            batch_size: Batch size per device (affects memory allocation)
        """
        self.config_path = config_path
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.node_list = node_list or "node[15-20]"
        self.output_dir = output_dir
        self.hf_cache = hf_cache
        self.preprocessing_num_workers = preprocessing_num_workers
        self.batch_size = batch_size
        self.model_tier = model_tier or "small"

        # Auto-generate job name if not provided
        if job_name is None:
            config_stem = Path(config_path).stem
            job_name = f"train-{config_stem}"
        self.job_name = job_name

        # Calculate smart resource defaults if not provided
        self.time_hours = time_hours or self._calculate_time_hours()
        self.mem_per_gpu = mem_per_gpu or self._calculate_mem_per_gpu()
        self.cpus_per_gpu = cpus_per_gpu or self._calculate_cpus_per_gpu()

        # Calculate resources
        self.total_cpus = self.cpus_per_gpu * num_gpus
        self.total_memory = self.mem_per_gpu * num_gpus
        self.gpu_string = f"gpu:{num_gpus}"
        if self.mem_per_gpu > 0:
            self.gpu_string += f",VRAM:{self.mem_per_gpu}G"

    def _calculate_time_hours(self) -> int:
        """Calculate job time based on model tier and optimization level."""
        tier_times = {
            "tiny": 36,  # ~1.5 days (optimized, fast)
            "small": 48,  # 2 days (optimized, moderate)
            "medium": 60,  # ~2.5 days
            "large": 72,  # 3 days
            "xlarge": 96,  # 4 days
        }
        return tier_times.get(self.model_tier, 48)

    def _calculate_mem_per_gpu(self) -> int:
        """Calculate memory per GPU based on batch size and tier."""
        # Base calculation: smaller batch size = less memory needed
        # tiny models with batch_size=2: ~20-24GB
        # small models with batch_size=2: ~28-32GB
        mem_base = {
            "tiny": 32,  # 2B models with optimized config
            "small": 40,  # 8B models
            "medium": 40,  # 20B models with grad accumulation
            "large": 48,  # 30B sparse models
            "xlarge": 48,  # 235B sparse models
        }
        base_mem = mem_base.get(self.model_tier, 40)

        # Adjust based on batch size
        if self.batch_size <= 2:
            mem_adjustment = 0  # Already accounted for in base
        elif self.batch_size <= 4:
            mem_adjustment = 4
        else:
            mem_adjustment = 8

        return base_mem + mem_adjustment

    def _calculate_cpus_per_gpu(self) -> int:
        """Calculate CPU cores based on preprocessing workers."""
        # Heuristic: 1 CPU per 2-3 preprocessing workers, min 4, max 8
        # With 8 preprocessing workers: 4-6 CPUs
        # With 16 preprocessing workers: 6-8 CPUs
        cpus = max(4, min(8, self.preprocessing_num_workers // 2))
        return cpus

    def generate(self) -> str:
        """Generate the sbatch script content."""
        # Ensure output dir exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        log_file = f"{self.output_dir}/train-%j.out"

        time_str = f"{self.time_hours - self.time_hours // 24 * 24:02d}:{(self.time_hours % 24)}"

        script = textwrap.dedent(
            f"""\
            #!/bin/bash -l
            #SBATCH --nodes={self.num_nodes}
            #SBATCH --ntasks-per-node=1
            #SBATCH --nodelist={self.node_list}
            #SBATCH --gpus-per-node={self.num_gpus}
            #SBATCH --cpus-per-task={self.total_cpus}
            #SBATCH --mem={self.total_memory}G
            #SBATCH --time={self.time_hours // 24}-{time_str}:00
            #SBATCH --job-name={self.job_name}
            #SBATCH --output={log_file}
            #SBATCH --error={log_file}
            
            # ============================================================================
            # ENVIRONMENT SETUP
            # ============================================================================
            
            set -e  # Exit on error
            
            echo "Starting training job: {self.job_name}"
            echo "Job ID: ${{SLURM_JOB_ID}}"
            echo "Allocated GPUs: ${{SLURM_GPUS}}"
            echo "Host: $(hostname)"
            echo "================================"
            
            # Activate virtual environment
            source /home/stud/falu/code/LLaMA-Factory/.venv/bin/activate
            
            # Set Hugging Face cache
            export HF_HOME={self.hf_cache}
            
            # ============================================================================
            # DISTRIBUTED TRAINING SETUP
            # ============================================================================
            """
        )

        # Add multi-GPU setup if needed
        if self.num_gpus > 1:
            script += textwrap.dedent(
                f"""\
                
                # Multi-GPU setup with torchrun
                export NPROC_PER_NODE={self.num_gpus}
                export NNODES={self.num_nodes}
                export NODE_RANK=0
                export MASTER_ADDR=${{SLURM_NODELIST%%,*}}
                export MASTER_PORT=29500
                
                echo "Multi-GPU Training Configuration:"
                echo "  GPUs per node: ${{NPROC_PER_NODE}}"
                echo "  Total nodes: ${{NNODES}}"
                echo "  Master address: ${{MASTER_ADDR}}"
                echo "  Master port: ${{MASTER_PORT}}"
                """
            )

        script += textwrap.dedent(
            f"""\
            
            # ============================================================================
            # TRAINING EXECUTION
            # ============================================================================
            
            cd /home/stud/falu/code/LLaMA-Factory
            
            echo "Running training with config: {self.config_path}"
            echo "================================"
            """
        )

        # Build training command
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
            script += textwrap.dedent(
                f"""\
                
                python src/train.py {self.config_path}
                """
            )

        script += textwrap.dedent(
            f"""\
            
            TRAIN_EXIT_CODE=$?
            
            # ============================================================================
            # JOB COMPLETION
            # ============================================================================
            
            if [ ${{TRAIN_EXIT_CODE}} -eq 0 ]; then
                echo "✓ Training completed successfully"
            else
                echo "✗ Training failed with exit code ${{TRAIN_EXIT_CODE}}"
                exit ${{TRAIN_EXIT_CODE}}
            fi
            
            echo "Job finished at: $(date)"
            """
        )

        return script

    def save(self, path: str) -> None:
        """Save sbatch script to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.generate())
        # Make executable
        Path(path).chmod(0o755)
        print(f"✓ SLURM script saved to: {path}")

    def get_submit_command(self, sbatch_path: str) -> str:
        """Get the command to submit the sbatch job."""
        return f"sbatch {sbatch_path}"

    def print_script(self) -> None:
        """Print the sbatch script to stdout."""
        print("\n" + "=" * 70)
        print("SBATCH SCRIPT")
        print("=" * 70)
        print(self.generate())
        print("=" * 70 + "\n")


def create_training_sbatch(
    config_path: str,
    sbatch_path: str,
    num_gpus: int = 1,
    job_name: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Convenience function to create and save sbatch script.

    Returns:
        Path to created sbatch script
    """
    generator = SbatchGenerator(
        config_path, num_gpus=num_gpus, job_name=job_name, **kwargs
    )
    generator.save(sbatch_path)
    return sbatch_path
