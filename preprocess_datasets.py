#!/usr/bin/env python3
"""Generate preprocessing configs and sbatch scripts for all dataset/family combinations."""

from pathlib import Path
import textwrap

FAMILIES = {
    "qwen3": {
        "model": "Qwen/Qwen3-VL-2B-Instruct",  # smallest model for tokenizer
        "template": "qwen3_vl_nothink",
    },
    "internvl3": {
        "model": "OpenGVLab/InternVL3_5-1B-HF",  # smallest model for tokenizer
        "template": "intern_vl",
    },
}

DATASET_TYPES = ["single_image", "multi_image_single_turn", "multi_image_multi_turn"]

CONFIG_DIR = Path("preprocessing_configs")
SBATCH_DIR = Path("preprocessing_sbatch")
DATASET_DIR = "/storage/user/falu/vis/processed_llamafactory"
HF_CACHE = "/storage/user/falu/.cache/huggingface"

PREPROCESSING_NUM_WORKERS = 32


def get_dataset_keys(
    dataset_name: str, family: str, dataset_types: list[str]
) -> tuple[list[str], list[str]]:
    """Get train and val dataset keys."""
    if dataset_name == "all":
        from train_config import DATASET_VALIDATION_MAP

        train, val = [], []
        for k, v in DATASET_VALIDATION_MAP.items():
            if f"_{family}_" in k and any(dt in k for dt in dataset_types):
                train.append(k)
                val.append(v)
        return train, val

    train = [f"{dataset_name}_train_{family}_{dt}" for dt in dataset_types]
    val = [f"{dataset_name}_val_{family}_{dt}" for dt in dataset_types]
    return train, val


def generate_config(family: str, dataset_name: str, dataset_types: list[str]) -> str:
    """Generate preprocessing YAML config."""
    fam_config = FAMILIES[family]
    train_ds, val_ds = get_dataset_keys(dataset_name, family, dataset_types)
    # Match path format from config_generator.py
    dataset_str = (
        "all" if dataset_name == "all" else f"{dataset_name}_{'_'.join(dataset_types)}"
    )
    tokenized_path = f"{DATASET_DIR}/tokenized/{family}/{dataset_str}"

    return textwrap.dedent(
        f"""\
        # Preprocessing config for {family} - {dataset_name}
        model_name_or_path: {fam_config['model']}
        template: {fam_config['template']}
        trust_remote_code: true
        
        stage: sft
        do_train: false

        image_max_pixels: 589824
        image_min_pixels: 1024
        video_max_pixels: 65536
        video_min_pixels: 256
        
        dataset: {','.join(train_ds)}
        eval_dataset: {','.join(val_ds)}

        dataset_dir: {DATASET_DIR}
        media_dir: {Path(DATASET_DIR).parent}
        cutoff_len: 8192
        overwrite_cache: true
        use_fast_tokenizer: true
        preprocessing_num_workers: {PREPROCESSING_NUM_WORKERS}
        
        tokenized_path: {tokenized_path}
        
        output_dir: /tmp/preprocess_{family}_{dataset_name}
    """
    )


def generate_sbatch(family: str, dataset_name: str, config_path: str) -> str:
    """Generate CPU-only sbatch script for preprocessing."""
    job_name = f"preproc-{family}-{dataset_name}"
    log_file = f"/usr/stud/falu/code/LLaMA-Factory/logs/preproc-%j.out"

    return textwrap.dedent(
        f"""\
        #!/bin/bash -l
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task={PREPROCESSING_NUM_WORKERS}
        #SBATCH --mem=96G
        #SBATCH --time=0-12:00:00
        #SBATCH --job-name={job_name}
        #SBATCH --output={log_file}
        #SBATCH --error={log_file}
        
        set -e
        
        echo "Preprocessing job: {job_name}"
        echo "Job ID: ${{SLURM_JOB_ID}}"
        echo "Host: $(hostname)"
        echo "CPUs: ${{SLURM_CPUS_PER_TASK}}"
        echo "================================"
        
        source /home/stud/falu/code/LLaMA-Factory/.venv/bin/activate
        export HF_HOME={HF_CACHE}
        
        cd /home/stud/falu/code/LLaMA-Factory
        
        echo "Running preprocessing with config: {config_path}"
        python src/train.py {config_path}
        
        echo "Preprocessing completed at: $(date)"
    """
    )


def main():
    CONFIG_DIR.mkdir(exist_ok=True)
    SBATCH_DIR.mkdir(exist_ok=True)

    all_sbatch = []

    # Just 2 jobs: one per family with all datasets + all types
    for family in FAMILIES:
        config_name = f"preproc_{family}_all.yaml"
        sbatch_name = f"preproc_{family}_all.sbatch"

        config_path = CONFIG_DIR / config_name
        sbatch_path = SBATCH_DIR / sbatch_name

        config_path.write_text(generate_config(family, "all", DATASET_TYPES))
        print(f"✓ {config_path}")

        sbatch_path.write_text(generate_sbatch(family, "all", str(config_path)))
        sbatch_path.chmod(0o755)
        print(f"✓ {sbatch_path}")

        all_sbatch.append(str(sbatch_path))

    # Generate submit-all script
    submit_script = SBATCH_DIR / "submit_all.sh"
    submit_script.write_text(
        "#!/bin/bash\n# Submit all preprocessing jobs\n"
        + "\n".join(f"sbatch {s}" for s in all_sbatch)
        + "\n"
    )
    submit_script.chmod(0o755)

    print(f"\n✓ Generated {len(all_sbatch)} preprocessing jobs")
    print(f"Run all with: {submit_script}")


if __name__ == "__main__":
    main()
