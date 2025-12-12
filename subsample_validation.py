#!/usr/bin/env python3
"""Replace validation splits in tokenized caches with small sampled versions."""

from pathlib import Path
from datasets import load_from_disk, DatasetDict
import numpy as np

TOKENIZED_BASE = Path("/storage/user/falu/vis/processed/tokenized")
FAMILIES = ["qwen3", "internvl3"]
TOTAL_EVAL_SAMPLES = 768 * 8  # ~128 per dataset × 12 datasets
SEED = 42


def replace_validation_with_sample(cache_path: Path, num_samples: int) -> None:
    """Replace validation split with random sample, preserving train split."""
    print(f"\n{'='*70}")
    print(f"Processing: {cache_path}")
    print("=" * 70)

    # Load existing cache
    print("Loading dataset...")
    dataset_dict = load_from_disk(str(cache_path))

    if "validation" not in dataset_dict:
        print("⊘ No validation split found")
        return

    val_full = dataset_dict["validation"]
    train_full = dataset_dict.get("train")

    total_val = len(val_full)
    total_train = len(train_full) if train_full else 0

    print(f"Current sizes:")
    print(f"  Train:      {total_train:,} samples")
    print(f"  Validation: {total_val:,} samples")

    if total_val <= num_samples:
        print(f"  → Already small enough, skipping")
        return

    # Randomly sample validation
    print(f"\nSampling validation to {num_samples} samples...")
    np.random.seed(SEED)
    indices = np.random.choice(total_val, num_samples, replace=False)
    indices = sorted(indices.tolist())  # Sort for better compression

    val_sampled = val_full.select(indices)

    # Create new dataset dict
    new_dataset_dict = DatasetDict(
        {
            "train": train_full,  # Unchanged
            "validation": val_sampled,  # Sampled
        }
    )

    new_path = cache_path.parent / f"{cache_path.name}_val_sampled"

    # Save over original
    print(f"Saving updated cache to {new_path}...")
    new_dataset_dict.save_to_disk(str(new_path))


def main():
    print("Fast Eval Cache Creator")
    print("=" * 70)
    print(f"Target validation size: {TOTAL_EVAL_SAMPLES} samples")
    print(f"Families: {', '.join(FAMILIES)}")

    for family in FAMILIES:
        cache_path = TOKENIZED_BASE / family / "all"

        if not cache_path.exists():
            print(f"\n⊘ Cache not found: {cache_path}")
            continue

        try:
            replace_validation_with_sample(cache_path, TOTAL_EVAL_SAMPLES)
        except Exception as e:
            print(f"\n✗ Error processing {family}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✓ All caches processed!")
    print("\nNext steps:")
    print("  1. Your tokenized caches now have small validation splits")
    print("  2. Training will use full train data + fast 768-sample eval")
    print("  3. Eval time: ~1 hour → ~3 minutes per checkpoint")
    print("=" * 70)


if __name__ == "__main__":
    main()
