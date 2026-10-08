"""Stratified splitting script for train/validation/test partitions.

Creates:
- data/splits/train.csv
- data/splits/val.csv
- data/splits/test.csv
- data/splits/split_summary.json
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_stratified_splits(
    catalog_path: Path | None = None,
    output_dir: Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    if catalog_path is None:
        catalog_path = PROJECT_ROOT / "data" / "processed" / "dataset_catalog.csv"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "splits"

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(catalog_path)
    total_samples = len(df)
    print(f"Total catalog samples: {total_samples}", flush=True)

    pes_df = df[df["label"] == 1]
    norm_df = df[df["label"] == 0]

    # Train sample
    train_pes = pes_df.sample(frac=train_ratio, random_state=seed)
    train_norm = norm_df.sample(frac=train_ratio, random_state=seed)
    train_df = pd.concat([train_pes, train_norm])

    # Remaining
    rem_pes = pes_df.drop(train_pes.index)
    rem_norm = norm_df.drop(train_norm.index)

    # Val vs Test
    val_frac = val_ratio / (val_ratio + test_ratio)
    val_pes = rem_pes.sample(frac=val_frac, random_state=seed)
    val_norm = rem_norm.sample(frac=val_frac, random_state=seed)
    val_df = pd.concat([val_pes, val_norm])

    test_pes = rem_pes.drop(val_pes.index)
    test_norm = rem_norm.drop(val_norm.index)
    test_df = pd.concat([test_pes, test_norm])

    # Reset & sort for determinism
    train_df = train_df.sort_values("image_id").reset_index(drop=True)
    val_df = val_df.sort_values("image_id").reset_index(drop=True)
    test_df = test_df.sort_values("image_id").reset_index(drop=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "total": total_samples,
        "seed": seed,
        "train": {
            "count": len(train_df),
            "pes_planus": int((train_df["label"] == 1).sum()),
            "normal": int((train_df["label"] == 0).sum()),
            "percentage": round(len(train_df) / total_samples * 100, 2),
        },
        "val": {
            "count": len(val_df),
            "pes_planus": int((val_df["label"] == 1).sum()),
            "normal": int((val_df["label"] == 0).sum()),
            "percentage": round(len(val_df) / total_samples * 100, 2),
        },
        "test": {
            "count": len(test_df),
            "pes_planus": int((test_df["label"] == 1).sum()),
            "normal": int((test_df["label"] == 0).sum()),
            "percentage": round(len(test_df) / total_samples * 100, 2),
        },
    }

    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Splits saved to {output_dir}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    create_stratified_splits()
