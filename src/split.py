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
from sklearn.model_selection import train_test_split

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
    print(f"Total catalog samples: {total_samples}")

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    # Second split: Val vs Test
    val_share_of_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        stratify=temp_df["label"],
        random_state=seed,
    )

    # Reset indices
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

    print(f"Splits saved to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    create_stratified_splits()
