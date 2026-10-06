"""Batch preprocessing script to process raw radiographs into standardized canonical dataset.

Saves:
- data/processed/pes_planus/*.jpg
- data/processed/normal/*.jpg
- data/processed/dataset_catalog.csv
"""

from __future__ import annotations
import os
import sys
import glob
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import cv2
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.preprocessor import FootRadiographPreprocessor


def process_single_image(args):
    raw_path, out_path, class_name, label = args
    try:
        img16 = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        if img16 is None:
            return None, f"Failed to read {raw_path}"

        preprocessor = FootRadiographPreprocessor(target_size=(512, 512))
        processed_img, meta = preprocessor.process(img16)

        # Save processed image as high-quality JPEG
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        y_min, y_max, x_min, x_max = meta["bbox"]
        record = {
            "image_id": Path(raw_path).stem,
            "class_name": class_name,
            "label": label,
            "raw_path": str(raw_path),
            "processed_path": str(out_path),
            "orig_height": meta["orig_height"],
            "orig_width": meta["orig_width"],
            "detected_direction": meta["detected_direction"],
            "flipped": meta["flipped"],
            "platform_y": meta["platform_y"],
            "bbox_ymin": y_min,
            "bbox_ymax": y_max,
            "bbox_xmin": x_min,
            "bbox_xmax": x_max,
            "status": "success",
        }
        return record, None
    except Exception as e:
        return None, f"Error on {raw_path}: {str(e)}"


def main():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    pes_files = sorted(glob.glob(str(raw_dir / "duz_taban_PNG" / "*.png")))
    norm_files = sorted(glob.glob(str(raw_dir / "Normal_PNG_Standardized" / "*.png")))

    print(f"Found {len(pes_files)} Pes Planus and {len(norm_files)} Normal images.")

    tasks = []
    for p in pes_files:
        stem = Path(p).stem
        out_p = processed_dir / "pes_planus" / f"{stem}.jpg"
        tasks.append((p, str(out_p), "pes_planus", 1))

    for p in norm_files:
        stem = Path(p).stem
        out_p = processed_dir / "normal" / f"{stem}.jpg"
        tasks.append((p, str(out_p), "normal", 0))

    records = []
    errors = []

    print(f"Starting batch preprocessing of {len(tasks)} images...")
    start_time = time.time()

    # Use CPU cores for parallel processing
    num_workers = min(os.cpu_count() or 4, 8)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Preprocessing"):
            rec, err = future.result()
            if rec:
                records.append(rec)
            if err:
                errors.append(err)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f} seconds! Processed: {len(records)}, Errors: {len(errors)}")

    if errors:
        print(f"First few errors: {errors[:5]}")

    df = pd.DataFrame(records)
    # Sort for deterministic ordering
    df = df.sort_values(by=["class_name", "image_id"]).reset_index(drop=True)

    catalog_path = processed_dir / "dataset_catalog.csv"
    df.to_csv(catalog_path, index=False)
    print(f"Saved dataset catalog to {catalog_path}")
    print("\nSummary by class:")
    print(df["class_name"].value_counts())
    print("\nDetected foot directions:")
    print(df["detected_direction"].value_counts())


if __name__ == "__main__":
    main()
