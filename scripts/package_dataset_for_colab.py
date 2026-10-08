"""Packaging script to prepare processed dataset and split manifests into a single zip archive for Google Colab."""

import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def package_dataset(output_zip: Path | str = None) -> Path:
    if output_zip is None:
        output_zip = PROJECT_ROOT / "data" / "flatfoot_processed_dataset.zip"
    else:
        output_zip = Path(output_zip)

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    splits_dir = PROJECT_ROOT / "data" / "splits"

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found at {processed_dir}")

    files_to_pack = []
    # Collect processed images
    for p in processed_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".csv"]:
            rel_path = p.relative_to(PROJECT_ROOT)
            files_to_pack.append((p, rel_path))

    # Collect split files
    if splits_dir.exists():
        for p in splits_dir.rglob("*"):
            if p.is_file():
                rel_path = p.relative_to(PROJECT_ROOT)
                files_to_pack.append((p, rel_path))

    print(f"Packaging {len(files_to_pack)} files into {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in tqdm(files_to_pack, desc="Compressing"):
            zf.write(src, arcname=str(arc).replace("\\", "/"))

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully created {output_zip} ({size_mb:.2f} MB)")
    print("\nNext step for Google Colab:")
    print("1. Upload this zip file to your Google Drive (e.g. MyDrive/flatfoot_processed_dataset.zip).")
    print("2. In Google Colab, mount Google Drive and unzip into your workspace.")
    return output_zip


if __name__ == "__main__":
    package_dataset()
