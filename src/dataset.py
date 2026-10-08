"""PyTorch Dataset definition and transforms for Foot Radiograph Classification."""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class FootRadiographDataset(Dataset):
    """PyTorch Dataset loading preprocessed foot radiographs."""

    def __init__(
        self,
        csv_path: Path | str,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
        cache_in_memory: bool = True,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.return_meta = return_meta
        self.cache_in_memory = cache_in_memory
        self.cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row["label"])

        if self.cache_in_memory and idx in self.cache:
            image = self.cache[idx]
        else:
            raw_path_str = str(row["processed_path"])
            img_path = Path(raw_path_str)
            if not img_path.exists():
                # Resolve cross-platform relative path (e.g. when run on Linux/Colab)
                norm_str = raw_path_str.replace("\\", "/")
                parts = Path(norm_str).parts
                if "data" in parts:
                    data_idx = parts.index("data")
                    rel_candidate = Path(*parts[data_idx:])
                    if rel_candidate.exists():
                        img_path = rel_candidate
                    else:
                        cand = Path("data/processed") / parts[-2] / parts[-1]
                        if cand.exists():
                            img_path = cand
                else:
                    cand = Path("data/processed") / parts[-2] / parts[-1]
                    if cand.exists():
                        img_path = cand

            image = Image.open(img_path).convert("RGB")
            if self.cache_in_memory:
                self.cache[idx] = image

        if self.transform is not None:
            image = self.transform(image)

        if self.return_meta:
            return image, label, {
                "image_id": row["image_id"],
                "class_name": row["class_name"],
                "path": str(row["processed_path"]),
            }

        return image, label


def get_transforms(
    image_size: int = 512,
    is_training: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Standard image transforms for training and validation/testing.

    Note: Horizontal flipping is omitted to maintain canonical right-pointing foot orientation.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.96, 1.04),
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def create_dataloaders(
    data_dir: Path | str = "data",
    img_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    """Build train, val, and test DataLoader instances."""
    data_dir = Path(data_dir)
    train_csv = data_dir / "splits" / "train.csv"
    val_csv = data_dir / "splits" / "val.csv"
    test_csv = data_dir / "splits" / "test.csv"

    train_tf = get_transforms(image_size=img_size, is_training=True)
    eval_tf = get_transforms(image_size=img_size, is_training=False)

    train_ds = FootRadiographDataset(train_csv, transform=train_tf)
    val_ds = FootRadiographDataset(val_csv, transform=eval_tf)
    test_ds = FootRadiographDataset(test_csv, transform=eval_tf, return_meta=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

