"""PyTorch Dataset definition and transforms for Foot Radiograph Classification."""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
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
            img_path = row["processed_path"]
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
