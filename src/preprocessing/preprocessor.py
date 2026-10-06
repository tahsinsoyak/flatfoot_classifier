"""Robust medical preprocessing pipeline for weight-bearing lateral foot radiographs.

Handles:
1. 16-bit to 8-bit dynamic range windowing (percentile-based).
2. CLAHE contrast enhancement for bone and soft-tissue delineation.
3. Anatomical foot orientation detection (Shin / Tibia-Fibula position).
4. Weight-bearing platform detection and foot ROI cropping (removes table, legs, L/R markers).
5. Canonical alignment (standardizes all feet to point right).
6. Square resizing (e.g. 512x512).
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class FootRadiographPreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        canonical_direction: str = "right",
    ) -> None:
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid
        )
        self.canonical_direction = canonical_direction

    def window_16to8(self, img16: np.ndarray) -> np.ndarray:
        """Robust percentile windowing from 16-bit to 8-bit."""
        p1, p99 = np.percentile(img16, (1, 99))
        if p99 <= p1:
            p1, p99 = float(img16.min()), float(img16.max())
        clipped = np.clip(img16, p1, p99)
        norm8 = ((clipped - p1) / (p99 - p1 + 1e-6) * 255.0).astype(np.uint8)
        return norm8

    def detect_orientation(self, img8: np.ndarray) -> str:
        """Infer foot direction ('left' or 'right') via upper leg / shin bone horizontal position.

        In weight-bearing lateral foot radiographs:
        - Tibia and fibula enter vertically into the ankle/calcaneus.
        - If shin is in the right half of the image -> Ankle is on right -> Toes point LEFT.
        - If shin is in the left half of the image -> Ankle is on left -> Toes point RIGHT.
        """
        h, w = img8.shape
        shin_slice = img8[int(h * 0.35) : int(h * 0.50), :]
        col_density = shin_slice.mean(axis=0)
        shin_x = int(np.argmax(col_density))
        return "left" if shin_x > (w / 2) else "right"

    def detect_foot_roi(self, img8: np.ndarray) -> Tuple[int, int, int, int, int]:
        """Detect the bounding box of the foot resting on the platform.

        Returns:
            (y_min, y_max, x_min, x_max, platform_y)
        """
        h, w = img8.shape

        # 1. Platform edge detection via horizontal Sobel filter
        sobel_y = cv2.Sobel(img8, cv2.CV_64F, 0, 1, ksize=5)
        search_y1, search_y2 = int(h * 0.45), int(h * 0.85)
        central_strip = np.abs(sobel_y[:, int(w * 0.2) : int(w * 0.8)])
        row_energy = np.mean(central_strip, axis=1)

        search_slice = row_energy[search_y1:search_y2]
        if len(search_slice) > 0:
            top_candidates = np.argsort(search_slice)[-5:] + search_y1
            platform_y = int(np.min(top_candidates))
        else:
            platform_y = int(h * 0.70)

        # 2. Foot bone segmentation in region above platform
        roi_above_plat = img8[: platform_y + int(h * 0.02), :]
        blur = cv2.GaussianBlur(roi_above_plat, (9, 9), 0)
        non_zero_pixels = blur[blur > 20]
        thresh_val = np.percentile(non_zero_pixels, 45) if len(non_zero_pixels) > 0 else 100
        _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

        # Only examine lower 60% of above-platform area to ignore upper shin width
        foot_zone_start = int(platform_y * 0.4)
        mask_foot_zone = mask[foot_zone_start:, :]

        col_counts = np.sum(mask_foot_zone > 0, axis=0)
        active_cols = np.where(col_counts > (mask_foot_zone.shape[0] * 0.1))[0]

        if len(active_cols) > 0:
            x_min = max(0, int(active_cols[0] - int(w * 0.04)))
            x_max = min(w, int(active_cols[-1] + int(w * 0.04)))
        else:
            x_min, x_max = int(w * 0.1), int(w * 0.9)

        # 3. Vertical bounds
        # y_max: just below the platform contact surface (includes ground line)
        y_max = min(h, platform_y + int(h * 0.02))
        foot_w = max(10, x_max - x_min)
        # Lateral foot typical aspect ratio is around 1.8:1 (w:h); height is approx 0.55*w
        desired_h = int(foot_w * 0.55)
        y_min = max(0, y_max - desired_h)

        return y_min, y_max, x_min, x_max, platform_y

    def process(
        self, img16_or_8: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Complete preprocessing pipeline on a single image.

        Args:
            img16_or_8: Grayscale input image (uint16 or uint8).

        Returns:
            processed_img: 8-bit image resized to target_size, CLAHE-enhanced, canonical right orientation.
            metadata: Info dictionary with bounding box, detected orientation, and platform position.
        """
        # Convert to 8-bit if needed
        if img16_or_8.dtype == np.uint16 or img16_or_8.max() > 255:
            img8 = self.window_16to8(img16_or_8)
        else:
            img8 = img16_or_8.copy()

        h_orig, w_orig = img8.shape

        # 1. Orientation detection
        detected_direction = self.detect_orientation(img8)

        # 2. Contrast enhancement
        enhanced = self.clahe.apply(img8)

        # 3. Foot ROI detection
        y_min, y_max, x_min, x_max, plat_y = self.detect_foot_roi(img8)

        # 4. Crop
        cropped = enhanced[y_min:y_max, x_min:x_max]

        # 5. Canonicalize (Flip horizontally if needed to point right)
        flipped = False
        if detected_direction != self.canonical_direction:
            cropped = cv2.flip(cropped, 1)
            flipped = True

        # 6. Resize to target size
        processed = cv2.resize(
            cropped, self.target_size, interpolation=cv2.INTER_AREA
        )

        meta = {
            "orig_height": h_orig,
            "orig_width": w_orig,
            "detected_direction": detected_direction,
            "canonical_direction": self.canonical_direction,
            "flipped": flipped,
            "platform_y": plat_y,
            "bbox": (y_min, y_max, x_min, x_max),
        }

        return processed, meta
