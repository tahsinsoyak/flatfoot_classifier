"""Robust medical preprocessing pipeline for weight-bearing lateral foot radiographs.

Handles:
1. 16-bit to 8-bit dynamic range windowing (percentile-based P1-P99).
2. Contrast-Limited Adaptive Histogram Equalization (CLAHE) for trabecular bone & joint delineation.
3. Anatomical tibial-axis foot orientation detection (canonical right-facing standardization).
4. Platform edge detection and tight foot ROI extraction (eliminates table legs, letter markers).
5. Aspect-ratio preserving letterbox formatting (Durum B) with >=30px safety margins to prevent bone clipping.
6. Uniform 512x512 matrix generation.
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
        mode: str = "letterbox",
        safety_margin_px: int = 30,
    ) -> None:
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid
        )
        self.canonical_direction = canonical_direction
        self.mode = mode
        self.safety_margin_px = safety_margin_px

    def window_16to8(self, img16: np.ndarray) -> np.ndarray:
        """Robust percentile windowing from 16-bit to 8-bit."""
        p1, p99 = np.percentile(img16, (1, 99))
        if p99 <= p1:
            p1, p99 = float(img16.min()), float(img16.max())
        clipped = np.clip(img16, p1, p99)
        norm8 = ((clipped - p1) / (p99 - p1 + 1e-6) * 255.0).astype(np.uint8)
        return norm8

    def detect_orientation(self, img8: np.ndarray) -> Tuple[str, int]:
        """Infer foot direction ('left' or 'right') and tibial shaft x-coordinate.

        In weight-bearing lateral foot radiographs:
        - Tibia and fibula enter vertically into the ankle mortise.
        - The upper third (25%-45% height) contains only the vertical shin bone.
        - If shin is in the right half -> ankle is on right -> toes point LEFT.
        - If shin is in the left half -> ankle is on left -> toes point RIGHT.
        """
        h, w = img8.shape
        shin_slice = img8[int(h * 0.25) : int(h * 0.45), :]
        shin_col = shin_slice.mean(axis=0)
        # Suppress peripheral 6% detector border noise
        shin_col[: int(w * 0.06)] = 0
        shin_col[int(w * 0.94) :] = 0
        shin_x = int(np.argmax(shin_col))
        direction = "left" if shin_x > (w / 2) else "right"
        return direction, shin_x

    def detect_foot_roi(
        self, img8: np.ndarray, direction: str, shin_x: int
    ) -> Tuple[int, int, int, int, int]:
        """Detect the bounding box of the foot resting on the platform with safety margin.

        Returns:
            (y_min, y_max, x_min, x_max, platform_y)
        """
        h, w = img8.shape

        # 1. Platform edge detection via horizontal Sobel filter
        sobel_y = cv2.Sobel(img8, cv2.CV_64F, 0, 1, ksize=5)
        central_strip = np.abs(sobel_y[:, int(w * 0.25) : int(w * 0.75)])
        row_energy = np.mean(central_strip, axis=1)

        search_y1, search_y2 = int(h * 0.40), int(h * 0.85)
        sub = row_energy[search_y1:search_y2]
        if len(sub) > 0:
            peaks = np.argsort(sub)[-10:] + search_y1
            stand_bottom = int(np.max(peaks))
            top_candidates = [
                p for p in peaks if stand_bottom - 280 <= p <= stand_bottom - 120
            ]
            if len(top_candidates) > 0:
                platform_top = int(np.max(top_candidates))
            else:
                platform_top = max(int(h * 0.45), stand_bottom - 215)
        else:
            platform_top = int(h * 0.65)

        # 2. Anatomical Foot Bounding Box
        # In lateral projections, adult foot length spans ~60-65% of the detector field.
        # Tibial shaft axis is ~25% from posterior calcaneus and ~75% from anterior distal phalanges.
        foot_len = int(w * 0.65)
        if direction == "right":  # Toes point right, calcaneus on left
            x_min = max(0, int(shin_x - foot_len * 0.26))
            x_max = min(w, int(shin_x + foot_len * 0.78))
        else:  # Toes point left, calcaneus on right
            x_min = max(0, int(shin_x - foot_len * 0.78))
            x_max = min(w, int(shin_x + foot_len * 0.26))

        actual_w = max(200, x_max - x_min)
        # Anatomical foot height (calcaneus base to talar dome) + safety margin
        foot_h = int(actual_w * 0.43)
        y_min = max(0, platform_top - foot_h)
        # Bottom margin: user-requested buffer below plantar sole to prevent cutting
        y_max = min(h, platform_top + self.safety_margin_px)

        return y_min, y_max, x_min, x_max, platform_top

    def process(
        self, img16_or_8: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Complete preprocessing pipeline on a single image.

        Args:
            img16_or_8: Grayscale input image (uint16 or uint8).

        Returns:
            processed_img: 8-bit image (512x512), CLAHE-enhanced, canonical right orientation.
            metadata: Info dictionary with bounding box, detected orientation, and platform position.
        """
        # 1. Convert to 8-bit if needed
        if img16_or_8.dtype == np.uint16 or img16_or_8.max() > 255:
            img8 = self.window_16to8(img16_or_8)
        else:
            img8 = img16_or_8.copy()

        h_orig, w_orig = img8.shape

        # 2. Tibial orientation detection
        detected_direction, shin_x = self.detect_orientation(img8)

        # 3. Contrast enhancement
        enhanced = self.clahe.apply(img8)

        # 4. Foot ROI detection with safety margins
        y_min, y_max, x_min, x_max, plat_y = self.detect_foot_roi(
            img8, detected_direction, shin_x
        )

        # 5. Crop
        cropped = enhanced[y_min:y_max, x_min:x_max]

        # 6. Canonicalize (Flip horizontally if needed to point right)
        flipped = False
        if detected_direction != self.canonical_direction:
            cropped = cv2.flip(cropped, 1)
            flipped = True

        # 7. Resizing with mode
        target_w, target_h = self.target_size
        if self.mode == "letterbox":
            # Preserve true anatomical aspect ratio, inner canvas padding
            inner_dim = int(target_w * 0.94)  # leaves ~15px canvas border
            ch, cw = cropped.shape
            scale = inner_dim / max(ch, cw)
            nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
            resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)

            processed = np.zeros(self.target_size, dtype=np.uint8)
            y_off = (target_h - nh) // 2
            x_off = (target_w - nw) // 2
            processed[y_off : y_off + nh, x_off : x_off + nw] = resized
        else:
            processed = cv2.resize(
                cropped, self.target_size, interpolation=cv2.INTER_AREA
            )

        meta = {
            "orig_height": h_orig,
            "orig_width": w_orig,
            "detected_direction": detected_direction,
            "canonical_direction": self.canonical_direction,
            "shin_x": shin_x,
            "flipped": flipped,
            "platform_y": plat_y,
            "bbox": (y_min, y_max, x_min, x_max),
            "mode": self.mode,
        }

        return processed, meta
