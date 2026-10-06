"""Grad-CAM (Gradient-weighted Class Activation Mapping) for Medical Interpretability."""

from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM heatmaps for any CNN backbone."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for a single input tensor of shape (1, C, H, W)."""
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        # Backward pass for the target class score
        score = output[0, target_class]
        score.backward()

        # Global average pooling on gradients
        # activations: (1, Channels, H, W), gradients: (1, Channels, H, W)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU on CAM (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    @staticmethod
    def overlay_heatmap(
        original_image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Resize heatmap to image size, colorize, and blend with original image."""
        h, w = original_image_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        blended = np.float32(heatmap_color_rgb) * alpha + np.float32(original_image_rgb) * (
            1.0 - alpha
        )
        return np.uint8(np.clip(blended, 0, 255))
