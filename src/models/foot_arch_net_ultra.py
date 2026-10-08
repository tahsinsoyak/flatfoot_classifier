"""FootArchNet-Ultra: Dual-Stream Cross-Attention Radiographic Architecture.

Key Architectural Principles:
1. Macro-Stream: Processes full-foot 512x512 lateral projection with CoordConv2d
   to anchor global calcaneal pitch and talar-first metatarsal angles.
2. Micro-Stream: Differentiable high-resolution zoom into the midfoot arch vault
   (navicular-cuneiform-talonavicular joint complex) at native focal resolution.
3. Cross-Attention Biomechanical Transformer (X-Arch Attention):
   Bidirectional Multi-Head Cross-Attention where global macro tokens query local
   micro bone architecture and vice versa.
4. Anisotropic Biomechanical Strip Pooling V2: Captures longitudinal arch span
   and vertical collapse dynamics.
5. Dual Global Pooling (GAP + GMP) with LayerNorm-GELU classification MLP.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CoordConv2d(nn.Module):
    """Injects normalized [-1, 1] 2D coordinate grid channels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        x_with_coords = torch.cat([x, coords], dim=1)
        return self.act(self.norm(self.conv(x_with_coords)))


class BiomechanicalStripPoolingV2(nn.Module):
    """Anisotropic strip pooling module capturing horizontal span and vertical collapse."""

    def __init__(self, in_channels: int, pool_channels: int = 64) -> None:
        super().__init__()
        self.conv_h = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(in_channels, pool_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_channels),
            nn.SiLU(inplace=True),
        )
        self.conv_w = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(in_channels, pool_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_channels),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(pool_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pool_h = F.interpolate(self.conv_h(x), size=(h, w), mode="bilinear", align_corners=False)
        pool_w = F.interpolate(self.conv_w(x), size=(h, w), mode="bilinear", align_corners=False)
        strip_features = torch.cat([pool_h, pool_w], dim=1)
        attention_mask = self.fusion(strip_features)
        return x * attention_mask


class CrossArchAttention(nn.Module):
    """Bidirectional Cross-Attention Transformer between Macro (global) and Micro (focal) tokens."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_macro_to_micro = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_micro_to_macro = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_macro = nn.LayerNorm(d_model)
        self.norm_micro = nn.LayerNorm(d_model)

        self.ffn_macro = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_micro = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm_macro_ffn = nn.LayerNorm(d_model)
        self.norm_micro_ffn = nn.LayerNorm(d_model)

    def forward(self, macro_tokens: torch.Tensor, micro_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # macro_tokens: [B, N_macro, D]
        # micro_tokens: [B, N_micro, D]

        # Macro queries Micro
        m2m, _ = self.cross_macro_to_micro(query=macro_tokens, key=micro_tokens, value=micro_tokens)
        macro_out = self.norm_macro(macro_tokens + m2m)
        macro_out = self.norm_macro_ffn(macro_out + self.ffn_macro(macro_out))

        # Micro queries Macro
        u2m, _ = self.cross_micro_to_macro(query=micro_tokens, key=macro_tokens, value=macro_tokens)
        micro_out = self.norm_micro(micro_tokens + u2m)
        micro_out = self.norm_micro_ffn(micro_out + self.ffn_micro(micro_out))

        return macro_out, micro_out


class FootArchNetUltra(nn.Module):
    """FootArchNet-Ultra: Dual-Stream Cross-Attention Radiographic Deep Learning Model."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.25,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Stream 1: Macro Stream (ConvNeXt-Tiny for high-level full-foot shape & angles)
        weights_macro = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        macro_net = models.convnext_tiny(weights=weights_macro)
        self.macro_backbone = macro_net.features  # output: [B, 768, H/32, W/32]
        self.macro_coord = CoordConv2d(768, embed_dim, kernel_size=1)
        self.macro_strip = BiomechanicalStripPoolingV2(embed_dim, pool_channels=64)

        # Stream 2: Micro Zoom Stream (EfficientNet-B0 for high-resolution midfoot vault)
        weights_micro = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        micro_net = models.efficientnet_b0(weights=weights_micro)
        self.micro_backbone = micro_net.features  # output: [B, 1280, H_crop/32, W_crop/32]
        self.micro_proj = nn.Sequential(
            nn.Conv2d(1280, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.micro_strip = BiomechanicalStripPoolingV2(embed_dim, pool_channels=64)

        # Cross-Attention Transformer Fusion
        self.cross_attention = CrossArchAttention(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
        )

        # Dual Global Pooling for both streams: (GAP + GMP) -> 2 * embed_dim per stream
        # Total pooled representation: 2 * (2 * embed_dim) = 4 * 256 = 1024
        fusion_dim = embed_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(384, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        # Refine conv for Grad-CAM explainability
        self.gradcam_refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

    def extract_arch_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts midfoot arch vault region: y in [35%, 88%], x in [20%, 80%]."""
        _, _, h, w = x.shape
        y1, y2 = int(h * 0.35), int(h * 0.88)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        crop = x[:, :, y1:y2, x1:x2]
        # Rescale crop to standard 256x256 for micro feature extraction
        return F.interpolate(crop, size=(256, 256), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape

        # 1. Macro Stream (Full foot)
        macro_feat = self.macro_backbone(x)  # [B, 768, 16, 16]
        macro_feat = self.macro_coord(macro_feat)  # [B, embed_dim, 16, 16]
        macro_feat = self.macro_strip(macro_feat)
        macro_feat = self.gradcam_refine(macro_feat)

        # 2. Micro Stream (Arch vault zoom)
        arch_crop = self.extract_arch_crop(x)
        micro_feat = self.micro_backbone(arch_crop)  # [B, 1280, 8, 8]
        micro_feat = self.micro_proj(micro_feat)  # [B, embed_dim, 8, 8]
        micro_feat = self.micro_strip(micro_feat)

        # 3. Tokenize for Cross-Attention
        # Macro: [B, 256, 16, 16] -> [B, 256, 256]
        _, c_m, h_m, w_m = macro_feat.shape
        macro_tokens = macro_feat.flatten(2).transpose(1, 2)  # [B, 256, embed_dim]

        # Micro: [B, 256, 8, 8] -> [B, 64, 256]
        _, c_u, h_u, w_u = micro_feat.shape
        micro_tokens = micro_feat.flatten(2).transpose(1, 2)  # [B, 64, embed_dim]

        # 4. Cross-Attention
        attended_macro, attended_micro = self.cross_attention(macro_tokens, micro_tokens)

        # 5. Dual Pooling (GAP + GMP) on Macro
        macro_gap = attended_macro.mean(dim=1)  # [B, embed_dim]
        macro_gmp = attended_macro.max(dim=1)[0]  # [B, embed_dim]

        # Dual Pooling on Micro
        micro_gap = attended_micro.mean(dim=1)  # [B, embed_dim]
        micro_gmp = attended_micro.max(dim=1)[0]  # [B, embed_dim]

        # 6. Global Fused Representation
        fused = torch.cat([macro_gap, macro_gmp, micro_gap, micro_gmp], dim=1)  # [B, embed_dim * 4]

        # 7. Classification Logits
        logits = self.classifier(fused)
        return logits


def create_foot_arch_net_ultra(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.25,
) -> FootArchNetUltra:
    """Factory helper for FootArchNet-Ultra."""
    return FootArchNetUltra(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
