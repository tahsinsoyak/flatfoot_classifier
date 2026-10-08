"""FootArchNet-V2: Hybrid CNN-Transformer Architecture with Biomechanical
Coordinate Convolution & Multi-Head Arch Attention for Radiographic Flatfoot Classification.

Key Innovations over V1:
1. CoordConv: Explicit (x, y) spatial metric grid channels to anchor ground platform and arch height.
2. TransArchAttention: Multi-Head Spatial Self-Attention (MHSA) Transformer block modeling long-range
   geometric dependencies between heel (posterior calcaneus), midfoot vault (navicular apex),
   and ball of the foot (1st metatarsal head).
3. Biomechanical Feature Pyramid Network (BFPN): Dense multi-scale lateral fusion connecting
   fine trabecular joint edges (64x64) with semantic foot curvature (16x16).
4. Anisotropic Strip Pooling V2: Horizontal arch span and vertical arch height cross-gating.
5. Dual Global Pooling (GAP + GMP) with LayerNorm and GELU regularized classification head.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CoordConv2d(nn.Module):
    """Appends normalized [-1, 1] 2D coordinate grid to feature maps,

    providing absolute spatial grounding for ground platform line and arch height.
    """

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
    """Anisotropic strip pooling module to capture horizontal arch span and vertical collapse."""

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


class TransArchAttention(nn.Module):
    """Multi-Head Spatial Self-Attention (MHSA) Transformer block over spatial feature tokens.

    Models long-range geometric dependencies across the medial longitudinal arch.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        grid_size: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Dynamic positional embedding interpolation if spatial size differs
        if tokens.shape[1] == self.pos_embed.shape[1]:
            pos = self.pos_embed
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2).reshape(1, c, self.grid_size, self.grid_size),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).flatten(2).transpose(1, 2)

        tokens = tokens + pos
        out_tokens = self.transformer(tokens)
        out_tokens = self.norm(out_tokens)
        out = out_tokens.transpose(1, 2).reshape(b, c, h, w)
        return out


class BiomechanicalFeaturePyramid(nn.Module):
    """Dense multi-scale feature pyramid with CoordConv and lateral connections."""

    def __init__(
        self,
        mid_channels: int,
        high_channels: int,
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        self.proj_mid = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.SiLU(inplace=True),
        )
        self.proj_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.SiLU(inplace=True),
        )
        self.coord_conv = CoordConv2d(out_channels, out_channels, kernel_size=3, bias=False)
        self.strip_pool = BiomechanicalStripPoolingV2(out_channels, pool_channels=64)
        self.trans_arch = TransArchAttention(d_model=out_channels, nhead=4, num_layers=2, grid_size=16)

        # Refinement convolution
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, f_mid: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        target_size = f_high.shape[2:]
        f_mid_aligned = F.interpolate(self.proj_mid(f_mid), size=target_size, mode="bilinear", align_corners=False)
        f_high_proj = self.proj_high(f_high)

        # Concatenate multi-scale representations
        f_cat = torch.cat([f_mid_aligned, f_high_proj], dim=1)

        # Apply CoordConv (spatial grounding)
        f_coord = self.coord_conv(f_cat)

        # Apply Strip Attention (anisotropic arch geometry)
        f_strip = self.strip_pool(f_coord)

        # Apply Multi-Head Self-Attention Transformer
        f_trans = self.trans_arch(f_strip)

        # Residual connection and refinement
        out = self.refine_conv(f_trans + f_strip)
        return out


class FootArchNetV2(nn.Module):
    """FootArchNet-V2: State-of-the-Art Deep Learning Model for Radiographic Flatfoot Classification.

    Integrates:
    - Pretrained Deep CNN Backbone (ConvNeXt-Tiny or EfficientNet-B2)
    - Biomechanical Feature Pyramid Network (BFPN)
    - Coordinate Convolutions (CoordConv)
    - Multi-Head Spatial Self-Attention (TransArch Transformer)
    - Anisotropic Strip Pooling V2
    - Dual Pooling Classification Head (GAP + GMP)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        backbone_type: str = "convnext_tiny",
    ) -> None:
        super().__init__()
        self.backbone_type = backbone_type

        if backbone_type == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            # ConvNeXt stages:
            # base.features[0..3]: stem + stage 1 (96 ch) + downsample + stage 2 (192 ch, 64x64)
            # base.features[4..7]: downsample + stage 3 (384 ch) + downsample + stage 4 (768 ch, 16x16)
            self.stage_low = nn.Sequential(*list(base.features[:4]))   # mid level: 192 ch at 64x64
            self.stage_high = nn.Sequential(*list(base.features[4:]))  # high level: 768 ch at 16x16
            mid_dim = 192
            high_dim = 768

        elif backbone_type == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b2(weights=weights)
            self.stage_low = nn.Sequential(*list(base.features[:4]))   # mid level: 48 ch at 64x64
            self.stage_high = nn.Sequential(*list(base.features[4:]))  # high level: 1408 ch at 16x16
            mid_dim = 48
            high_dim = 1408

        elif backbone_type == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            base = models.densenet121(weights=weights)
            # DenseNet stages
            self.stage_low = nn.Sequential(
                base.features.conv0, base.features.norm0, base.features.relu0, base.features.pool0,
                base.features.denseblock1, base.features.transition1,
                base.features.denseblock2, base.features.transition2,
            )  # 256 ch at 64x64
            self.stage_high = nn.Sequential(
                base.features.denseblock3, base.features.transition3,
                base.features.denseblock4, base.features.norm5,
            )  # 1024 ch at 16x16
            mid_dim = 256
            high_dim = 1024
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        # Multi-scale Transformer & CoordConv Pyramid
        self.fusion_channels = 256
        self.pyramid = BiomechanicalFeaturePyramid(
            mid_channels=mid_dim,
            high_channels=high_dim,
            out_channels=self.fusion_channels,
        )

        # Classification Head with Dual Pooling (GAP + GMP)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_channels * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(64, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_mid = self.stage_low(x)
        f_high = self.stage_high(f_mid)
        f_fused = self.pyramid(f_mid, f_high)
        return f_mid, f_high, f_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, f_fused = self.extract_features(x)

        # Dual pooling: Global Average Pooling + Global Max Pooling
        avg_pool = F.adaptive_avg_pool2d(f_fused, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(f_fused, 1).flatten(1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        logits = self.classifier(pooled)
        return logits


def create_foot_arch_net_v2(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    backbone_type: str = "convnext_tiny",
) -> FootArchNetV2:
    """Factory function for FootArchNet-V2."""
    return FootArchNetV2(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        backbone_type=backbone_type,
    )


if __name__ == "__main__":
    print("Testing FootArchNet-V2 (ConvNeXt-Tiny backbone)...")
    m = create_foot_arch_net_v2(pretrained=False, backbone_type="convnext_tiny")
    x = torch.randn(2, 3, 512, 512)
    y = m(x)
    print(f"Output shape: {y.shape}")
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.2f}M")
