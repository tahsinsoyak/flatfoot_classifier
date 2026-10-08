"""FootArchNet: Biomechanically-Guided Multi-Scale Attention Network
for Radiographic Pes Planus (Flatfoot) Classification.

Designed specifically for weight-bearing lateral foot radiographs by combining:
1. Multi-scale feature extraction (joint spaces + global longitudinal arch morphology).
2. Biomechanical Strip Attention Module (BSAM) for anisotropic foot geometry.
3. Spatial & Channel Attention Gating (SCAG) targeting the medial longitudinal arch.
4. Dual-pooling (AvgPool + MaxPool) robust classification head.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class StripPooling(nn.Module):
    """Anisotropic Strip Pooling module to capture horizontal arch span and vertical arch height."""

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


class BiomechanicalAttentionModule(nn.Module):
    """Channel and Spatial Attention module focused on tarsal joints and longitudinal arch."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(16, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        # Strip Pooling for anisotropic foot geometry
        self.strip_pool = StripPooling(in_channels, pool_channels=mid_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention
        ca = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        x = x * self.channel_sigmoid(ca)

        # Strip Attention
        x = self.strip_pool(x)

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        return x


class MultiScaleFusionBlock(nn.Module):
    """Fuses intermediate localized joint features with high-level global arch morphology."""

    def __init__(self, mid_channels: int, high_channels: int, out_channels: int = 384) -> None:
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
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.attention = BiomechanicalAttentionModule(out_channels)

    def forward(self, f_mid: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions to high-level resolution
        target_size = f_high.shape[2:]
        f_mid_proj = F.interpolate(self.proj_mid(f_mid), size=target_size, mode="bilinear", align_corners=False)
        f_high_proj = self.proj_high(f_high)

        # Concat and refine
        concat = torch.cat([f_mid_proj, f_high_proj], dim=1)
        fused = self.fusion_conv(concat)
        fused = self.attention(fused)
        return fused


class FootArchNet(nn.Module):
    """FootArchNet: Custom Deep Learning Model for Flatfoot Radiograph Classification.

    Integrates:
    - Multi-scale backbone feature taps (Stage 3 & Stage 5 of EfficientNet or ResNet)
    - MultiScaleFusionBlock with Biomechanical Strip Attention
    - Dual Global Pooling (Avg + Max)
    - Regularized MLP classification head with LayerNorm and GELU
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        backbone_type: str = "efficientnet_b2",
    ) -> None:
        super().__init__()
        self.backbone_type = backbone_type

        if backbone_type == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b2(weights=weights)
            # Feature stages
            # features[0..3]: stem and low/mid features (e.g. 48 channels at 64x64)
            # features[4..7]: mid/high features (e.g. 1408 channels at 16x16)
            self.stage_low = nn.Sequential(*list(base.features[:4]))  # mid level: 48 ch
            self.stage_high = nn.Sequential(*list(base.features[4:]))  # high level: 1408 ch
            mid_dim = 48
            high_dim = 1408
        elif backbone_type == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            self.stage_low = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2)  # 512 ch
            self.stage_high = nn.Sequential(base.layer3, base.layer4)  # 2048 ch
            mid_dim = 512
            high_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        # Multi-scale Biomechanical Fusion
        self.fusion_channels = 384
        self.fusion_block = MultiScaleFusionBlock(
            mid_channels=mid_dim,
            high_channels=high_dim,
            out_channels=self.fusion_channels,
        )

        # Classification Head with Dual Pooling
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_channels * 2, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(192, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(64, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_mid = self.stage_low(x)
        f_high = self.stage_high(f_mid)
        f_fused = self.fusion_block(f_mid, f_high)
        return f_mid, f_high, f_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, f_fused = self.extract_features(x)

        # Dual pooling: Global Average Pooling + Global Max Pooling
        avg_pool = F.adaptive_avg_pool2d(f_fused, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(f_fused, 1).flatten(1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        logits = self.classifier(pooled)
        return logits


def create_foot_arch_net(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    backbone_type: str = "efficientnet_b2",
) -> FootArchNet:
    """Factory function for FootArchNet."""
    return FootArchNet(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        backbone_type=backbone_type,
    )


if __name__ == "__main__":
    # Smoke test
    print("Testing FootArchNet (EfficientNet-B2 backbone)...")
    m = create_foot_arch_net(pretrained=False, backbone_type="efficientnet_b2")
    x = torch.randn(2, 3, 512, 512)
    y = m(x)
    print(f"Output shape: {y.shape}")
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.2f}M")
