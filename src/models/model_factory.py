"""Model Factory for Flatfoot Radiograph Classification.

Provides unified interface for modern backbones:
- resnet50
- efficientnet_b2
- convnext_tiny
- densenet121
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


def create_model(
    model_name: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a CNN/ViT model with custom classifier head."""
    model_name = model_name.lower().replace("-", "_")

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "densenet201":
        weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
        model = models.densenet201(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_name in ["foot_arch_net", "footarchnet"]:
        from src.models.foot_arch_net import create_foot_arch_net
        model = create_foot_arch_net(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            backbone_type="efficientnet_b2",
        )

    elif model_name in ["foot_arch_net_v2", "footarchnet_v2", "footarchnetv2"]:
        from src.models.foot_arch_net_v2 import create_foot_arch_net_v2
        model = create_foot_arch_net_v2(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            backbone_type="convnext_tiny",
        )

    elif model_name in ["foot_arch_net_ultra", "footarchnet_ultra", "footarchnetultra"]:
        from src.models.foot_arch_net_ultra import create_foot_arch_net_ultra
        model = create_foot_arch_net_ultra(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    elif model_name in ["swin_t", "swin_tiny", "swin"]:
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            "Supported: foot_arch_net_ultra, foot_arch_net_v2, foot_arch_net, swin_t, densenet201, densenet121, resnet50, convnext_tiny, efficientnet_b2"
        )

    return model

