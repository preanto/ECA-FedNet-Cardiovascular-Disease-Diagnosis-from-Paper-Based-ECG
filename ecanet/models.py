"""
ECA-Net: EfficientNet-B3 backbone with a sequential channel-then-spatial
attention module (Sec. III-F, Fig. 5).

Parameter budget reproduced from Tables IV and VII (three-class head):

    EfficientNet-B3 backbone      10,696,232    97.28 %
    Channel attention (r = 16)       294,912     2.68 %
    Spatial attention (7 x 7)             98   0.0009 %
    Classification head                4,611    0.042 %
    -----------------------------------------------------
    Total                         10,995,853      100 %

The four ablation configurations of Table XII are obtained through the
`attention` argument: 'none' (M_base), 'channel' (M_CA), 'spatial' (M_SA) and
'dual' (M_dual, the proposed model).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

from .config import ModelConfig


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM), Eq. (1).

        M_c(F) = sigma( MLP(AvgPool(F)) + MLP(MaxPool(F)) )

    The shared MLP is implemented with two 1x1 convolutions and a reduction
    ratio r = 16, i.e. 1536 -> 96 -> 1536 for the EfficientNet-B3 feature map.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self._mlp(self.avg_pool(x)) + self._mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM), Eq. (3).

        M_s(F') = sigma( f_7x7([AvgPool_c(F'); MaxPool_c(F')]) )

    A single 7x7 convolution over the two channel-wise statistic maps: 98
    parameters (2 x 7 x 7, no bias).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))


class CBAM(nn.Module):
    """Sequential channel -> spatial attention (Eqs. 1-4).

    The ordering matters: the spatial map is computed from the already
    channel-recalibrated feature map F', so suppressing an uninformative channel
    changes where the spatial branch attends (Sec. III-F).

    `mode` selects the ablation configuration of Table XII.
    """

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7, mode: str = "dual"):
        super().__init__()
        if mode not in {"none", "channel", "spatial", "dual"}:
            raise ValueError(f"Unknown attention mode: {mode}")
        self.mode = mode
        self.channel_attention = (ChannelAttention(channels, reduction)
                                  if mode in {"channel", "dual"} else None)
        self.spatial_attention = (SpatialAttention(spatial_kernel)
                                  if mode in {"spatial", "dual"} else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_attention is not None:
            x = x * self.channel_attention(x)        # F' = M_c(F) (x) F
        if self.spatial_attention is not None:
            x = x * self.spatial_attention(x)        # F'' = M_s(F') (x) F'
        return x


# ---------------------------------------------------------------------------
# ECA-Net
# ---------------------------------------------------------------------------
class ECANet(nn.Module):
    """Efficient Cardio-Attention Network (Algorithm 1)."""

    def __init__(self,
                 num_classes: int,
                 attention: str = "dual",
                 reduction: int = 16,
                 spatial_kernel: int = 7,
                 dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        weights = tvm.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = tvm.efficientnet_b3(weights=weights)
        channels = backbone.classifier[1].in_features      # 1536

        self.features = backbone.features                   # -> (B, 1536, 10, 10) at 300 px
        self.attention = CBAM(channels, reduction, spatial_kernel, mode=attention)
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(channels, num_classes))
        self.attention_mode = attention
        self.feature_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # F
        x = self.attention(x)         # F''
        x = self.avgpool(x)           # global average pooling
        return self.classifier(torch.flatten(x, 1))

    # -- helpers -----------------------------------------------------------
    def parameter_groups(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """(backbone parameters, head parameters).

        The manuscript (Table III) uses one learning rate for all parameters;
        this split is only needed for the optional freeze warm-up and for
        optimisers that treat the head separately.
        """
        head = list(self.attention.parameters()) + list(self.classifier.parameters())
        head_ids = {id(p) for p in head}
        backbone = [p for p in self.parameters() if id(p) not in head_ids]
        return backbone, head

    def gradcam_target_layer(self) -> nn.Module:
        """Final convolutional stage of the backbone (Sec. IV-H)."""
        return self.features[-1]

    def parameter_breakdown(self) -> dict:
        """Reproduce Table VII."""
        def _n(module) -> int:
            return sum(p.numel() for p in module.parameters())

        channel = _n(self.attention.channel_attention) if self.attention.channel_attention else 0
        spatial = _n(self.attention.spatial_attention) if self.attention.spatial_attention else 0
        breakdown = {
            "EfficientNet-B3 Backbone": _n(self.features),
            "Channel Attention": channel,
            "Spatial Attention": spatial,
            "Classification Head": _n(self.classifier),
        }
        breakdown["Total"] = sum(breakdown.values())
        return breakdown


# ---------------------------------------------------------------------------
# Baselines — Table VI / Table X
# ---------------------------------------------------------------------------
_BASELINE_FACTORIES = {
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights, "classifier.1"),
    "efficientnet_b3": (tvm.efficientnet_b3, tvm.EfficientNet_B3_Weights, "classifier.1"),
    "efficientnet_v2_s": (tvm.efficientnet_v2_s, tvm.EfficientNet_V2_S_Weights, "classifier.1"),
    "densenet121": (tvm.densenet121, tvm.DenseNet121_Weights, "classifier"),
    "resnet34": (tvm.resnet34, tvm.ResNet34_Weights, "fc"),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights, "classifier.3"),
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights, "classifier.2"),
    "vit_b_16": (tvm.vit_b_16, tvm.ViT_B_16_Weights, "heads.head"),
    "swin_t": (tvm.swin_t, tvm.Swin_T_Weights, "head"),
}


def _replace_head(model: nn.Module, path: str, num_classes: int) -> nn.Module:
    """Swap the classifier at a dotted attribute path and return the new layer."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    old = parent[int(last)] if last.isdigit() else getattr(parent, last)
    new = nn.Linear(old.in_features, num_classes)
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)
    return new


def build_model(name: str,
                num_classes: int,
                cfg: ModelConfig | None = None) -> Tuple[nn.Module, List[nn.Parameter]]:
    """Return (model, head_parameters) for ECA-Net or any comparison backbone.

    All models are initialised with ImageNet-pretrained weights (Sec. IV-A) and
    trained under one identical protocol.
    """
    cfg = cfg or ModelConfig()

    if name in {"ECA-Net", "eca_net", "ecanet"}:
        model = ECANet(num_classes,
                       attention=cfg.attention,
                       reduction=cfg.reduction_ratio,
                       spatial_kernel=cfg.spatial_kernel,
                       dropout=cfg.dropout,
                       pretrained=cfg.pretrained)
        _, head = model.parameter_groups()
        return model, head

    if name not in _BASELINE_FACTORIES:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Available: ECA-Net, {', '.join(_BASELINE_FACTORIES)}")

    factory, weights_enum, head_path = _BASELINE_FACTORIES[name]
    model = factory(weights=weights_enum.DEFAULT if cfg.pretrained else None)
    head = _replace_head(model, head_path, num_classes)
    return model, list(head.parameters())


def build_ablation_model(variant: str, num_classes: int,
                         cfg: ModelConfig | None = None) -> nn.Module:
    """Build one of the Table XII configurations ('none'/'channel'/'spatial'/'dual')."""
    cfg = cfg or ModelConfig()
    return ECANet(num_classes,
                  attention=variant,
                  reduction=cfg.reduction_ratio,
                  spatial_kernel=cfg.spatial_kernel,
                  dropout=cfg.dropout,
                  pretrained=cfg.pretrained)
