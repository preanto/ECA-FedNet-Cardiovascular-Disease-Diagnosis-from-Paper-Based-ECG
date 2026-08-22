"""
Grad-CAM interpretability analysis (Sec. IV-H, Fig. 10).

Activation maps are taken from the final convolutional stage of the
EfficientNet-B3 backbone.  Setting `after_attention=True` instead hooks the
output of the CBAM block, which is useful for comparing where the attention
module redirects the spatial focus relative to the plain backbone.

The manuscript is explicit that this analysis is qualitative: the maps were not
scored against clinician-annotated regions, and no localisation metric was
computed (Sec. V-E).  Nothing in this module should be read as evidence of
localisation accuracy.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Standard Grad-CAM: channel weights are the spatially averaged gradients
    of the target logit with respect to the chosen activation map."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(self, input_tensor: torch.Tensor,
                 class_index: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """Return (cam in [0, 1] at the activation resolution, class index, probability)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        score = logits[:, class_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks captured no activations; check the target layer.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, class_index, float(probs[0, class_index].item())


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray,
                alpha: float = 0.4,
                colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Blend a heat map onto the original RGB image."""
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (1 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap.astype(np.float32)
    return np.uint8(np.clip(blended, 0, 255))


def resolve_target_layer(model: nn.Module, after_attention: bool = False) -> nn.Module:
    """Pick the layer to hook.

    Default: the final convolutional stage of the backbone, as described in
    Sec. IV-H.  `after_attention=True` hooks the CBAM output instead.
    """
    if after_attention and hasattr(model, "attention"):
        return model.attention
    if hasattr(model, "gradcam_target_layer"):
        return model.gradcam_target_layer()
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Could not resolve a Grad-CAM target layer for this model.")


def generate_cams(model: nn.Module,
                  images: torch.Tensor,
                  device: torch.device,
                  class_indices: Optional[List[int]] = None,
                  after_attention: bool = False) -> List[Tuple[np.ndarray, int, float]]:
    """Grad-CAM for a batch, computed one sample at a time (gradients required)."""
    target_layer = resolve_target_layer(model, after_attention)
    results = []
    with GradCAM(model, target_layer) as cam_extractor:
        for i in range(images.shape[0]):
            single = images[i:i + 1].to(device).requires_grad_(True)
            target = None if class_indices is None else class_indices[i]
            results.append(cam_extractor(single, target))
    return results
