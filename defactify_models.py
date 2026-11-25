"""Model and inference utilities for the Defactify image detector.

Checkpoint schema used by the training notebook
-----------------------------------------------
A frontend-safe checkpoint is a ``torch.save``-d dictionary containing only
PyTorch tensors plus plain Python metadata, so it can be loaded with
``torch.load(..., weights_only=True)``::

    {
        "schema_version": 1,
        "model_name": "rgb_baseline" | "rgb_fourier_concat" | "rgb_fourier_attention_aux",
        "model_config": {"fusion": "concat" | "attention" | None, "auxiliary": bool},
        "model_state_dict": model.state_dict(),
        "image_size": 224,
        "threshold": 0.5,
        "source_names": {0: "real", 1: "stable-diffusion", ...},
        # optional training-only keys may be present; inference ignores them
    }

The loader below validates the architecture name/configuration and always
constructs models with ``pretrained=False`` to avoid network downloads during
inference. The returned score is a detector probability estimate, not proof
that an individual image is real or AI-generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

DEFAULT_IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 0.5
CHECKPOINT_SCHEMA_VERSION = 1

MODEL_RGB_BASELINE = "rgb_baseline"
MODEL_RGB_FOURIER_CONCAT = "rgb_fourier_concat"
MODEL_RGB_FOURIER_ATTENTION_AUX = "rgb_fourier_attention_aux"
SUPPORTED_MODEL_NAMES = (
    MODEL_RGB_BASELINE,
    MODEL_RGB_FOURIER_CONCAT,
    MODEL_RGB_FOURIER_ATTENTION_AUX,
)

RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DETECTOR_CAUTION = (
    "This score is a model estimate for AI-generated versus real-photograph "
    "classes; it is not proof that a specific image is genuine or fake."
)


def make_image_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Return the RGB tensor preprocessing used during training and inference."""
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
        ]
    )


image_transform = make_image_transform(DEFAULT_IMAGE_SIZE)


def prepare_image(
    image: Image.Image | str | Path, image_size: int = DEFAULT_IMAGE_SIZE
) -> torch.Tensor:
    """Load/convert one image and return a batched ``[1, 3, H, W]`` tensor."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            pil_image = opened.convert("RGB")
            return make_image_transform(image_size)(pil_image).unsqueeze(0)
    return make_image_transform(image_size)(image.convert("RGB")).unsqueeze(0)


def normalize_rgb(x: torch.Tensor) -> torch.Tensor:
    return (x - RGB_MEAN.to(x.device)) / RGB_STD.to(x.device)


def fourier_map(x: torch.Tensor) -> torch.Tensor:
    gray = x.mean(dim=1, keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1))
    magnitude = torch.log1p(torch.abs(spectrum))
    std = magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (magnitude - magnitude.mean(dim=(-2, -1), keepdim=True)) / std


class RGBBaseline(nn.Module):
    """ResNet-18 RGB binary baseline matching the notebook architecture."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.encoder(normalize_rgb(x)).squeeze(-1), None


class TwoBranchDetector(nn.Module):
    """RGB/Fourier detector with concat or attention fusion."""

    def __init__(
        self, fusion: str = "concat", auxiliary: bool = False, pretrained: bool = True
    ):
        super().__init__()
        if fusion not in ("concat", "attention"):
            raise ValueError("fusion must be 'concat' or 'attention'")
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.rgb_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.frequency_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.fusion = fusion
        if fusion == "attention":
            self.rgb_projection = nn.Conv2d(512, 128, 1)
            self.cross_attention = nn.MultiheadAttention(
                128, num_heads=4, batch_first=True
            )
        self.head = nn.Linear(512 + 128, 1)
        self.source_head = nn.Linear(512 + 128, 6) if auxiliary else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        rgb = self.rgb_encoder(normalize_rgb(x))
        freq = self.frequency_encoder(fourier_map(x))
        rgb_pooled = rgb.mean(dim=(-2, -1))
        if self.fusion == "attention":
            query = self.rgb_projection(rgb).flatten(2).transpose(1, 2)
            key_value = freq.flatten(2).transpose(1, 2)
            attended, _ = self.cross_attention(
                query, key_value, key_value, need_weights=False
            )
            freq_features = attended.mean(dim=1)
        else:
            freq_features = freq.mean(dim=(-2, -1))
        features = torch.cat([rgb_pooled, freq_features], dim=1)
        binary_logits = self.head(features).squeeze(-1)
        source_logits = (
            self.source_head(features) if self.source_head is not None else None
        )
        return binary_logits, source_logits


def expected_model_config(model_name: str) -> dict[str, Any]:
    """Return the canonical architecture config for one experiment name."""
    if model_name == MODEL_RGB_BASELINE:
        return {"fusion": None, "auxiliary": False}
    if model_name == MODEL_RGB_FOURIER_CONCAT:
        return {"fusion": "concat", "auxiliary": False}
    if model_name == MODEL_RGB_FOURIER_ATTENTION_AUX:
        return {"fusion": "attention", "auxiliary": True}
    raise ValueError(
        f"Unsupported model_name {model_name!r}; expected one of {SUPPORTED_MODEL_NAMES}"
    )


def _validate_model_config(
    model_name: str, model_config: Mapping[str, Any] | None
) -> dict[str, Any]:
    expected = expected_model_config(model_name)
    if model_config is None:
        return expected
    actual = dict(model_config)
    for key, expected_value in expected.items():
        if key in actual and actual[key] != expected_value:
            raise ValueError(
                f"Checkpoint model_config[{key!r}]={actual[key]!r} does not match "
                f"{model_name!r} expected value {expected_value!r}"
            )
    return {**expected, **actual}


def build_detector(
    model_name: str,
    model_config: Mapping[str, Any] | None = None,
    pretrained: bool = False,
) -> nn.Module:
    """Construct one of the three planned detector architectures.

    ``pretrained`` defaults to ``False`` so imports, tests, and frontend
    inference never download ImageNet weights accidentally. Training code may
    pass ``pretrained=True`` explicitly.
    """
    config = _validate_model_config(model_name, model_config)
    if model_name == MODEL_RGB_BASELINE:
        return RGBBaseline(pretrained=pretrained)
    return TwoBranchDetector(
        fusion=str(config["fusion"]),
        auxiliary=bool(config["auxiliary"]),
        pretrained=pretrained,
    )


def model_from_config(
    model_name: str,
    model_config: Mapping[str, Any] | None = None,
    pretrained: bool = False,
) -> nn.Module:
    """Alias used by notebook code that builds models from checkpoint metadata."""
    return build_detector(model_name, model_config, pretrained=pretrained)


def make_detector_checkpoint(
    model: nn.Module,
    model_name: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    source_names: Mapping[int, str] | None = None,
    model_config: Mapping[str, Any] | None = None,
    **extra_metadata: Any,
) -> dict[str, Any]:
    """Build the checkpoint dictionary expected by ``load_detector``."""
    config = _validate_model_config(model_name, model_config)
    checkpoint: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_name": model_name,
        "model_config": config,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "image_size": int(image_size),
        "threshold": float(threshold),
        "source_names": dict(source_names or {0: "real", 1: "AI-generated"}),
    }
    checkpoint.update(extra_metadata)
    return checkpoint


@dataclass(frozen=True)
class LoadedDetector:
    """Loaded detector bundle for frontend or notebook inference."""

    model: nn.Module
    model_name: str
    model_config: Mapping[str, Any]
    image_size: int
    threshold: float
    source_names: Mapping[Any, Any]
    device: torch.device

    def predict_proba(self, image: Image.Image | str | Path) -> float:
        return predict_ai_probability(self.model, image, self.image_size, self.device)

    def predict(self, image: Image.Image | str | Path) -> dict[str, Any]:
        return prediction_report(self, image)


def load_detector(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> LoadedDetector:
    """Safely load a trained detector checkpoint for inference.

    The checkpoint is read with ``weights_only=True`` and the architecture is
    reconstructed locally with ``pretrained=False``. This avoids arbitrary class
    unpickling and external pretrained-weight downloads at inference time.
    """
    target_device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=target_device, weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Detector checkpoint must be a dictionary")

    schema_version = checkpoint.get("schema_version", CHECKPOINT_SCHEMA_VERSION)
    if int(schema_version) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema_version {schema_version!r}")

    model_name = checkpoint.get("model_name")
    if not isinstance(model_name, str):
        raise ValueError("Detector checkpoint is missing string key 'model_name'")
    model_config = _validate_model_config(model_name, checkpoint.get("model_config"))

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        # Backward-compatible with earlier notebook drafts; new checkpoints
        # should always use 'model_state_dict'.
        state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(
            "Detector checkpoint is missing mapping key 'model_state_dict'"
        )

    model = build_detector(model_name, model_config, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(target_device).eval()

    return LoadedDetector(
        model=model,
        model_name=model_name,
        model_config=model_config,
        image_size=int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE)),
        threshold=float(checkpoint.get("threshold", DEFAULT_THRESHOLD)),
        source_names=checkpoint.get("source_names", {0: "real", 1: "AI-generated"}),
        device=target_device,
    )


def predict_ai_probability(
    model: nn.Module,
    image: Image.Image | str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: str | torch.device | None = None,
) -> float:
    """Return the model's AI-generated probability estimate for one image."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    target_device = torch.device(device)
    was_training = model.training
    model.eval()
    batch = prepare_image(image, image_size).to(target_device)
    with torch.no_grad():
        logits, _ = model(batch)
        probability = torch.sigmoid(logits).detach().cpu().reshape(-1)[0].item()
    if was_training:
        model.train()
    return float(probability)


def prediction_report(
    detector: LoadedDetector, image: Image.Image | str | Path
) -> dict[str, Any]:
    """Return a frontend-friendly score dict with the required caution text."""
    probability = detector.predict_proba(image)
    predicted_label = (
        "AI-generated" if probability >= detector.threshold else "real photograph"
    )
    return {
        "ai_probability": probability,
        "threshold": detector.threshold,
        "predicted_label": predicted_label,
        "caution": DETECTOR_CAUTION,
    }
