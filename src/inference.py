"""Image preprocessing and model inference utilities."""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import create_model

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

# ImageNet normalization used by pretrained EfficientNet
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Resolve weights relative to project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_PATH = _PROJECT_ROOT / "weights" / "best_model.pth"


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a normalized batch tensor (1, 3, 224, 224).

    Args:
        image: PIL Image in any mode; converted to RGB before transform.

    Returns:
        Float tensor on CPU, ready for model forward pass.
    """
    rgb = image.convert("RGB")
    tensor = _TRANSFORM(rgb)
    return tensor.unsqueeze(0)


def load_model(
    weights_path: Path | str | None = None,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """
    Load EfficientNet-B0 with trained weights from disk.

    Args:
        weights_path: Path to .pth checkpoint; defaults to weights/best_model.pth.
        device: Target device ('cpu' recommended for deployment).

    Returns:
        Model in eval mode on the requested device.

    Raises:
        FileNotFoundError: If the weights file does not exist.
    """
    path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found at: {path}")

    model = create_model(num_classes=len(CLASS_NAMES))
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict(
    model: torch.nn.Module,
    image: Image.Image,
    device: str | torch.device = "cpu",
) -> Tuple[str, float, Dict[str, float]]:
    """
    Run inference on a single MRI image.

    Args:
        model: Loaded PyTorch model in eval mode.
        image: PIL Image upload.
        device: Device for inference.

    Returns:
        Tuple of (predicted class name, confidence score, per-class probability dict).
    """
    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(probs.argmax())
    class_name = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    prob_dict = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}

    return class_name, confidence, prob_dict
