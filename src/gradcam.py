"""Grad-CAM explainability (placeholder for a future release)."""

from typing import Optional

import numpy as np
from PIL import Image


def generate_gradcam(
    model,
    image: Image.Image,
    target_class: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Produce a Grad-CAM heatmap overlay for the given image.

    TODO: Implement Grad-CAM using forward/backward hooks on the last
          convolutional block of EfficientNet-B0, then upsample and
          blend with the input image for display in the Streamlit UI.

    Args:
        model: Loaded classification model.
        image: Original PIL image.
        target_class: Class index to explain; None uses the predicted class.

    Returns:
        None until Grad-CAM is implemented.
    """
    return None
