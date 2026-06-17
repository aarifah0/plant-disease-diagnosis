"""Image loading, validation, and preprocessing for the plant disease model."""
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.utils import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ImageValidationError(ValueError):
    """Raised when an uploaded file fails validation before inference."""


def validate_upload(filename: str, size_bytes: int) -> None:
    """Validate file extension and size against the product limits.

    Raises ImageValidationError with a user-facing message on failure.
    """
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Please upload a JPG, PNG, or JPEG image.")

    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise ImageValidationError(f"File exceeds the {MAX_UPLOAD_SIZE_MB}MB limit.")


def load_image(source) -> Image.Image:
    """Load an image from a file path, file-like object, or bytes into RGB.

    Raises ImageValidationError if the file cannot be read as an image.
    """
    if isinstance(source, (bytes, bytearray)):
        source = BytesIO(source)
    try:
        return Image.open(source).convert("RGB")
    except UnidentifiedImageError as e:
        raise ImageValidationError("Unable to read image file.") from e


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image into a normalized model input batch of shape (1, 3, H, W)."""
    return inference_transform(image).unsqueeze(0)
