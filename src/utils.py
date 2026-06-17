"""Shared helpers: paths, JSON loading, device selection, label formatting."""
import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_CLASS_NAMES_PATH = DATA_DIR / "class_names.json"
DEFAULT_DISEASE_DB_PATH = DATA_DIR / "disease_db.json"
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "resnet50_plantdisease.pth"

MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def get_device() -> torch.device:
    """Return the best available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_json(path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def load_class_names(path=DEFAULT_CLASS_NAMES_PATH) -> list:
    return load_json(path)


def load_disease_db(path=DEFAULT_DISEASE_DB_PATH) -> dict:
    return load_json(path)


def format_class_name(raw_name: str) -> str:
    """Turn a raw class name into a readable label, e.g. 'Tomato___Late_blight' -> 'Tomato - Late blight'."""
    plant, _, condition = raw_name.partition("___")
    condition = condition.replace("_", " ").strip()
    return f"{plant} - {condition}" if condition else plant


def confidence_level(confidence: float) -> str:
    """Bucket a 0-1 confidence score into 'high', 'medium', or 'low'."""
    if confidence > 0.80:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"
