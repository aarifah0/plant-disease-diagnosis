"""Evaluate the trained ResNet50 on the validation split.

Outputs (written to results/):
  classification_report.csv  — per-class precision, recall, F1, support
  confusion_matrix.png       — row-normalised heatmap (% of true class)
  summary.json               — overall accuracy + macro/weighted averages

Usage:
  python -m src.evaluate --data-dir data/raw
  python -m src.evaluate --data-dir data/raw --weights models/resnet50_plantdisease.pth
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import load_model
from src.preprocess import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.utils import DEFAULT_WEIGHTS_PATH, format_class_name, get_device, load_class_names

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Two-letter plant abbreviations for compact confusion matrix axis labels.
_PLANT_ABBREV = {
    "Apple": "Ap", "Blueberry": "Bl", "Cherry": "Ch", "Corn": "Co",
    "Grape": "Gr", "Orange": "Or", "Peach": "Pe", "Pepper": "PP",
    "Potato": "Po", "Raspberry": "Ra", "Soybean": "So", "Squash": "Sq",
    "Strawberry": "St", "Tomato": "To",
}


def _short_label(class_name: str) -> str:
    """'Tomato___Late_blight' → 'To-Late blight' for compact axis tick labels."""
    label = format_class_name(class_name)          # 'Tomato - Late blight'
    plant, sep, condition = label.partition(" - ")
    abbrev = _PLANT_ABBREV.get(plant, plant[:2])
    return f"{abbrev}-{condition}" if sep else abbrev


def _capped_indices(dataset: datasets.ImageFolder, max_per_class: int, seed: int = 42) -> list:
    """Pick up to max_per_class sample indices per class, deterministically."""
    import random
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    rng = random.Random(seed)
    selected = []
    for indices in by_class.values():
        rng.shuffle(indices)
        selected.extend(indices[:max_per_class])
    return selected


def run_inference(model, data_dir: Path, batch_size: int, num_workers: int, device,
                  max_per_class: int = None):
    """Run the model over the validation split and return (all_labels, all_preds, class_names)."""
    val_dir = data_dir / "valid"
    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}\n"
            "Run 'python -m src.download_data' to get the dataset."
        )

    dataset = datasets.ImageFolder(val_dir, transform=_eval_transform)
    if max_per_class is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, _capped_indices(dataset, max_per_class))
        print(f"  (capped to {max_per_class} images/class → {len(dataset)} total)")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    total_images = len(dataset)
    all_labels, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_labels.append(labels)
            all_preds.append(preds)
            if (i + 1) % 5 == 0:
                print(f"  Processed {min((i + 1) * batch_size, total_images)}/{total_images} images...", flush=True)

    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
        dataset.classes,
    )


def save_classification_report(y_true, y_pred, class_names, out_dir: Path):
    """Write a per-class CSV and return the report dict."""
    import pandas as pd

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    rows = []
    for name in class_names:
        row = report[name]
        rows.append({
            "class": name,
            "label": format_class_name(name),
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1_score": round(row["f1-score"], 4),
            "support": int(row["support"]),
        })
    df = pd.DataFrame(rows)
    path = out_dir / "classification_report.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return report


def save_confusion_matrix(y_true, y_pred, class_names, out_dir: Path):
    """Save a row-normalised (recall) confusion matrix heatmap as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    # Normalise each row so values are recall (fraction of true class).
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    short_labels = [_short_label(c) for c in class_names]
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(22, 19))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (fraction of true class)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Confusion Matrix — row-normalised (recall per class)", fontsize=13, pad=14)

    # Annotate cells only where recall >= 5% to avoid clutter.
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val >= 0.05:
                ax.text(j, i, f"{val:.0%}",
                        ha="center", va="center", fontsize=5.5,
                        color="white" if val > thresh else "black")

    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_summary(report: dict, accuracy: float, elapsed: float, n_samples: int, out_dir: Path):
    summary = {
        "overall_accuracy": round(accuracy, 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "n_samples": n_samples,
        "inference_seconds_total": round(elapsed, 1),
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {path}")
    return summary


def print_summary(summary: dict):
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Overall accuracy : {summary['overall_accuracy'] * 100:.2f}%")
    print(f"  Macro F1         : {summary['macro_f1'] * 100:.2f}%")
    print(f"  Macro precision  : {summary['macro_precision'] * 100:.2f}%")
    print(f"  Macro recall     : {summary['macro_recall'] * 100:.2f}%")
    print(f"  Weighted F1      : {summary['weighted_f1'] * 100:.2f}%")
    print(f"  Samples evaluated: {summary['n_samples']:,}")
    print(f"  Inference time   : {summary['inference_seconds_total']:.1f}s total")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the plant disease model on the validation set.")
    parser.add_argument("--data-dir", required=True, help="Directory containing the valid/ subfolder.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH), help="Path to model weights.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-valid-per-class", type=int, default=None, help="Cap images per class (fast approximate eval).")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR), help="Where to write output files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading imports and model (this may take 15-30s on first run)...", flush=True)

    device = get_device()
    print(f"Device : {device}")
    print(f"Weights: {args.weights}")
    print(f"Data   : {args.data_dir}/valid")
    print()

    class_names = load_class_names()
    print("Loading model weights...", flush=True)
    model = load_model(args.weights, num_classes=len(class_names), device=device)
    print("Model loaded.", flush=True)

    print("Running inference on validation set...")
    t0 = time.perf_counter()
    y_true, y_pred, dataset_classes = run_inference(
        model, Path(args.data_dir), args.batch_size, args.num_workers, device,
        max_per_class=args.max_valid_per_class,
    )
    elapsed = time.perf_counter() - t0

    # Warn if dataset class order differs from class_names.json.
    if dataset_classes != class_names:
        print(
            "WARNING: dataset class order does not match class_names.json. "
            "Using the order discovered from the dataset folder."
        )
        class_names = dataset_classes

    accuracy = (y_true == y_pred).mean()
    report = save_classification_report(y_true, y_pred, class_names, out_dir)
    save_confusion_matrix(y_true, y_pred, class_names, out_dir)
    summary = save_summary(report, accuracy, elapsed, len(y_true), out_dir)
    print_summary(summary)


if __name__ == "__main__":
    main()
