"""Train the plant disease ResNet50 classifier.

Defaults to freezing the convolutional backbone and training only the final
FC layer, which keeps training practical on CPU-only machines. Pass
--full-finetune to unfreeze the whole network instead.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.model import build_model
from src.preprocess import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.utils import DEFAULT_WEIGHTS_PATH, DATA_DIR, get_device

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def capped_indices_per_class(dataset: datasets.ImageFolder, max_per_class: int, seed: int = 42) -> list:
    """Pick up to max_per_class sample indices for each class, shuffled per class."""
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    selected = []
    for indices in by_class.values():
        rng.shuffle(indices)
        selected.extend(indices[:max_per_class])
    return selected


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    max_train_per_class: int = None,
    max_valid_per_class: int = None,
):
    train_full = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    valid_full = datasets.ImageFolder(data_dir / "valid", transform=eval_transform)
    classes = train_full.classes

    train_ds = train_full
    if max_train_per_class is not None:
        train_ds = Subset(train_full, capped_indices_per_class(train_full, max_train_per_class))

    valid_ds = valid_full
    if max_valid_per_class is not None:
        valid_ds = Subset(valid_full, capped_indices_per_class(valid_full, max_valid_per_class))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, classes


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def run_epoch(model, loader, criterion, optimizer, device, train: bool, max_batches: int = None):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def sync_class_names(classes: list, path: Path) -> None:
    """Overwrite data/class_names.json if the dataset's class order differs."""
    expected = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if classes != expected:
        print(f"Updating {path} to match the dataset's {len(classes)} discovered classes.")
        path.write_text(json.dumps(classes, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train the plant disease ResNet50 classifier.")
    parser.add_argument("--data-dir", required=True, help="Directory containing train/ and valid/ subfolders.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--full-finetune", dest="freeze_backbone", action="store_false", default=True)
    parser.add_argument("--output", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug: cap train batches per epoch.")
    parser.add_argument("--max-valid-batches", type=int, default=None, help="Debug: cap valid batches per epoch.")
    parser.add_argument("--max-train-per-class", type=int, default=None, help="Cap train images per class (subset training).")
    parser.add_argument("--max-valid-per-class", type=int, default=None, help="Cap valid images per class (subset training).")
    args = parser.parse_args()

    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
    print(f"Using device: {device} ({torch.get_num_threads()} threads)")

    data_dir = Path(args.data_dir)
    train_loader, valid_loader, classes = build_dataloaders(
        data_dir,
        args.batch_size,
        args.num_workers,
        max_train_per_class=args.max_train_per_class,
        max_valid_per_class=args.max_valid_per_class,
    )
    print(f"Found {len(classes)} classes, {len(train_loader.dataset)} train / {len(valid_loader.dataset)} valid images.")
    sync_class_names(classes, DATA_DIR / "class_names.json")

    model = build_model(num_classes=len(classes), pretrained=True)
    if args.freeze_backbone:
        freeze_backbone(model)
        print("Backbone frozen — training final FC layer only.")
    else:
        print("Full fine-tune — training all layers.")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, max_batches=args.max_train_batches
        )
        val_loss, val_acc = run_epoch(
            model, valid_loader, criterion, optimizer, device, train=False, max_batches=args.max_valid_batches
        )
        elapsed = time.perf_counter() - start

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  Saved new best model to {output_path} (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
