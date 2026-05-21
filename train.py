"""Train EfficientNet-B0 on the Kaggle Brain Tumor MRI dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.inference import CLASS_NAMES
from src.model import create_model

# Folder names (lowercase) -> class index; order matches CLASS_NAMES in inference.py
FOLDER_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "no_tumor": 2,
    "pituitary": 3,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _collect_samples(training_dir: Path) -> list[tuple[Path, int]]:
    """Walk Training/ class folders and return (path, label) pairs."""
    if not training_dir.is_dir():
        raise FileNotFoundError(
            f"Training folder not found: {training_dir}\n"
            "Expected layout: data/Training/{glioma,meningioma,notumor,pituitary}/"
        )

    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(training_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        key = class_dir.name.lower().replace(" ", "_")
        if key not in FOLDER_TO_IDX:
            print(f"Skipping unknown folder: {class_dir.name}")
            continue
        label = FOLDER_TO_IDX[key]
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))

    if not samples:
        raise RuntimeError(f"No images found under {training_dir}")
    return samples


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, val_tf


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MIND brain tumor classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset root")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="weights",
        help="Directory for best_model.pth",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    data_dir = (project_root / args.data_dir).resolve()
    training_dir = data_dir / "Training"
    weights_dir = project_root / args.weights_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {CLASS_NAMES}")

    samples = _collect_samples(training_dir)
    labels = [label for _, label in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=args.val_split,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_tf, val_tf = build_transforms()
    train_loader = DataLoader(
        BrainTumorDataset(train_samples, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        BrainTumorDataset(val_samples, val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_model(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best val acc {val_acc:.4f}, checkpoint saved")

    print(f"Saved to {best_path}")


if __name__ == "__main__":
    main()
