"""Data preprocessing: split and dataloader construction.

Splits the raw image directory into train / val / test subsets using
``split-folders``, then builds PyTorch ``DataLoader`` objects with
appropriate augmentation for each split.

Usage:
    # Split data (run once before training)
    python preprocess.py --config config.yaml

    # Inspect data statistics and visualize samples
    python preprocess.py --config config.yaml --inspect
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import torch
import yaml
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow loading truncated images without raising an error
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path (str): Path to ``config.yaml``.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed (int): Seed value. Defaults to ``42``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_class_distribution(directory: str) -> dict[str, int]:
    """Count the number of images per class in a directory.

    Args:
        directory (str): Root directory containing one subfolder per class.

    Returns:
        dict[str, int]: Mapping from class name to image count.
    """
    if not os.path.exists(directory):
        return {}
    distribution = {}
    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            distribution[class_name] = count
    return distribution


def print_dataset_stats(raw_dir: str, split_dir: str) -> None:
    """Print image counts for the raw dataset and each split.

    Args:
        raw_dir (str): Original dataset directory.
        split_dir (str): Directory containing train/val/test splits.
    """
    print("\n" + "=" * 50)
    print("  Raw dataset")
    print("=" * 50)
    dist = get_class_distribution(raw_dir)
    if dist:
        print(f"  Total: {sum(dist.values())} images")
        for cls, count in dist.items():
            print(f"  {cls:<15}: {count}")
    else:
        print(f"  No data found at: {raw_dir}")

    for split in ("train", "val", "test"):
        split_path = os.path.join(split_dir, split)
        dist = get_class_distribution(split_path)
        if dist:
            print(f"\n  {split.upper()} split — {sum(dist.values())} images")
            for cls, count in dist.items():
                print(f"  {cls:<15}: {count}")


def split_data(raw_dir: str, split_dir: str, ratio: tuple, seed: int = 42) -> None:
    """Split raw images into train / val / test directories.

    Uses ``split-folders`` to copy images into train/val/test subfolders
    while preserving the class subfolder structure.

    Args:
        raw_dir (str): Source directory with one subfolder per class.
        split_dir (str): Destination directory for the split output.
        ratio (tuple): Split ratios as ``(train, val, test)``.
        seed (int): Random seed for reproducible splits. Defaults to ``42``.
    """
    print(f"\nSplitting data: {raw_dir} → {split_dir}")
    print(f"Ratio — train: {ratio[0]} | val: {ratio[1]} | test: {ratio[2]}")
    splitfolders.ratio(raw_dir, output=split_dir, seed=seed, ratio=ratio, move=False)
    print("Split complete.")


def build_dataloaders(
    split_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build PyTorch DataLoaders for train, val, and test splits.

    Applies random augmentation (rotation, random crop, horizontal flip)
    to the training set and deterministic center-crop to val/test.
    All splits are normalized with ImageNet statistics.

    Args:
        split_dir (str): Directory containing ``train/``, ``val/``, ``test/``
            subfolders, each with one subfolder per class.
        batch_size (int): Number of samples per batch. Defaults to ``32``.
        num_workers (int): DataLoader worker processes. Use ``0`` on Windows.
            Defaults to ``0``.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader, list[str]]:
            - **train_loader** — augmented training DataLoader.
            - **val_loader**   — deterministic validation DataLoader.
            - **test_loader**  — deterministic test DataLoader.
            - **class_names**  — alphabetically sorted class name list.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(split_dir, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(split_dir, "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(os.path.join(split_dir, "test"),  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset.classes


def inspect_data(loader: DataLoader, class_names: list[str]) -> None:
    """Visualize sample images before and after preprocessing.

    Displays a 2×4 grid showing the first four training images in their
    original form (top row) and after augmentation / normalization (bottom row).
    Also prints global pixel statistics for the dataset.

    Args:
        loader (DataLoader): Training DataLoader.
        class_names (list[str]): Ordered list of class names.
    """
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    imagenet_std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # Compute global statistics over the entire dataset
    norm_min, norm_max, norm_sum = float("inf"), float("-inf"), 0.0
    total_pixels = 0

    for imgs, _ in loader:
        norm_min = min(norm_min, imgs.min().item())
        norm_max = max(norm_max, imgs.max().item())
        norm_sum += imgs.sum().item()
        total_pixels += imgs.numel()

    print("\n" + "=" * 50)
    print("  Preprocessing verification")
    print("=" * 50)
    print(f"  Normalized range : [{norm_min:.4f}, {norm_max:.4f}]")
    print(f"  Normalized mean  : {norm_sum / total_pixels:.4f}")
    print("\n  Label mapping:")
    for idx, name in enumerate(class_names):
        print(f"  {idx} → {name}")

    # Side-by-side visualization: original vs augmented
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        img_path, label_idx = loader.dataset.samples[i]

        # Top row: raw image from disk
        with Image.open(img_path).convert("RGB") as raw:
            axes[0, i].imshow(raw)
            axes[0, i].set_title(f"Original: {class_names[label_idx]}")
            axes[0, i].axis("off")

        # Bottom row: after augmentation and normalization
        img_tensor, _ = loader.dataset[i]
        img_np = img_tensor.numpy() * imagenet_std + imagenet_mean
        img_np = np.clip(img_np.transpose(1, 2, 0), 0, 1)
        axes[1, i].imshow(img_np)
        axes[1, i].set_title(f"Augmented (224×224): {class_names[label_idx]}")
        axes[1, i].axis("off")

    plt.tight_layout()
    save_path = "results/preprocess_check.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path)
    print(f"\n  Saved visualization to: {save_path}")
    plt.show()


def main() -> None:
    """Entry point for data splitting and inspection."""
    parser = argparse.ArgumentParser(description="Preprocess dragon fruit disease dataset.")
    parser.add_argument("--config",  default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--inspect", action="store_true",   help="Visualize samples after splitting")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    raw_dir   = cfg["data"]["raw_dir"]
    split_dir = cfg["data"]["split_dir"]
    ratio     = (
        cfg["preprocess"]["train_ratio"],
        cfg["preprocess"]["val_ratio"],
        cfg["preprocess"]["test_ratio"],
    )

    if not os.path.exists(os.path.join(split_dir, "train")):
        split_data(raw_dir, split_dir, ratio, seed=cfg["train"]["seed"])
    else:
        print(f"Split directory already exists: {split_dir} — skipping split.")

    print_dataset_stats(raw_dir, split_dir)

    if args.inspect:
        train_loader, _, _, class_names = build_dataloaders(
            split_dir,
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
        )
        inspect_data(train_loader, class_names)


if __name__ == "__main__":
    main()
