"""Training script for the dragon fruit disease classifier.

Trains a Vision Transformer (ViT-B/16) on the split dataset produced by
``preprocess.py``. Saves the best checkpoint based on validation accuracy,
evaluates on the test set, and exports training history to CSV.

Usage:
    python train.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from preprocess import build_dataloaders
from models.vit import get_vit_model


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path (str): Path to ``config.yaml``.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def save_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list[str],
    save_path: str,
) -> None:
    """Save a confusion matrix heatmap to disk.

    Args:
        y_true (list): Ground-truth class indices.
        y_pred (list): Predicted class indices.
        class_names (list[str]): Ordered class name list.
        save_path (str): Output file path (e.g. ``results/confusion_matrix.png``).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Best Validation Checkpoint")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def train(cfg: dict) -> None:
    """Run the full training, validation, and test evaluation pipeline.

    Trains ViT-B/16 with AdamW and ReduceLROnPlateau scheduling.
    Applies early stopping when validation accuracy does not improve for
    ``patience`` consecutive epochs. After training, reloads the best
    checkpoint and evaluates on the test set.

    Args:
        cfg (dict): Configuration dictionary loaded from ``config.yaml``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # --- Data ---
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        split_dir=cfg["data"]["split_dir"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # --- Model ---
    model = get_vit_model(num_classes).to(device)

    # --- Loss, optimizer, scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    # Reduce LR by 10× when val loss plateaus for 2 consecutive epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    num_epochs      = cfg["train"]["epochs"]
    patience        = cfg["train"]["patience"]
    best_val_acc    = 0.0
    epochs_no_improve = 0
    history: list[dict] = []
    checkpoint_path = os.path.join(results_dir, "best_vit.pth")

    # --- Training loop ---
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        model.train()
        train_loss, train_correct = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss    += loss.item() * inputs.size(0)
            train_correct += (preds == labels).sum().item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc  = train_correct / len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds_list: list = []
        val_labels_list: list = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_preds_list.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc        = accuracy_score(val_labels_list, val_preds_list)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels_list, val_preds_list, average="macro", zero_division=0
        )

        print(
            f"  Train — loss: {epoch_train_loss:.4f} | acc: {epoch_train_acc:.4f}\n"
            f"  Val   — loss: {epoch_val_loss:.4f} | acc: {val_acc:.4f} | "
            f"precision: {precision:.4f} | recall: {recall:.4f} | f1: {f1:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss, "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,     "val_acc": val_acc,
            "val_precision": precision,     "val_recall": recall, "val_f1": f1,
        })

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            save_confusion_matrix(
                val_labels_list, val_preds_list, class_names,
                os.path.join(results_dir, "confusion_matrix_val.png"),
            )
            print(f"  * Best checkpoint saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1

        scheduler.step(epoch_val_loss)

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs).")
            break

    # Save training history
    history_path = os.path.join(results_dir, "history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nTraining history saved: {history_path}")

    # --- Test evaluation ---
    print("\n" + "=" * 50)
    print("  Test set evaluation (best checkpoint)")
    print("=" * 50)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_preds_list: list = []
    test_labels_list: list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds_list.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels_list, test_preds_list)
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(
        test_labels_list, test_preds_list, average="macro", zero_division=0
    )

    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Precision : {t_prec:.4f}")
    print(f"  Recall    : {t_rec:.4f}")
    print(f"  F1-Score  : {t_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(test_labels_list, test_preds_list, target_names=class_names, zero_division=0))

    save_confusion_matrix(
        test_labels_list, test_preds_list, class_names,
        os.path.join(results_dir, "confusion_matrix_test.png"),
    )

    pd.DataFrame([{
        "test_acc": test_acc, "test_precision": t_prec,
        "test_recall": t_rec, "test_f1": t_f1,
    }]).to_csv(os.path.join(results_dir, "test_results.csv"), index=False)


def main() -> None:
    """Entry point for model training."""
    parser = argparse.ArgumentParser(description="Train ViT-B/16 on dragon fruit disease dataset.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
