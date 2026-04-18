"""Ablation study: evaluate the contribution of each ViT component.

Trains and evaluates six ViT variants by systematically removing key
architectural components (positional embedding, class token, MLP block)
to quantify their individual and combined contributions.

Variants:
    S1 — Full ViT (baseline)
    S2 — No positional embedding
    S3 — No class token (uses Global Average Pooling instead)
    S4 — No MLP block in Transformer
    S5 — No positional embedding + no class token
    S6 — No positional embedding + no class token + no MLP

Usage:
    python ablation_study.py --config config.yaml --epochs 20
"""

import argparse
import copy
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score

from preprocess import build_dataloaders
from models.ablation.vit_full    import get_vit_full
from models.ablation.vit_no_pos  import get_vit_no_pos
from models.ablation.vit_no_cls  import get_vit_no_cls
from models.ablation.vit_no_mlp  import get_vit_no_mlp
from models.ablation.vit_no_both import get_vit_no_both
from models.ablation.vit_no_all  import get_vit_no_all


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_scenario(
    name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    epochs: int = 20,
) -> dict:
    """Train and evaluate one ablation variant.

    Trains with AdamW for ``epochs`` epochs, tracks the best validation
    accuracy, then evaluates the best checkpoint on the test set.

    Args:
        name (str): Descriptive name for this scenario (used in output table).
        model (nn.Module): Ablation model variant.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        epochs (int): Number of training epochs. Defaults to ``20``.

    Returns:
        dict: Result row with keys ``"Scenario"``, ``"Test ACC"``, ``"Test F1"``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())
    start_time    = time.time()

    print(f"\n[{name}] Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                _, preds = torch.max(model(inputs), 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    # Test evaluation with best weights
    model.load_state_dict(best_weights)
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            _, preds = torch.max(model(inputs), 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1  = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    duration = (time.time() - start_time) / 60

    print(f"[{name}] Done in {duration:.1f} min | best_val_acc={best_val_acc:.4f} | test_acc={test_acc:.4f} | test_f1={test_f1:.4f}")

    return {"Scenario": name, "Test ACC": round(test_acc, 4), "Test F1": round(test_f1, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViT ablation study.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=20,  help="Training epochs per scenario")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        split_dir=cfg["data"]["split_dir"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    scenarios = [
        ("S1: Full ViT (baseline)",               get_vit_full(num_classes)),
        ("S2: No positional embedding",            get_vit_no_pos(num_classes)),
        ("S3: No class token (GAP)",               get_vit_no_cls(num_classes)),
        ("S4: No MLP block",                       get_vit_no_mlp(num_classes)),
        ("S5: No pos embed + no class token",      get_vit_no_both(num_classes)),
        ("S6: No pos embed + no cls + no MLP",     get_vit_no_all(num_classes)),
    ]

    results = []
    for name, model in scenarios:
        res = run_scenario(name, model, train_loader, val_loader, test_loader, epochs=args.epochs)
        results.append(res)

    df = pd.DataFrame(results)
    save_path = os.path.join(results_dir, "ablation_results.csv")
    df.to_csv(save_path, index=False)

    print("\n" + "=" * 60)
    print("  Ablation Study Results")
    print("=" * 60)
    print(df.to_markdown(index=False))
    print(f"\nResults saved: {save_path}")


if __name__ == "__main__":
    main()
