"""Training history visualization and model comparison utilities.

Reads CSV files produced by ``train.py`` and ``ablation_study.py`` and
generates publication-ready plots comparing learning curves and final metrics.

Usage:
    python visualize.py --config config.yaml
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def plot_learning_curves(history_path: str, save_dir: str) -> None:
    """Plot training and validation loss / accuracy curves for one model.

    Args:
        history_path (str): Path to the ``history.csv`` file produced by
            ``train.py``, containing per-epoch ``train_loss``, ``val_loss``,
            ``train_acc``, and ``val_acc`` columns.
        save_dir (str): Directory to save the output figure.
    """
    if not os.path.exists(history_path):
        print(f"[SKIP] History file not found: {history_path}")
        return

    df = pd.read_csv(history_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="steelblue", marker="o", markersize=4)
    ax1.plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="tomato",    marker="x", markersize=4)
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy subplot
    ax2.plot(df["epoch"], df["train_acc"], label="Train Acc", color="steelblue", marker="o", markersize=4)
    ax2.plot(df["epoch"], df["val_acc"],   label="Val Acc",   color="tomato",    marker="x", markersize=4)
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Learning curves saved: {save_path}")


def plot_ablation_comparison(ablation_path: str, save_dir: str) -> None:
    """Plot a grouped bar chart comparing ablation study results.

    Args:
        ablation_path (str): Path to ``ablation_results.csv`` produced by
            ``ablation_study.py``, with columns ``Scenario``, ``Test ACC``,
            and ``Test F1``.
        save_dir (str): Directory to save the output figure.
    """
    if not os.path.exists(ablation_path):
        print(f"[SKIP] Ablation results not found: {ablation_path}")
        return

    df = pd.read_csv(ablation_path)
    df_melted = df.melt(id_vars="Scenario", value_vars=["Test ACC", "Test F1"],
                        var_name="Metric", value_name="Value")

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_melted, x="Scenario", y="Value", hue="Metric", palette="muted")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.4f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontsize=9,
        )

    plt.title("Ablation Study — ViT Component Removal")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "ablation_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Ablation comparison saved: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training results.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    plot_learning_curves(
        history_path=os.path.join(results_dir, "history.csv"),
        save_dir=results_dir,
    )
    plot_ablation_comparison(
        ablation_path=os.path.join(results_dir, "ablation_results.csv"),
        save_dir=results_dir,
    )


if __name__ == "__main__":
    main()
