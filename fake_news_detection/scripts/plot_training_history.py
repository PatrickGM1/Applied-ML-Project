"""Plot train vs validation loss to show overfitting analysis.

Reads per-epoch losses from artifacts/final/training_history.json
and produces a 2x2 grid of plots (one per experiment).

Run:
    python fake_news_detection/scripts/plot_training_history.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
HISTORY_PATH = PROJECT_DIR / "artifacts" / "final" / "training_history.json"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "final"


def main():
    with open(HISTORY_PATH, encoding="utf-8") as fh:
        history = json.load(fh)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("BERT Training: Train vs Validation Loss (Overfitting Analysis)", fontsize=14, fontweight="bold")

    plot_order = [
        ("binary_bert_text_only", "Binary — Text Only"),
        ("binary_bert_metadata", "Binary — Text + Metadata"),
        ("multiclass_bert_text_only", "Multiclass — Text Only"),
        ("multiclass_bert_metadata", "Multiclass — Text + Metadata"),
    ]

    for ax, (key, title) in zip(axes.flat, plot_order):
        data = history[key]
        epochs = data["epochs"]
        train_loss = data["avg_train_loss"]
        val_loss = data["val_loss"]
        best_epoch = data["best_epoch"]

        ax.plot(epochs, train_loss, "o-", label="Train loss", color="#7c6af7", linewidth=2)
        ax.plot(epochs, val_loss, "s-", label="Val loss", color="#f87171", linewidth=2)

        ax.axvline(x=best_epoch, color="#4ade80", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")

        gap = [t - v for t, v in zip(train_loss, val_loss)]
        for i, (e, g) in enumerate(zip(epochs, gap)):
            if i >= 2 and g < -0.05:
                ax.annotate(
                    f"gap={g:.3f}",
                    xy=(e, val_loss[i]),
                    xytext=(e + 0.15, val_loss[i] + 0.02),
                    fontsize=7, color="#888",
                )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(epochs)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "training_loss_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
