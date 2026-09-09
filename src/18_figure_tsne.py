"""
Figure: t-SNE Feature Visualisation (FedAvg vs Fed-EA, S2 vs S3)
====================================================================
Loads tsne_features.json (produced by tsne_feature_extraction.py on
Kaggle) and produces a 2x2 grid: subject (S3 best / S2 BCI-illiterate)
x condition (FedAvg no-alignment / Fed-EA), showing how well each
condition's feature space separates the 4 motor-imagery classes.

Run locally -- matplotlib + scikit-learn only, no GPU needed.
Requires tsne_features.json downloaded from Kaggle into the same
directory (or update FEATURES_PATH below).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

FEATURES_PATH = "./tsne_features.json"
OUTPUT_PATH = "./fig_tsne.png"
DPI = 300

CLASS_NAMES = ["Left Hand", "Right Hand", "Both Feet", "Tongue"]
CLASS_COLORS = ["#1abc9c", "#e67e22", "#8e44ad", "#95a5a6"]


def main():
    with open(FEATURES_PATH) as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    panels = [
        ("S3_fedavg", "S3 (best subject) — FedAvg", axes[0, 0]),
        ("S3_fedea",  "S3 (best subject) — Fed-EA", axes[0, 1]),
        ("S2_fedavg", "S2 (BCI-illiterate) — FedAvg", axes[1, 0]),
        ("S2_fedea",  "S2 (BCI-illiterate) — Fed-EA", axes[1, 1]),
    ]

    for key, title, ax in panels:
        if key not in data:
            ax.text(0.5, 0.5, f"Missing: {key}", ha='center', va='center')
            ax.axis('off')
            continue

        features = np.array(data[key]["features"])
        labels = np.array(data[key]["labels"])
        test_acc = data[key]["test_acc"]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
        proj = tsne.fit_transform(features)

        for c in range(4):
            mask = labels == c
            ax.scatter(proj[mask, 0], proj[mask, 1], c=CLASS_COLORS[c],
                       label=CLASS_NAMES[c], s=18, alpha=0.75, edgecolors='none')

        ax.set_title(f"{title}\n(test acc: {test_acc*100:.1f}%)", fontsize=10)
        ax.set_xlabel("t-SNE dim 1", fontsize=8)
        ax.set_ylabel("t-SNE dim 2", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0, 0].legend(loc='upper right', fontsize=7, frameon=False)

    fig.suptitle("t-SNE Feature Visualisation: FedAvg vs. Fed-EA\n"
                  "(BCI-IV 2a, EEGNet, held-out target subject's E-session features)",
                  fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()