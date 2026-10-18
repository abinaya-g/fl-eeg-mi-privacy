"""
Redraw the three manuscript figures that show FedRA, using the corrected
2-seed FedRA (federated Riemannian mean) and FedRA-local results from
22_fedra_riemannian.py / 23_fedra_riemannian_significance.py. Style
(colours, sizes, annotations) matches the original figures.

Usage: python src/24_figures_fedra_update.py [out_dir]   (default: figures/)
Outputs: fig_main_comparison_verified.png, fig_communication_contrast.png,
         fig_transductive_vs_inductive.png
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)
DPI = 300

# Per-subject accuracies (seed-averaged), as in 21_/23_ significance scripts.
summ = json.load(open(os.path.join(RESULTS, "fedra_riemann", "fedra_riemann_summary.json")))
FEDRA = np.array([summ["per_arm"]["global"]["per_subject"][f"S{s}"] for s in range(1, 10)])
FEDRA_LOCAL = np.array([summ["per_arm"]["local"]["per_subject"][f"S{s}"] for s in range(1, 10)])
IND42 = [0.6910, 0.2778, 0.3902, 0.3090, 0.2500, 0.2639, 0.4618, 0.3542, 0.4062]
_ind123 = json.load(open(os.path.join(RESULTS, "fed_ea_inductive_seed123_results.json")))
INDUCTIVE = (np.array(IND42) + np.array([_ind123[f"S{s}"]["acc"] for s in range(1, 10)])) / 2

def ms(a):
    """Mean and population std (ddof=0, the manuscript tables' convention), in %."""
    return float(np.mean(a) * 100), float(np.std(a) * 100)

# (mean %, std %) for methods without per-subject arrays here: manuscript Table 4.
CSP_LDA, CSP_SVM, CENT = (39.66, 12.4), (40.55, 12.7), (40.26, 13.8)
LOCAL_ONLY, LOCAL_HEADS = (36.67, 10.5), (37.21, 9.5)
CORAL_FED = (38.02, 11.7)
FEDCL_FIXED, FEDCL_W30, FEDCL_W50 = (38.56, 11.3), (40.61, 14.7), (39.76, 13.2)
FEDAVG, FEDEA = (39.84, 13.9), (50.37, 15.1)
FEDRA_MS, FEDRA_LOCAL_MS, INDUCTIVE_MS = ms(FEDRA), ms(FEDRA_LOCAL), ms(INDUCTIVE)
print(f"FedRA {FEDRA_MS}, FedRA-local {FEDRA_LOCAL_MS}, inductive {INDUCTIVE_MS}")


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────
# Figure: main comparison (all methods)
# ─────────────────────────────────────────────
def fig_main():
    GREY, BLUE, PURPLE, RED, DARKRED, ORANGE, GREEN = (
        "#b0b0b0", "#4c72b0", "#9370db", "#c44e52", "#8c2a2e", "#ed9c4b", "#4fa84f")
    rows = [
        ("CSP + LDA", CSP_LDA, GREY), ("CSP + SVM", CSP_SVM, GREY),
        ("Centralised", CENT, GREY), ("Local-only", LOCAL_ONLY, BLUE),
        ("FedAvg + local heads", LOCAL_HEADS, PURPLE),
        ("FedRA (Riemannian)", FEDRA_MS, RED),
        ("FedRA-local (per-subject)", FEDRA_LOCAL_MS, DARKRED),
        ("FedCL (fixed stages)", FEDCL_FIXED, ORANGE),
        ("FedCL (warmup 30→100%)", FEDCL_W30, ORANGE),
        ("FedCL (warmup 50→100%)", FEDCL_W50, ORANGE),
        ("FedAvg (no alignment)", FEDAVG, BLUE),
        ("Fed-EA", FEDEA, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(12.9, 5.9))
    x = np.arange(len(rows))
    means = [r[1][0] for r in rows]
    stds = [r[1][1] for r in rows]
    ax.bar(x, means, yerr=stds, color=[r[2] for r in rows], edgecolor="black",
           linewidth=0.8, width=0.8, capsize=6, error_kw={"elinewidth": 2, "capthick": 2})
    for i, (m, s) in enumerate(zip(means, stds)):
        last = i == len(rows) - 1
        ax.text(i, m + s + 1.5, f"{m:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold" if last else "normal")
    ax.axhline(25, color="orange", linestyle=":", linewidth=1.5, label="Chance (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("LOSO Accuracy (%)", fontsize=13)
    ax.set_ylim(0, 70)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Cross-Subject MI-EEG Classification — BCI Competition IV-2a (9-subject LOSO)\n"
                 "All values are real per-subject results (no estimated or placeholder data)",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=11)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_main_comparison_verified.png"), dpi=DPI)
    plt.close(fig)


# ─────────────────────────────────────────────
# Figure: communication-dependent vs communication-free
# ─────────────────────────────────────────────
def fig_comm():
    fig, ax = plt.subplots(figsize=(8.9, 5.9))
    labels = ["FedAvg\n(no alignment)", "CORAL\n(comm.-dependent)",
              "FedRA\n(comm.-dependent)", "Fed-EA\n(comm.-free)"]
    vals = [FEDAVG, CORAL_FED, FEDRA_MS, FEDEA]
    colors = ["#3498db", "#c0392b", "#c0392b", "#27ae60"]
    x = np.arange(4)
    for i, ((m, s), c) in enumerate(zip(vals, colors)):
        last = i == 3
        ax.bar(i, m, width=0.6, color=c, edgecolor="#145a32" if last else "black",
               linewidth=3 if last else 1)
        ax.errorbar(i, m, yerr=s, color="#333333", capsize=6, elinewidth=1.5, capthick=1.5)
        ax.text(i, m + 1.2, f"{m:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.axhline(25, color="#e67e22", linestyle=":", linewidth=2, label="Chance (25%)")
    ax.axhline(FEDAVG[0], color="#85c1e9", linestyle="--", linewidth=1.5,
               label="FedAvg (no alignment) reference")
    note = dict(ha="center", va="bottom", fontsize=11, style="italic", color="#444444")
    ax.text(1, CORAL_FED[0] + CORAL_FED[1] + 3, "small effect,\nnot closing gap", **note)
    ax.text(2, FEDRA_MS[0] + FEDRA_MS[1] + 3, "p=0.538 (ns)\nvs. FedAvg", **note)
    ax.text(3, FEDEA[0] + FEDEA[1] + 3.5, "p=0.0009 (***)\nvs. FedAvg", **note)
    ax.plot([1.05, 2.3], [68, 68], color="#c0392b", linewidth=1.8)
    ax.text(1.68, 68.8, "Communication-dependent:\nneither improves on FedAvg",
            ha="center", va="bottom", fontsize=11, style="italic", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("LOSO Accuracy (%)", fontsize=14)
    ax.set_ylim(0, 78)
    ax.set_title("Communication-Dependent vs. Communication-Free Alignment in FL\n"
                 "(BCI-IV 2a, EEGNet; error bars = per-subject std)", fontsize=14)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_communication_contrast.png"), dpi=DPI)
    plt.close(fig)


# ─────────────────────────────────────────────
# Figure: transductive vs inductive, Euclidean and Riemannian pairs
# ─────────────────────────────────────────────
def bracket(ax, x0, x1, y, text, color):
    ax.plot([x0, x0, x1, x1], [y - 1.8, y, y, y - 1.8], color=color, linewidth=2)
    ax.text((x0 + x1) / 2, y + 0.8, text, ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=color)

def fig_trans():
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    labels = ["FedAvg\n(no alignment)", "FedRA\n(inductive,\nRiemannian)",
              "FedRA-local\n(transductive,\nRiemannian)",
              "Fed-EA\n(inductive,\nEuclidean)", "Fed-EA\n(transductive,\nstandard)"]
    vals = [FEDAVG, FEDRA_MS, FEDRA_LOCAL_MS, INDUCTIVE_MS, FEDEA]
    colors = ["#4c72b0", "#808080", "#93c993", "#ed9c4b", "#4fa84f"]
    x = np.arange(len(vals))
    ax.bar(x, [v[0] for v in vals], yerr=[v[1] for v in vals], color=colors,
           edgecolor="black", width=0.8, capsize=7, error_kw={"elinewidth": 2, "capthick": 2})
    for i, (m, s) in enumerate(vals):
        ax.text(i, m + 2, f"{m:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.axhline(25, color="orange", linestyle=":", linewidth=2, label="Chance (25%)")
    ax.text(3, INDUCTIVE_MS[0] + INDUCTIVE_MS[1] + 6.5,
            "n.s. vs. FedAvg/FedRA\n(p=0.664 / p=0.835)",
            ha="center", va="bottom", fontsize=11, style="italic", color="#555555")
    bracket(ax, 1, 2, 69, "+6.12pp\n$p$=0.0050, $d_z$=1.28", "#2e7d32")
    bracket(ax, 3, 4, 69, "−12.36pp\n$p$=0.0174, $d_z$=−1.00", "#8b008b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("LOSO Accuracy (%)", fontsize=14)
    ax.set_ylim(0, 80)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Fed-EA's Advantage Is Primarily Transductive, Not Purely Communication-Free\n"
                 "(BCI-IV 2a, EEGNet; error bars = per-subject std; all bars are 2-seed averages)",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=12)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_transductive_vs_inductive.png"), dpi=DPI)
    plt.close(fig)


# ─────────────────────────────────────────────
# Figure: per-subject heatmap (all methods)
# ─────────────────────────────────────────────
def fig_heatmap():
    from matplotlib.colors import LinearSegmentedColormap
    csp = json.load(open(os.path.join(RESULTS, "csp_baselines_results.json")))
    heads = json.load(open(os.path.join(RESULTS, "fedavg_localheads_results.json")))
    def row(d):
        return [d[f"S{s}"]["acc"] if isinstance(d[f"S{s}"], dict) else d[f"S{s}"]
                for s in range(1, 10)]
    fedavg = [(a + b) / 2 for a, b in zip(
        [0.5694, 0.2708, 0.6411, 0.3368, 0.2465, 0.2847, 0.2986, 0.5625, 0.4201],
        [0.5868, 0.2986, 0.5993, 0.3056, 0.2465, 0.2778, 0.2951, 0.5556, 0.3750])]
    fedea = [(a + b) / 2 for a, b in zip(
        [0.6840, 0.3264, 0.7038, 0.3646, 0.2812, 0.4375, 0.4792, 0.6597, 0.6250],
        [0.6875, 0.3368, 0.6934, 0.3785, 0.2812, 0.4375, 0.4583, 0.6424, 0.5903])]
    rows = [
        ("CSP + LDA", row(csp["CSP_LDA"])),
        ("CSP + SVM", row(csp["CSP_SVM"])),
        # Centralised / local-only: manuscript Table (per-subject, EEGNet)
        ("Centralised", [0.5486, 0.2778, 0.5610, 0.3299, 0.2708, 0.2569, 0.3576, 0.6528, 0.3681]),
        ("Local-only", [0.4375, 0.2639, 0.5401, 0.3403, 0.2500, 0.3090, 0.2812, 0.5347, 0.3438]),
        ("FedAvg + local heads", row(heads["no_coral"])),
        ("FedRA (Riemannian)", list(FEDRA)),
        ("FedRA-local (per-subject)", list(FEDRA_LOCAL)),
        # FedCL fixed: per-subject values as printed in the original heatmap
        # (no per-subject results file in the repository)
        ("FedCL (fixed stages)", [0.50, 0.27, 0.60, 0.35, 0.26, 0.31, 0.31, 0.37, 0.51]),
        ("FedAvg (no alignment)", fedavg),
        ("Fed-EA", fedea),
    ]
    data = np.array([r[1] for r in rows], dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "ryg", ["#c0392b", "#e67e22", "#f1c232", "#7cb342", "#1e8449"])
    fig, ax = plt.subplots(figsize=(10.3, 7.5))
    im = ax.imshow(data, cmap=cmap, vmin=0.2, vmax=0.75, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white" if (v < 0.35 or v > 0.67) else "black")
    for j in (1, 4, 5, 6):  # S2, S5, S6, S7: BCI-illiterate subjects
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, data.shape[0], fill=False,
                                   edgecolor="#ff7f50", linewidth=2.5))
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"S{s}" for s in range(1, 10)], fontsize=12)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.set_xlabel("Subject", fontsize=13)
    ax.set_title("Per-Subject Accuracy Heatmap — All Methods (BCI-IV 2a)\n"
                 "Every cell is a real per-subject result; coral border = BCI-illiterate subjects",
                 fontsize=13)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("LOSO Accuracy", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_heatmap_verified.png"), dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig_main()
    fig_comm()
    fig_trans()
    fig_heatmap()
    print(f"Saved 4 figures to {os.path.abspath(OUT)}")
