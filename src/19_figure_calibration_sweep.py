"""
Figure: Fed-EA Accuracy vs. Target-Calibration-Size (BCI-IV 2a, EEGNet)
==========================================================================
Purpose: visualise the dose-response relationship between how much of
the held-out target subject's unlabelled E-session data is used to
compute the Fed-EA whitening matrix, and resulting LOSO accuracy.
Spans the two endpoints already in the manuscript (fully inductive,
0% target data; standard transductive, 100% target data) plus the new
25/50/75% calibration-sweep results.

Data source: by default, this script loads
fed_ea_calibration_sweep_results.json (produced by
fed_ea_calibration_sweep_v2.py) and recomputes the same per-fraction
summary as that script's print_summary(). If that file is not found
(e.g. running this locally rather than on Kaggle), it falls back to
the exact summary values already reported by that script's own run,
hardcoded below with their source noted -- this fallback path is only
for convenience plotting and should be replaced by loading the real
JSON before this figure is treated as final.

The frac=0.00 (fully inductive) and FedAvg-no-alignment reference
values are taken from the already-verified fed_ea_inductive_results.json
and fedavg_global_reverify results (39.84%/0.139 std, matching
Table 1 of the manuscript) -- NOT recomputed here.

Output: fig_calibration_sweep.png (300 DPI), ready to reference in the
manuscript alongside the existing fig_coral_lambda_verified.png etc.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
RESULTS_JSON = os.path.join(SAVE_DIR, "fed_ea_calibration_sweep_results.json")
OUT_PNG = os.path.join(SAVE_DIR, "fig_calibration_sweep.png")

# ─────────────────────────────────────────────
# Reference points already verified elsewhere in the project — NOT
# computed by this script.
# ─────────────────────────────────────────────
INDUCTIVE_MEAN = 0.3802   # 2-seed average (42, 123), fed_ea_inductive results
INDUCTIVE_STD  = 0.1336   # std across all 18 raw subject-seed cells (pooled, not pre-averaged),
                           # for visual consistency with the other four points' error bars
FEDAVG_MEAN    = 0.3984   # Table 1 / fedavg_global_reverify (2-seed avg)
FEDAVG_STD     = 0.139    # Table 1, "FedAvg (no alignment)" row

# ─────────────────────────────────────────────
# Fallback summary — EXACT values from the verified calibration-sweep
# run (fed_ea_calibration_sweep_v2.py output), used only if the JSON
# checkpoint is not present in this environment. Replace by loading
# the real file whenever possible.
# ─────────────────────────────────────────────
FALLBACK_SUMMARY = {
    "0.25": {"mean_acc": 0.5067444444444446, "std_acc": 0.15437849152946653, "n_subject_seed_cells": 18},
    "0.50": {"mean_acc": 0.4996177777777777, "std_acc": 0.14762910641783483, "n_subject_seed_cells": 18},
    "0.75": {"mean_acc": 0.4973755555555555, "std_acc": 0.13966818639675257, "n_subject_seed_cells": 18},
    "1.00": {"mean_acc": 0.5041222222222221, "std_acc": 0.15187801594830985, "n_subject_seed_cells": 18},
}

def load_summary():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            data = json.load(f)
        if "summary" in data and data["summary"]:
            print(f"Loaded real summary from {RESULTS_JSON}")
            return data["summary"]
        # summary block missing/empty -- recompute from per_run entries
        per_run = data.get("per_run", {})
        summary = {}
        for frac_str in ["0.25", "0.50", "0.75", "1.00"]:
            frac = float(frac_str)
            per_subject_seed = {}
            for run in per_run.values():
                if abs(run["fraction"] - frac) > 1e-6:
                    continue
                k = (run["subject"], run["seed"])
                per_subject_seed.setdefault(k, []).append(run["acc"])
            if not per_subject_seed:
                continue
            means = [float(np.mean(v)) for v in per_subject_seed.values()]
            summary[frac_str] = {"mean_acc": float(np.mean(means)),
                                  "std_acc": float(np.std(means)),
                                  "n_subject_seed_cells": len(means)}
        if summary:
            print(f"Recomputed summary from per_run entries in {RESULTS_JSON}")
            return summary
    print("WARNING: results JSON not found or unusable -- using the "
          "hardcoded fallback summary values. Verify these match your "
          "actual run before treating this figure as final.")
    return FALLBACK_SUMMARY

def main():
    summary = load_summary()

    fracs = [0.00, 0.25, 0.50, 0.75, 1.00]
    means = [INDUCTIVE_MEAN * 100]
    stds  = [INDUCTIVE_STD * 100]
    for f in [0.25, 0.50, 0.75, 1.00]:
        key = f"{f:.2f}"
        means.append(summary[key]["mean_acc"] * 100)
        stds.append(summary[key]["std_acc"] * 100)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(fracs, means, yerr=stds, fmt='o-', color='#1f77b4',
                ecolor='#1f77b4', elinewidth=1.2, capsize=4, markersize=7,
                linewidth=2, label='Fed-EA (target-calibration sweep)',
                zorder=3)

    ax.axhline(FEDAVG_MEAN * 100, color='#888888', linestyle='--',
               linewidth=1.3, zorder=1,
               label=f'FedAvg, no alignment ({FEDAVG_MEAN*100:.2f}%)')
    ax.axhline(25.0, color='#cccccc', linestyle=':', linewidth=1,
               zorder=1, label='Chance level (25.00%)')

    ax.annotate('Fully inductive\n(source-derived proxy)',
                xy=(0.00, means[0]), xytext=(0.06, means[0] - 11),
                fontsize=8.5, ha='left',
                arrowprops=dict(arrowstyle='->', color='#444444', lw=0.8))
    ax.annotate('Standard transductive\n(all target data)',
                xy=(1.00, means[-1]), xytext=(0.70, means[-1] + 7),
                fontsize=8.5, ha='left',
                arrowprops=dict(arrowstyle='->', color='#444444', lw=0.8))

    # Significance summary box (from calibration_sweep_significance results,
    # n=18 subject-seed cells, Holm-Bonferroni corrected across 10 comparisons)
    ax.text(0.02, 0.97,
            "All pairwise fraction comparisons (25–100%): n.s.\n"
            "(Holm-adjusted $p$ > 0.34, $n$=9)\n"
            "frac=0.25 vs. FedAvg: Holm-adjusted $p$ = 0.015",
            transform=ax.transAxes, fontsize=7.8, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                      edgecolor='#cccccc', linewidth=0.8))

    ax.set_xlabel('Fraction of target subject\'s unlabelled E-session\n'
                   'trials used to compute EA whitening matrix', fontsize=11)
    ax.set_ylabel('LOSO Accuracy (%)', fontsize=11)
    ax.set_title('Fed-EA Accuracy vs. Target-Calibration-Data Fraction\n'
                  '(BCI Competition IV-2a, EEGNet, 2-seed average)',
                  fontsize=11.5)
    ax.set_xticks(fracs)
    ax.set_xticklabels(['0.00\n(inductive)', '0.25', '0.50', '0.75',
                         '1.00\n(transductive)'])
    ax.set_ylim(15, 70)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {OUT_PNG}")

    print("\nValues plotted:")
    for f, m, s in zip(fracs, means, stds):
        print(f"  frac={f:.2f}  mean={m:.2f}%  std={s:.2f}%")

if __name__ == "__main__":
    main()
