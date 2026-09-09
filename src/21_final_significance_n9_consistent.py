"""
FINAL SIGNIFICANCE ANALYSIS — consistent n=9 convention
==========================================================================
This is the analysis that produced the exact numbers reported in the
manuscript for: (1) the main inductive-vs-transductive Fed-EA ablation
(Section "Why Transductive Euclidean Alignment...", Discussion), and
(2) the target-calibration-size sweep's ten-comparison Holm-corrected
family (Results, Table "Paired significance testing for the
target-calibration-size sweep").

IMPORTANT METHODOLOGICAL NOTE: an earlier draft of this analysis used
n=18 (treating each subject-seed pair as an independent sample) for
some of these comparisons. This was inconsistent with the rest of the
paper's convention, established in the main Fed-EA-vs-baselines
results, of averaging the two seeds per subject FIRST and then running
paired tests at n=9 (explicitly, the paper notes the Wilcoxon p-value
for the main Fed-EA-vs-FedAvg comparison "attains the smallest value
possible at n=9"). This script uses n=9 throughout, matching that
established convention, and is the version whose output is actually
reported in the manuscript.

Two reporting conventions are used, matching how the rest of the paper
already treats different kinds of comparisons:
  (a) STANDALONE, uncorrected -- for the paper's primary, single
      targeted ablation (inductive vs. transductive Fed-EA), matching
      how e.g. "Fed-EA exceeds FedRA by 9.65 points (p=0.0011)" is
      reported elsewhere without folding it into a larger correction
      family.
  (b) The CALIBRATION-SWEEP FAMILY (10 comparisons), Holm-Bonferroni
      corrected internally, matching how the main Fed-EA-vs-baselines
      table and the FedRA/CORAL tables are each corrected within their
      own family.

The same underlying comparison (inductive vs. calibrated/transductive
Fed-EA) can therefore be significant under framing (a) and not survive
correction under framing (b) -- this is not a contradiction, it is a
direct consequence of which comparison family a test is corrected
against. The manuscript reports both framings explicitly rather than
selecting whichever is more favourable.

Inputs (edit these to point at your actual result files/values):
  - Per-subject accuracy for: Inductive (seed 42 + seed 123),
    FedAvg (seed 42 + seed 123), FedRA (single seed),
    standard transductive Fed-EA (seed 42 + seed 123).
  - fed_ea_calibration_sweep_results.json (produced by
    15_fedea_calibration_sweep.py / 20_calibration_sweep_significance.py's
    upstream data).
"""

import json
import os
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ─────────────────────────────────────────────
# Per-subject accuracy, both seeds where applicable.
# Sourced from: cspfedavg.ipynb Cells 5,6 (FedAvg), 7,8 (Fed-EA
# transductive), 10 (FedRA, single seed), 14 (Fed-EA inductive, seed
# 42) + fed_ea_inductive_seed123_results.json (seed 123).
# ─────────────────────────────────────────────
FEDAVG_42  = {1:0.5694,2:0.2708,3:0.6411,4:0.3368,5:0.2465,6:0.2847,7:0.2986,8:0.5625,9:0.4201}
FEDAVG_123 = {1:0.5868,2:0.2986,3:0.5993,4:0.3056,5:0.2465,6:0.2778,7:0.2951,8:0.5556,9:0.3750}

FEDEA_TRANS_42  = {1:0.6840,2:0.3264,3:0.7038,4:0.3646,5:0.2812,6:0.4375,7:0.4792,8:0.6597,9:0.6250}
FEDEA_TRANS_123 = {1:0.6875,2:0.3368,3:0.6934,4:0.3785,5:0.2812,6:0.4375,7:0.4583,8:0.6424,9:0.5903}

FEDRA = {1:0.6181,2:0.3160,3:0.5331,4:0.3299,5:0.2535,6:0.2847,7:0.3438,8:0.5000,9:0.4861}  # single-seed

INDUCTIVE_42 = {1:0.6910,2:0.2778,3:0.3902,4:0.3090,5:0.2500,6:0.2639,7:0.4618,8:0.3542,9:0.4062}
with open(os.path.join(RESULTS_DIR, "fed_ea_inductive_seed123_results.json")) as f:
    _raw = json.load(f)
INDUCTIVE_123 = {int(k[1:]): v["acc"] for k, v in _raw.items()}

def seed_avg(d1, d2):
    return {s: (d1[s] + d2[s]) / 2 for s in d1}

FEDAVG = seed_avg(FEDAVG_42, FEDAVG_123)
FEDEA_TRANSDUCTIVE = seed_avg(FEDEA_TRANS_42, FEDEA_TRANS_123)
INDUCTIVE = seed_avg(INDUCTIVE_42, INDUCTIVE_123)

# Calibration sweep: average both seeds AND all within-fraction splits
# together per subject, giving one n=9 value per fraction.
with open(os.path.join(RESULTS_DIR, "fed_ea_calibration_sweep_results.json")) as f:
    _calib_raw = json.load(f)
_per_run = _calib_raw["per_run"]
_tmp = {"0.25": {}, "0.50": {}, "0.75": {}, "1.00": {}}
for r in _per_run.values():
    fk = f"{r['fraction']:.2f}"
    _tmp[fk].setdefault(r["subject"], []).append(r["acc"])
CALIB = {fk: {s: float(np.mean(v)) for s, v in d.items()} for fk, d in _tmp.items()}

# ─────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────
def cohens_dz(diffs):
    d = np.asarray(diffs, dtype=float)
    sd = d.std(ddof=1)
    return 0.0 if sd == 0 else float(d.mean() / sd)

def bootstrap_ci(diffs, n_boot=10000, rng=np.random.RandomState(42)):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    boots = np.array([d[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def paired(name_a, da, name_b, db):
    keys = sorted(set(da) & set(db))
    a = np.array([da[k] for k in keys])
    b = np.array([db[k] for k in keys])
    diffs = a - b
    t, p_t = stats.ttest_rel(a, b)
    w, p_w = stats.wilcoxon(a, b)
    dz = cohens_dz(diffs)
    lo, hi = bootstrap_ci(diffs)
    return {"comparison": f"{name_a} vs {name_b}", "n": len(keys),
            "mean_diff_pct": round(float(diffs.mean()) * 100, 3),
            "t_p": float(p_t), "wilcoxon_p": float(p_w), "dz": round(dz, 3),
            "bootstrap_95ci_pct": [round(lo * 100, 2), round(hi * 100, 2)]}

def holm(results, key, out):
    m = len(results)
    order = sorted(range(m), key=lambda i: results[i][key])
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * results[idx][key]
        running = max(running, adj)
        results[idx][out] = round(min(running, 1.0), 5)

def main():
    print("=" * 80)
    print("  (a) STANDALONE comparisons — reported without correction,")
    print("      matching this paper's treatment of targeted ablations")
    print("      (e.g. Fed-EA vs. FedRA reported the same way)")
    print("=" * 80)
    standalone = [
        paired("Inductive", INDUCTIVE, "Transductive", FEDEA_TRANSDUCTIVE),
        paired("Inductive", INDUCTIVE, "FedAvg", FEDAVG),
        paired("Inductive", INDUCTIVE, "FedRA", FEDRA),
    ]
    for r in standalone:
        print(f"  {r['comparison']:<28} n={r['n']} diff={r['mean_diff_pct']:>7.2f}pp "
              f"t_p={r['t_p']:.4f} w_p={r['wilcoxon_p']:.4f} dz={r['dz']:>6.2f} "
              f"95%CI={r['bootstrap_95ci_pct']}")

    print(f"\n{'=' * 80}")
    print("  (b) CALIBRATION-SWEEP FAMILY — Holm-Bonferroni corrected")
    print("      across all 10 comparisons listed, n=9 throughout")
    print("=" * 80)
    family = [
        paired("Inductive", INDUCTIVE, "FedAvg", FEDAVG),
        paired("Inductive", INDUCTIVE, "frac=0.25", CALIB["0.25"]),
        paired("Inductive", INDUCTIVE, "frac=1.00", CALIB["1.00"]),
        paired("frac=0.25", CALIB["0.25"], "frac=0.50", CALIB["0.50"]),
        paired("frac=0.25", CALIB["0.25"], "frac=0.75", CALIB["0.75"]),
        paired("frac=0.25", CALIB["0.25"], "frac=1.00", CALIB["1.00"]),
        paired("frac=0.50", CALIB["0.50"], "frac=0.75", CALIB["0.75"]),
        paired("frac=0.50", CALIB["0.50"], "frac=1.00", CALIB["1.00"]),
        paired("frac=0.75", CALIB["0.75"], "frac=1.00", CALIB["1.00"]),
        paired("frac=0.25", CALIB["0.25"], "FedAvg", FEDAVG),
    ]
    holm(family, "t_p", "holm_t")
    holm(family, "wilcoxon_p", "holm_w")
    for r in family:
        print(f"  {r['comparison']:<28} n={r['n']} diff={r['mean_diff_pct']:>7.2f}pp "
              f"t_p={r['t_p']:.4f} w_p={r['wilcoxon_p']:.4f} dz={r['dz']:>6.2f} "
              f"Holm_t={r['holm_t']:.4f} Holm_w={r['holm_w']:.4f}")

    with open("final_significance_n9_consistent.json", "w") as f:
        json.dump({"standalone": standalone, "calibration_family": family}, f, indent=2)
    print("\nSaved: final_significance_n9_consistent.json")

if __name__ == "__main__":
    main()
