"""
SECOND-DATASET DECOMPOSITION — SUMMARY AND PAIRED SIGNIFICANCE
====================================================================
Reads the per-fold JSONs from 25_decomposition_multidataset.py.

BNCI 2014-002 (n=14): seed-averaged per-subject accuracy per arm, and the
eight paired comparisons of the decomposition (paired t, Wilcoxon,
Cohen's d_z, bootstrap 95% CI), Holm-corrected within this family.

BCI-IV 2a reproduction check: FedAvg and Fed-EA re-run with the unified
script (seed 42) against the published seed-42 per-subject results
(21_final_significance_n9_consistent.py).

Usage: python src/26_decomposition_significance.py [results/decomposition]
Output: <dir>/decomposition_summary.json and a printed table.
"""

import glob, json, os, sys
import numpy as np
from scipy import stats

LABELS = {"fedavg": "FedAvg", "ea": "Fed-EA (transductive)",
          "ea_inductive": "Fed-EA (inductive)", "ra_global": "FedRA (shared, inductive)",
          "ra_local": "FedRA-local (transductive)"}
COMPARISONS = [("ea", "fedavg"), ("ea_inductive", "fedavg"), ("ea", "ea_inductive"),
               ("ra_global", "fedavg"), ("ra_local", "fedavg"), ("ra_local", "ra_global"),
               ("ea", "ra_local"), ("ea", "ra_global")]
PUBLISHED_2A_SEED42 = {
    "fedavg": [0.5694, 0.2708, 0.6411, 0.3368, 0.2465, 0.2847, 0.2986, 0.5625, 0.4201],
    "ea":     [0.6840, 0.3264, 0.7038, 0.3646, 0.2812, 0.4375, 0.4792, 0.6597, 0.6250],
}


def cohens_dz(d):
    d = np.asarray(d, float)
    sd = d.std(ddof=1)
    return 0.0 if sd == 0 else float(d.mean() / sd)

def bootstrap_ci(d, n_boot=10000, rng=np.random.RandomState(42)):
    d = np.asarray(d, float)
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    _, p_t = stats.ttest_rel(a, b)
    _, p_w = stats.wilcoxon(a, b)
    lo, hi = bootstrap_ci(a - b)
    return {"n": len(a), "mean_a_pct": round(a.mean() * 100, 2), "mean_b_pct": round(b.mean() * 100, 2),
            "mean_diff_pct": round((a - b).mean() * 100, 2), "t_p": float(p_t), "wilcoxon_p": float(p_w),
            "dz": round(cohens_dz(a - b), 3), "ci95_pct": [round(lo * 100, 2), round(hi * 100, 2)]}

def holm(results, key, out):
    m = len(results)
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: results[i][key])):
        running = max(running, (m - rank) * results[i][key])
        results[i][out] = min(running, 1.0)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "results", "decomposition")
    runs = {}
    for p in glob.glob(os.path.join(run_dir, "*_seed*_S*.json")):
        r = json.load(open(p))
        runs.setdefault(r["dataset"], {}).setdefault(r["arm"], {}).setdefault(
            r["seed"], {})[int(r["target"][1:])] = r
    summary = {}

    # ── BNCI 2014-002 ──
    d = runs.get("bnci2014002", {})
    n_subj = 14
    per_arm = {}
    for arm, seeds in d.items():
        complete = {sd: v for sd, v in seeds.items() if len(v) == n_subj}
        for sd, v in seeds.items():
            if len(v) != n_subj:
                print(f"  WARNING: bnci2014002 {arm} seed {sd}: {len(v)}/{n_subj} folds — excluded")
        if not complete:
            continue
        acc = np.mean([[complete[sd][s]["acc"] for s in range(1, n_subj + 1)] for sd in complete], axis=0)
        f1 = np.mean([[complete[sd][s]["f1"] for s in range(1, n_subj + 1)] for sd in complete], axis=0)
        per_arm[arm] = {"label": LABELS[arm], "seeds": sorted(complete),
                        "per_seed_mean_pct": {sd: round(np.mean([v[s]["acc"] for s in v]) * 100, 2)
                                              for sd, v in complete.items()},
                        "mean_pct": round(acc.mean() * 100, 2), "std": round(float(acc.std()), 3),
                        "macro_f1": round(float(f1.mean()), 3),
                        "per_subject": [round(float(x), 4) for x in acc]}
    comps = []
    for a, b in COMPARISONS:
        if a in per_arm and b in per_arm:
            r = paired(per_arm[a]["per_subject"], per_arm[b]["per_subject"])
            r["comparison"] = f"{LABELS[a]} vs {LABELS[b]}"
            comps.append(r)
    if comps:
        holm(comps, "t_p", "holm_t")
        holm(comps, "wilcoxon_p", "holm_w")
    summary["bnci2014002"] = {"per_arm": per_arm, "comparisons": comps}

    print("=" * 96 + "\n  BNCI 2014-002 (n=14, seed-averaged)\n" + "=" * 96)
    for arm in LABELS:
        if arm in per_arm:
            s = per_arm[arm]
            print(f"  {s['label']:<30} {s['mean_pct']:6.2f}%  std {s['std']:.3f}  F1 {s['macro_f1']:.3f}  "
                  f"per-seed {s['per_seed_mean_pct']}")
    print("-" * 96)
    for r in comps:
        print(f"  {r['comparison']:<58} {r['mean_diff_pct']:+6.2f}pp  t_p={r['t_p']:.4f} "
              f"w_p={r['wilcoxon_p']:.4f} dz={r['dz']:+.2f}  Holm t/w={r['holm_t']:.4f}/{r['holm_w']:.4f}")

    # ── BCI-IV 2a reproduction check ──
    rep = {}
    for arm, pub in PUBLISHED_2A_SEED42.items():
        v = runs.get("bci2a", {}).get(arm, {}).get(42, {})
        if len(v) == 9:
            new = [v[s]["acc"] for s in range(1, 10)]
            r = paired(new, pub)
            r["per_subject_new"] = new
            rep[arm] = r
    summary["bci2a_reproduction_seed42"] = rep
    if rep:
        print("=" * 96 + "\n  BCI-IV 2a reproduction (unified script vs published, seed 42)\n" + "=" * 96)
        for arm, r in rep.items():
            print(f"  {LABELS[arm]:<30} new {r['mean_a_pct']:.2f}% vs published {r['mean_b_pct']:.2f}%  "
                  f"diff {r['mean_diff_pct']:+.2f}pp  t_p={r['t_p']:.3f}")

    out = os.path.join(run_dir, "decomposition_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
