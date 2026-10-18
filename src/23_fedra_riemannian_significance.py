"""
FEDRA (TRUE RIEMANNIAN MEAN) — SUMMARY AND PAIRED SIGNIFICANCE, n=9
====================================================================
Collects the per-fold JSONs written by 22_fedra_riemannian.py, averages
the two seeds per subject (same n=9 convention as
21_final_significance_n9_consistent.py), and runs the same paired tests
(paired t, Wilcoxon, Cohen's d_z, bootstrap 95% CI) against the FedAvg,
Fed-EA and original arithmetic-mean FedRA reference values.

Usage: python src/23_fedra_riemannian_significance.py [results/fedra_riemann]
Output: <dir>/fedra_riemann_summary.json and a printed table.
"""

import glob, json, os, sys
import numpy as np
from scipy import stats

# Reference per-subject accuracies, copied unchanged from
# 21_final_significance_n9_consistent.py.
FEDAVG_42  = {1:0.5694,2:0.2708,3:0.6411,4:0.3368,5:0.2465,6:0.2847,7:0.2986,8:0.5625,9:0.4201}
FEDAVG_123 = {1:0.5868,2:0.2986,3:0.5993,4:0.3056,5:0.2465,6:0.2778,7:0.2951,8:0.5556,9:0.3750}
FEDEA_TRANS_42  = {1:0.6840,2:0.3264,3:0.7038,4:0.3646,5:0.2812,6:0.4375,7:0.4792,8:0.6597,9:0.6250}
FEDEA_TRANS_123 = {1:0.6875,2:0.3368,3:0.6934,4:0.3785,5:0.2812,6:0.4375,7:0.4583,8:0.6424,9:0.5903}
FEDRA_ARITH = {1:0.6181,2:0.3160,3:0.5331,4:0.3299,5:0.2535,6:0.2847,7:0.3438,8:0.5000,9:0.4861}  # single-seed

def seed_avg(d1, d2):
    return {s: (d1[s] + d2[s]) / 2 for s in d1}

FEDAVG = seed_avg(FEDAVG_42, FEDAVG_123)
FEDEA_TRANSDUCTIVE = seed_avg(FEDEA_TRANS_42, FEDEA_TRANS_123)


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
    _, p_t = stats.ttest_rel(a, b)
    _, p_w = stats.wilcoxon(a, b)
    lo, hi = bootstrap_ci(diffs)
    return {"comparison": f"{name_a} vs {name_b}", "n": len(keys),
            "mean_a_pct": round(a.mean() * 100, 2), "mean_b_pct": round(b.mean() * 100, 2),
            "mean_diff_pct": round(diffs.mean() * 100, 2),
            "t_p": round(float(p_t), 5), "wilcoxon_p": round(float(p_w), 5),
            "dz": round(cohens_dz(diffs), 3),
            "bootstrap_95ci_pct": [round(lo * 100, 2), round(hi * 100, 2)]}


def load_runs(run_dir):
    runs = {}
    for path in glob.glob(os.path.join(run_dir, "*_seed*_S*.json")):
        with open(path) as f:
            r = json.load(f)
        runs.setdefault(r["arm"], {}).setdefault(r["seed"], {})[int(r["target"][1:])] = r
    return runs


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "results", "fedra_riemann")
    runs = load_runs(run_dir)
    if not runs:
        raise SystemExit(f"No run files found in {run_dir}")

    summary = {"per_arm": {}, "comparisons": []}
    for arm in sorted(runs):
        seeds = runs[arm]
        complete = {sd: v for sd, v in seeds.items() if len(v) == 9}
        for sd, v in seeds.items():
            if len(v) != 9:
                print(f"  WARNING: {arm} seed {sd} has {len(v)}/9 folds — excluded")
        if not complete:
            continue
        per_seed = {sd: {s: v[s]["acc"] for s in v} for sd, v in complete.items()}
        per_subject = {s: float(np.mean([per_seed[sd][s] for sd in per_seed]))
                       for s in range(1, 10)}
        label = {"global": "FedRA-Riemann (global, inductive)",
                 "local": "RA-local (per-subject, transductive)"}[arm]
        summary["per_arm"][arm] = {
            "label": label, "seeds": sorted(per_seed),
            "per_seed_mean_pct": {sd: round(np.mean(list(a.values())) * 100, 2)
                                  for sd, a in per_seed.items()},
            "mean_pct": round(np.mean(list(per_subject.values())) * 100, 2),
            "per_subject": {f"S{s}": round(a, 4) for s, a in per_subject.items()},
        }
        summary["comparisons"] += [
            paired(label, per_subject, "FedAvg", FEDAVG),
            paired("Fed-EA", FEDEA_TRANSDUCTIVE, label, per_subject),
            paired(label, per_subject, "FedRA-arith (orig.)", FEDRA_ARITH),
        ]

    if {"global", "local"} <= set(summary["per_arm"]):
        g = {int(k[1:]): v for k, v in summary["per_arm"]["global"]["per_subject"].items()}
        l = {int(k[1:]): v for k, v in summary["per_arm"]["local"]["per_subject"].items()}
        summary["comparisons"].append(paired("RA-local", l, "FedRA-Riemann", g))

    print("=" * 80)
    for arm, s in summary["per_arm"].items():
        print(f"  {s['label']:<40} mean={s['mean_pct']:.2f}%  per-seed={s['per_seed_mean_pct']}")
    print("-" * 80)
    for r in summary["comparisons"]:
        print(f"  {r['comparison']:<58} diff={r['mean_diff_pct']:>6.2f}pp "
              f"t_p={r['t_p']:.4f} w_p={r['wilcoxon_p']:.4f} dz={r['dz']:>5.2f} "
              f"CI={r['bootstrap_95ci_pct']}")

    out = os.path.join(run_dir, "fedra_riemann_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
