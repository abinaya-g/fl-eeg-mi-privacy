"""
FED-EA CALIBRATION-SWEEP — SIGNIFICANCE TESTING
==========================================================================
Purpose: apply the same paired statistical-testing standard used
throughout the rest of this project (paired t-test, Wilcoxon
signed-rank, Cohen's dz, bootstrap 95% CI, Holm-Bonferroni correction
across the full comparison family) to the target-calibration-size
sweep, so the "accuracy plateaus by 25% calibration" claim is backed
by the same rigor as every other comparison in the manuscript
(CORAL vs. FedAvg, FedRA vs. FedAvg, inductive vs. transductive, etc.)
rather than asserted from eyeballing means and error bars.

Inputs (all REQUIRED, script will not fabricate missing data):
  - fed_ea_calibration_sweep_results.json   (from fed_ea_calibration_sweep_v2.py)
  - fed_ea_inductive_results.json           (the original inductive-variant run)
  - fedavg_global_reverify_results.json     (seed 42)
  - fedavg_global_reverify_seed123_results.json (seed 123)

Pairing note, stated explicitly because it affects which comparisons
are even valid: the calibration-sweep and FedAvg-baseline results are
2-seed (n=18 subject×seed cells), but the original inductive-variant
run (fed_ea_inductive_results.json) was a SINGLE-SEED run (seed 42
only, n=9 subjects) based on how it was generated in this project.
Comparisons involving the inductive variant are therefore restricted
to the matching seed=42 subset of the other conditions (n=9), so every
paired test compares like-for-like subjects under the same seed. If
you later re-run the inductive variant with a second seed, re-run this
script -- comparisons against it will automatically use the full n=18
once both files report matching seeds.

Comparison family tested (all pairs relevant to the calibration-sweep
claim; Holm-Bonferroni correction is applied ONCE across this entire
family, not per-comparison, matching Table 8's existing convention):
  1. Inductive        vs. FedAvg (no alignment)     [n=9,  sanity check -- already reported elsewhere]
  2. Inductive        vs. frac=0.25
  3. Inductive        vs. frac=1.00 (transductive)
  4. frac=0.25        vs. frac=0.50
  5. frac=0.25        vs. frac=0.75
  6. frac=0.25        vs. frac=1.00
  7. frac=0.50        vs. frac=0.75
  8. frac=0.50        vs. frac=1.00
  9. frac=0.75        vs. frac=1.00
 10. frac=0.25        vs. FedAvg (no alignment)     [n=18]

Output: calibration_sweep_significance_results.json + a printed table
in the same format as the manuscript's existing significance tables.
"""

import json
import os
import numpy as np
from scipy import stats

SAVE_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

PATHS = {
    "calib":   os.path.join(SAVE_DIR, "fed_ea_calibration_sweep_results.json"),
    "induct":  os.path.join(SAVE_DIR, "fed_ea_inductive_results.json"),
    "fedavg42":  os.path.join(SAVE_DIR, "fedavg_global_reverify_results.json"),
    "fedavg123": os.path.join(SAVE_DIR, "fedavg_global_reverify_seed123_results.json"),
}
OUT_JSON = os.path.join(SAVE_DIR, "calibration_sweep_significance_results.json")

N_SUBJECTS = 9
N_BOOTSTRAP = 10000
RNG = np.random.RandomState(42)

# ─────────────────────────────────────────────
# LOAD DATA — fail loudly and specifically if anything is missing,
# rather than substituting placeholder numbers.
# ─────────────────────────────────────────────
def require_file(key):
    path = PATHS[key]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\nMissing required file: {path}\n"
            f"This script does not fabricate results -- please make sure "
            f"all four input JSON files listed in the docstring are present "
            f"in {SAVE_DIR} before running.\n")
    with open(path) as f:
        return json.load(f)

def load_calibration_per_subject_seed():
    """Returns dict: fraction_str -> {(subject, seed): mean_acc_over_splits}"""
    data = require_file("calib")
    per_run = data.get("per_run", data)  # tolerate either wrapped or flat format
    out = {"0.25": {}, "0.50": {}, "0.75": {}, "1.00": {}}
    tmp = {"0.25": {}, "0.50": {}, "0.75": {}, "1.00": {}}
    for run in per_run.values():
        frac_key = f"{run['fraction']:.2f}"
        if frac_key not in tmp:
            continue
        k = (run["subject"], run["seed"])
        tmp[frac_key].setdefault(k, []).append(run["acc"])
    for frac_key, d in tmp.items():
        for k, v in d.items():
            out[frac_key][k] = float(np.mean(v))
    return out

def load_fedavg_per_subject_seed():
    """Returns dict: (subject, seed) -> acc, combining the two
    single-seed FedAvg re-verification files."""
    out = {}
    for seed, key in [(42, "fedavg42"), (123, "fedavg123")]:
        data = require_file(key)
        # Expected format: {"S1": {"acc":..., "f1":...}, ...} -- same
        # format used by every other per-fold results JSON in this project.
        for subj_key, v in data.items():
            if not subj_key.startswith("S"):
                continue
            subj = int(subj_key[1:])
            out[(subj, seed)] = float(v["acc"])
    return out

def load_inductive_per_subject():
    """Returns dict: subject -> acc, and the seed it was run under
    (inferred from the file if present, else assumed 42 per this
    project's convention for single-seed runs unless stated otherwise)."""
    data = require_file("induct")
    out = {}
    seed_used = data.get("seed", 42)  # fall back to 42, project's default single-seed
    for subj_key, v in data.items():
        if not subj_key.startswith("S"):
            continue
        subj = int(subj_key[1:])
        out[subj] = float(v["acc"]) if isinstance(v, dict) else float(v)
    print(f"  Inductive results: {len(out)} subjects, assumed seed={seed_used} "
          f"(single-seed run; verify this matches how fed_ea_inductive_results.json "
          f"was actually generated).")
    return out, seed_used

# ─────────────────────────────────────────────
# STATISTICS — matching the project's existing convention exactly
# ─────────────────────────────────────────────
def cohens_dz(diffs):
    d = np.asarray(diffs, dtype=float)
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(d.mean() / sd)

def bootstrap_ci_mean_diff(diffs, n_boot=N_BOOTSTRAP, ci=0.95, rng=RNG):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = d[rng.randint(0, n, n)]
        boot_means[i] = sample.mean()
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)

def paired_test(name_a, vals_a, name_b, vals_b, common_keys):
    """vals_a, vals_b: dict key -> acc. common_keys: iterable of keys
    present in both. Returns a result dict; raises if fewer than 2
    matched pairs are found (cannot run a paired test on <2 pairs)."""
    a = np.array([vals_a[k] for k in common_keys])
    b = np.array([vals_b[k] for k in common_keys])
    n = len(a)
    if n < 2:
        raise ValueError(f"Only {n} matched pairs between {name_a} and {name_b} "
                          f"-- cannot run a paired test.")
    diffs = a - b
    mean_diff = float(diffs.mean())

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        # all differences zero, or too few non-zero diffs
        w_p = 1.0

    dz = cohens_dz(diffs)
    ci_lo, ci_hi = bootstrap_ci_mean_diff(diffs)

    return {
        "comparison": f"{name_a} vs. {name_b}",
        "n_pairs": n,
        "mean_diff_pct": round(mean_diff * 100, 3),
        "t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "cohens_dz": round(dz, 3),
        "bootstrap_95ci_pct": [round(ci_lo * 100, 3), round(ci_hi * 100, 3)],
    }

def holm_bonferroni(results, p_key):
    """Adds a 'holm_p' field to each result dict, applying Holm-Bonferroni
    correction across the FULL set of results passed in (matching this
    project's existing convention of correcting once across the whole
    comparison family, not per sub-comparison)."""
    m = len(results)
    order = sorted(range(m), key=lambda i: results[i][p_key])
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * results[idx][p_key]
        running_max = max(running_max, adj)
        results[idx]["holm_p"] = round(min(running_max, 1.0), 5)
    return results

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Loading data...")
    calib = load_calibration_per_subject_seed()
    fedavg = load_fedavg_per_subject_seed()
    inductive, induct_seed = load_inductive_per_subject()

    # restrict comparisons involving the (single-seed) inductive variant
    # to the matching-seed subset of the other conditions, so every
    # paired test compares the SAME subjects under the SAME seed.
    inductive_keyed = {(s, induct_seed): acc for s, acc in inductive.items()}

    def common(d1, d2):
        return sorted(set(d1.keys()) & set(d2.keys()))

    results = []

    # 1. Inductive vs FedAvg (sanity check -- already reported elsewhere
    #    in the manuscript at n=9; recomputed here for completeness)
    keys = common(inductive_keyed, fedavg)
    results.append(paired_test("Inductive", inductive_keyed,
                                "FedAvg (no alignment)", fedavg, keys))

    # 2. Inductive vs frac=0.25
    keys = common(inductive_keyed, calib["0.25"])
    results.append(paired_test("Inductive", inductive_keyed,
                                "frac=0.25", calib["0.25"], keys))

    # 3. Inductive vs frac=1.00 (transductive)
    keys = common(inductive_keyed, calib["1.00"])
    results.append(paired_test("Inductive", inductive_keyed,
                                "frac=1.00 (transductive)", calib["1.00"], keys))

    # 4-9. Adjacent and non-adjacent fraction comparisons (full n=18)
    frac_pairs = [("0.25", "0.50"), ("0.25", "0.75"), ("0.25", "1.00"),
                  ("0.50", "0.75"), ("0.50", "1.00"), ("0.75", "1.00")]
    for fa, fb in frac_pairs:
        keys = common(calib[fa], calib[fb])
        results.append(paired_test(f"frac={fa}", calib[fa],
                                    f"frac={fb}", calib[fb], keys))

    # 10. frac=0.25 vs FedAvg baseline (full n=18)
    keys = common(calib["0.25"], fedavg)
    results.append(paired_test("frac=0.25", calib["0.25"],
                                "FedAvg (no alignment)", fedavg, keys))

    # Holm-Bonferroni correction across the t-test p-values in this
    # entire family (matching the manuscript's existing convention);
    # done separately for the Wilcoxon p-values as a second family.
    results_t = [dict(r) for r in results]
    for r, rt in zip(results, results_t):
        rt["t_p"] = r["t_p"]
    holm_bonferroni(results, "t_p")
    for r in results:
        r["holm_p_ttest"] = r.pop("holm_p")
    holm_bonferroni(results, "wilcoxon_p")
    for r in results:
        r["holm_p_wilcoxon"] = r.pop("holm_p")

    print(f"\n{'='*100}")
    print("  CALIBRATION-SWEEP SIGNIFICANCE TESTING — SUMMARY")
    print(f"{'='*100}")
    header = (f"{'Comparison':<38}{'n':>4}{'Δ (pp)':>10}{'t-test p':>11}"
              f"{'Wilcoxon p':>12}{'dz':>7}{'Holm p(t)':>11}{'Holm p(W)':>11}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['comparison']:<38}{r['n_pairs']:>4}{r['mean_diff_pct']:>10.2f}"
              f"{r['t_p']:>11.4f}{r['wilcoxon_p']:>12.4f}{r['cohens_dz']:>7.2f}"
              f"{r['holm_p_ttest']:>11.4f}{r['holm_p_wilcoxon']:>11.4f}")

    print(f"\n  Interpretation guide:")
    print(f"  - If frac=0.25 vs frac=1.00 is NOT significant after Holm")
    print(f"    correction: supports 'accuracy plateaus by 25% calibration'.")
    print(f"  - If Inductive vs frac=0.25 IS significant with a large")
    print(f"    positive effect: confirms a small amount of target data")
    print(f"    recovers most of Fed-EA's advantage.")
    print(f"  - Note the n=9 comparisons (rows involving 'Inductive') use")
    print(f"    fewer pairs than the n=18 comparisons -- treat p-values")
    print(f"    from the smaller sample with appropriately more caution.")

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved: {OUT_JSON}")

if __name__ == "__main__":
    main()
