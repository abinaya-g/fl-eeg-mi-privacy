"""
BCI-IV 2a — CSP + LDA Cross-Subject Baseline
==============================================
Classical ML baseline for comparison with FL framework.
Runs entirely on CPU — no GPU needed.

Protocol:
  - LOSO cross-subject: test on each subject's E session
  - Source pool: all other subjects' T sessions only
    (using T only for source, not E, to match standard CSP protocol)
  - CSP: 6 spatial filters (3 per class pair, OVR decomposition)
  - Features: log-variance of CSP-filtered signals
  - Classifier: LDA with shrinkage (regularised for small samples)
  - Also runs SVM for comparison

Note on CSP for 4-class:
  Standard CSP is binary. For 4-class we use One-vs-Rest (OVR):
  train 4 binary CSP filters, concatenate features, classify with LDA/SVM.

Results will be added to the final comparison table.
"""

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os, json, time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/datasets/abinayajone/bci-iv-2a-mi"
SAVE_DIR   = "/kaggle/working"

FS         = 250
T_START    = 2.5
T_END      = 6.0
EPOCH_LEN  = int((T_END - T_START) * FS)   # 875 samples
EOG_THRESH = 100.0                           # µV post-bandpass
N_CLASSES  = 4
N_SUBJECTS = 9
N_CHANNELS = 22

CSP_FILTERS_PER_CLASS = 2   # filters per side per binary CSP → total 4 per class pair
                             # with OVR: 4 classes × 4 filters = 16 features total

print(f"CSP filters per class (OVR): {CSP_FILTERS_PER_CLASS} per side")
print(f"Total CSP features: {N_CLASSES * CSP_FILTERS_PER_CLASS * 2}")


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def bandpass(data, lo=8, hi=30, fs=FS, order=4):
    """8-30 Hz — mu and beta bands most relevant for MI."""
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=-1)


def load_session(path):
    """
    Load one .mat session → (X, y)
    X: (N, C, T) raw epochs after bandpass
    y: (N,) 0-indexed labels
    EOG rejection applied after bandpass.
    """
    mat  = loadmat(path, struct_as_record=False, squeeze_me=True)
    data = mat['data']
    X_list, y_list = [], []

    for run_idx in range(len(data)):
        run = data[run_idx]
        try:
            raw_X  = run.X.T
            raw_y  = run.y
            t_pos  = run.trial
            fs_run = run.fs
        except AttributeError:
            continue
        if not hasattr(raw_y, '__len__') or len(raw_y) == 0:
            continue

        eeg = bandpass(raw_X[:N_CHANNELS], lo=8, hi=30)

        for onset, lbl in zip(t_pos, raw_y):
            if lbl < 1 or lbl > 4:
                continue
            s = int(onset + T_START * fs_run)
            e = int(onset + T_END   * fs_run)
            if e > eeg.shape[1]:
                continue
            epoch = eeg[:, s:e]
            if epoch.shape[1] != EPOCH_LEN:
                continue
            if np.max(np.abs(epoch)) > EOG_THRESH:
                continue
            X_list.append(epoch)
            y_list.append(lbl - 1)

    if len(X_list) == 0:
        return None, None

    X = np.stack(X_list)   # (N, C, T)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def load_all_sessions():
    sessions = {}
    print("\nLoading all sessions...")
    for s in range(1, N_SUBJECTS + 1):
        for sess in ['T', 'E']:
            fname = f"A0{s}{sess}.mat"
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found")
                continue
            X, y = load_session(fpath)
            if X is None:
                continue
            key = f"S{s}{sess}"
            sessions[key] = (X, y)
            classes = [int((y == c).sum()) for c in range(N_CLASSES)]
            print(f"  {key}: {len(y)}/288 ({288-len(y)} rej) classes={classes}")
    return sessions


# ─────────────────────────────────────────────
# CSP IMPLEMENTATION
# ─────────────────────────────────────────────
def compute_covariance(X):
    """
    Compute normalised covariance matrix for a set of trials.
    X: (N, C, T)
    Returns: (C, C) mean covariance matrix
    """
    covs = []
    for trial in X:
        # Normalise by trace
        cov = trial @ trial.T
        cov /= np.trace(cov)
        covs.append(cov)
    return np.mean(covs, axis=0)


def csp_binary(X_pos, X_neg, n_filters=2):
    """
    Binary CSP: find spatial filters that maximise variance for X_pos
    and minimise for X_neg.

    X_pos: (N+, C, T) — positive class trials
    X_neg: (N-, C, T) — negative class trials
    n_filters: number of filters per side (total = 2*n_filters)

    Returns: W (C, 2*n_filters) filter matrix
    """
    cov_pos = compute_covariance(X_pos)
    cov_neg = compute_covariance(X_neg)

    # Generalised eigenvalue problem: cov_pos * w = λ * (cov_pos + cov_neg) * w
    combined = cov_pos + cov_neg
    eigenvalues, eigenvectors = eigh(cov_pos, combined)

    # Sort by eigenvalue — take n_filters from each end
    idx     = np.argsort(eigenvalues)
    # Best filters: largest (most variance for pos) and smallest (most for neg)
    sel_idx = np.concatenate([idx[:n_filters], idx[-n_filters:]])
    W       = eigenvectors[:, sel_idx]  # (C, 2*n_filters)
    return W


def csp_ovr(X, y, n_filters=2):
    """
    One-vs-Rest CSP for multiclass.
    Trains one binary CSP per class against all others.

    Returns: list of W matrices, one per class
    """
    classes = np.unique(y)
    filters = []
    for c in classes:
        X_pos = X[y == c]
        X_neg = X[y != c]
        W = csp_binary(X_pos, X_neg, n_filters=n_filters)
        filters.append(W)
    return filters   # list of N_CLASSES matrices, each (C, 2*n_filters)


def extract_csp_features(X, filters):
    """
    Apply CSP filters and extract log-variance features.

    X: (N, C, T)
    filters: list of (C, 2*n_filters) matrices

    Returns: (N, N_CLASSES * 2 * n_filters) feature matrix
    """
    N = X.shape[0]
    features = []

    for W in filters:
        # W: (C, F)  X: (N, C, T)
        # Apply spatial filter: project C channels onto F filters
        # filtered: (N, F, T)
        filtered = np.tensordot(X, W, axes=([1], [0]))  # (N, T, F)
        filtered = filtered.transpose(0, 2, 1)           # (N, F, T)
        # Log variance per filter across time
        var     = np.var(filtered, axis=2)               # (N, F)
        log_var = np.log(var + 1e-8)
        features.append(log_var)

    return np.concatenate(features, axis=1)   # (N, total_features)


# ─────────────────────────────────────────────
# LOSO
# ─────────────────────────────────────────────
def run_loso_csp(sessions, classifier_name='LDA'):
    print(f"\n{'='*60}")
    print(f"  CSP + {classifier_name} — LOSO Cross-Subject")
    print(f"{'='*60}")

    results  = {}
    all_accs = []
    all_f1s  = []

    for test_subj in range(1, N_SUBJECTS + 1):
        tgt_T = f"S{test_subj}T"
        tgt_E = f"S{test_subj}E"

        if tgt_T not in sessions or tgt_E not in sessions:
            print(f"  Skipping S{test_subj} — missing session")
            continue

        t0 = time.time()

        # Build source pool — T sessions only from other subjects
        src_X_parts, src_y_parts = [], []
        for s in range(1, N_SUBJECTS + 1):
            if s == test_subj:
                continue
            k = f"S{s}T"
            if k in sessions:
                src_X_parts.append(sessions[k][0])
                src_y_parts.append(sessions[k][1])

        src_X = np.concatenate(src_X_parts)   # (N_src, C, T)
        src_y = np.concatenate(src_y_parts)

        tgt_X_E, tgt_y_E = sessions[tgt_E]

        print(f"\n  Fold {test_subj}: Test=S{test_subj}")
        print(f"    Src: {len(src_y)} trials | "
              f"Tgt T: {len(sessions[tgt_T][1])} | "
              f"Tgt E: {len(tgt_y_E)}")

        # Train CSP filters on source data
        filters = csp_ovr(src_X, src_y, n_filters=CSP_FILTERS_PER_CLASS)

        # Extract features
        X_src_feat = extract_csp_features(src_X, filters)
        X_tgt_feat = extract_csp_features(tgt_X_E, filters)

        # Normalise features
        scaler     = StandardScaler()
        X_src_feat = scaler.fit_transform(X_src_feat)
        X_tgt_feat = scaler.transform(X_tgt_feat)

        # Train classifier on source features
        if classifier_name == 'LDA':
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        elif classifier_name == 'SVM':
            clf = SVC(kernel='rbf', C=1.0, gamma='scale',
                      decision_function_shape='ovr', random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        clf.fit(X_src_feat, src_y)

        # Predict on target E session
        preds = clf.predict(X_tgt_feat)
        acc   = accuracy_score(tgt_y_E, preds)
        f1    = f1_score(tgt_y_E, preds, average='macro', zero_division=0)

        elapsed = time.time() - t0
        print(f"    Acc={acc:.4f}  F1={f1:.4f}  [{elapsed:.1f}s]")
        print(classification_report(
            tgt_y_E, preds,
            target_names=['Left Hand', 'Right Hand', 'Both Feet', 'Tongue'],
            zero_division=0
        ))

        results[f"S{test_subj}"] = {
            "acc": round(float(acc), 4),
            "f1":  round(float(f1), 4),
            "n":   int(len(tgt_y_E))
        }
        all_accs.append(float(acc))
        all_f1s.append(float(f1))

    # Summary
    print(f"\n{'='*60}")
    print(f"  CSP + {classifier_name} LOSO SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Subj':<8} {'Acc':>8} {'F1':>8}")
    print(f"  {'-'*28}")
    for s in range(1, N_SUBJECTS + 1):
        k = f"S{s}"
        if k in results:
            r = results[k]
            print(f"  {k:<8} {r['acc']:>8.4f} {r['f1']:>8.4f}")
    print(f"  {'-'*28}")
    if all_accs:
        print(f"  {'Mean':<8} {np.mean(all_accs):>8.4f} {np.mean(all_f1s):>8.4f}")
        print(f"  {'Std':<8} {np.std(all_accs):>8.4f}")

    return results, np.mean(all_accs), np.mean(all_f1s)


# ─────────────────────────────────────────────
# FINAL COMPARISON TABLE
# ─────────────────────────────────────────────
def print_final_comparison(csp_lda_acc, csp_lda_f1,
                            csp_svm_acc, csp_svm_f1):
    base = 0.4100
    print(f"\n{'='*72}")
    print(f"  COMPLETE RESULTS — BCI-IV 2a, 9-subject LOSO")
    print(f"  All methods: unsupervised cross-subject (no target labels)")
    print(f"{'='*72}")
    print(f"  {'Method':<42} {'Privacy':>7} {'Acc':>8} {'vs Base':>10}")
    print(f"  {'-'*70}")

    rows = [
        ("CSP + LDA (classical baseline)",     "✗", csp_lda_acc),
        ("CSP + SVM (classical baseline)",     "✗", csp_svm_acc),
        ("Centralised EEGNet (no adapt)",      "✗", 0.4100),
        ("Centralised + CORAL (λ=10)",          "✗", 0.4358),
        ("FedAvg global (proposed)",            "✓", 0.4613),
        ("FedAvg + local heads",                "✓", 0.4346),
        ("FedAvg + local heads + CORAL",        "✓", 0.4316),
    ]

    for name, priv, acc in rows:
        d     = acc - base
        arrow = f"▲{d:.4f}" if d > 0 else (f"▼{abs(d):.4f}" if d < 0 else "─")
        best  = "← BEST" if acc == max(r[2] for r in rows) else ""
        print(f"  {name:<42} {priv:>7} {acc:>8.4f} {arrow:>10}  {best}")

    print(f"\n  Chance (4-class): 0.2500")
    print(f"\n  Note: CTL (Gao et al., 2026) reports 73.13% on same dataset")
    print(f"  but uses 160 LABELED TARGET SAMPLES — semi-supervised, not")
    print(f"  comparable to our fully unsupervised cross-subject protocol.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    np.random.seed(42)

    print("="*60)
    print("  BCI-IV 2a — CSP + LDA/SVM Classical Baseline")
    print("  Running on CPU — no GPU required")
    print("="*60)

    sessions = load_all_sessions()
    print(f"\nLoaded {len(sessions)} sessions.")

    # Run CSP + LDA
    lda_results, lda_acc, lda_f1 = run_loso_csp(sessions, 'LDA')

    # Run CSP + SVM
    svm_results, svm_acc, svm_f1 = run_loso_csp(sessions, 'SVM')

    # Print final comparison
    print_final_comparison(lda_acc, lda_f1, svm_acc, svm_f1)

    # Save results
    out = {
        "CSP_LDA": {
            "mean_acc": round(lda_acc, 4),
            "mean_f1":  round(lda_f1, 4),
            "per_subject": lda_results
        },
        "CSP_SVM": {
            "mean_acc": round(svm_acc, 4),
            "mean_f1":  round(svm_f1, 4),
            "per_subject": svm_results
        }
    }
    out_path = os.path.join(SAVE_DIR, "csp_baseline_results.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()