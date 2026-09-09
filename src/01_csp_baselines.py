"""
BCI-IV 2a — CSP+LDA and CSP+SVM BASELINES (per-subject LOSO logging)
=======================================================================
Purpose: Figure 4's heatmap currently shows APPROXIMATED per-subject
values for CSP+LDA and CSP+SVM (interpolated from the mean accuracy
only, per the original caption). A reviewer flagged this as
inappropriate for an "empirical benchmark" figure. This script produces
REAL per-subject LOSO accuracy for both methods so the heatmap can be
redrawn without approximation.

Protocol: identical LOSO structure as every other method in the paper —
for each of the 9 folds, CSP filters + classifier are fit on the pooled
T-session data of the 8 source subjects, then evaluated on the held-out
target subject's E session. Zero target labels used at any stage.

Preprocessing differs from the EEGNet pipeline as specified in the
paper's Section 3.2: CSP-based baselines use an 8-30 Hz bandpass
(mu + beta bands), not the 4-40 Hz band used for EEGNet.

CSP implementation: standard one-vs-rest (OVR) multiclass extension.
For each of the 4 classes, solve the generalized eigenvalue problem
between that class's pooled covariance and the pooled covariance of all
other classes; take the top and bottom m eigenvectors as spatial filters
for that class. Log-variance of the CSP-projected signal is the feature
vector fed to LDA / SVM.

Output: csp_baselines_results.json with per-subject accuracy for both
methods, directly usable to redraw Figure 4 without approximation.
"""

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import os, json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/datasets/abinayajone/bci-iv-2a-mi"
SAVE_DIR   = "/kaggle/working"

FS         = 250
T_START    = 2.5
T_END      = 6.0
EPOCH_LEN  = int((T_END - T_START) * FS)
EOG_THRESH = 100.0
N_CLASSES  = 4
N_SUBJECTS = 9
M_FILTERS  = 3   # CSP filters per end (top m + bottom m = 2m per OVR problem)

# ─────────────────────────────────────────────
# PREPROCESSING — 8-30 Hz bandpass (CSP-specific, per paper Section 3.2)
# ─────────────────────────────────────────────
def bandpass_csp(data, lo=8, hi=30, fs=FS, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=-1)

def load_session_csp(path):
    """Loads a session with the CSP-specific 8-30 Hz bandpass.
    No EMS normalisation here — CSP operates on raw (bandpassed)
    covariance structure, consistent with standard CSP+LDA/SVM practice;
    EMS is an EEGNet-specific step per the paper's preprocessing text."""
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
        eeg = bandpass_csp(raw_X[:22])
        for onset, lbl in zip(t_pos, raw_y):
            if lbl < 1 or lbl > 4:
                continue
            s = int(onset + T_START * fs_run)
            e = int(onset + T_END * fs_run)
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
    X = np.stack(X_list).astype(np.float64)   # (n_trials, 22, T)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def load_all_sessions_csp():
    sessions = {}
    print("\nLoading all sessions (8-30 Hz CSP band)...")
    for s in range(1, N_SUBJECTS + 1):
        for sess in ['T', 'E']:
            fname = f"A0{s}{sess}.mat"
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fpath):
                continue
            X, y = load_session_csp(fpath)
            if X is None:
                continue
            sessions[f"S{s}{sess}"] = (X, y)
            print(f"  S{s}{sess}: {len(y)} trials")
    return sessions


# ─────────────────────────────────────────────
# ONE-VS-REST MULTICLASS CSP
# ─────────────────────────────────────────────
def trial_covariances(X):
    """X: (n_trials, C, T) -> (n_trials, C, C) normalised covariances."""
    n, C, T = X.shape
    covs = np.empty((n, C, C))
    for i in range(n):
        e = X[i]
        cov = e @ e.T
        cov /= np.trace(cov)
        covs[i] = cov
    return covs

def fit_ovr_csp(X_train, y_train, n_classes=N_CLASSES, m=M_FILTERS):
    """Returns a list of spatial filter matrices, one per class,
    each of shape (2m, C): top-m and bottom-m generalised eigenvectors
    of (class-k covariance) vs (pooled covariance of all other classes)."""
    covs = trial_covariances(X_train)
    filters_per_class = []
    for c in range(n_classes):
        mask_c = (y_train == c)
        C_c = covs[mask_c].mean(axis=0)
        C_rest = covs[~mask_c].mean(axis=0)
        # Generalised eigenproblem: C_c v = lambda (C_c + C_rest) v
        eigvals, eigvecs = eigh(C_c, C_c + C_rest)
        # eigvecs columns sorted ascending by eigval; take bottom m and top m
        order = np.argsort(eigvals)
        idx = np.concatenate([order[:m], order[-m:]])
        W_c = eigvecs[:, idx].T   # (2m, C)
        filters_per_class.append(W_c)
    return filters_per_class

def csp_log_var_features(X, filters_per_class):
    """X: (n_trials, C, T) -> (n_trials, n_classes * 2m) log-variance features."""
    n = X.shape[0]
    feats = []
    for i in range(n):
        e = X[i]
        row = []
        for W_c in filters_per_class:
            proj = W_c @ e                      # (2m, T)
            var = np.var(proj, axis=1)
            var = var / (var.sum() + 1e-12)
            row.append(np.log(var + 1e-12))
        feats.append(np.concatenate(row))
    return np.stack(feats)


# ─────────────────────────────────────────────
# LOSO — CSP+LDA and CSP+SVM
# ─────────────────────────────────────────────
def run_csp_loso(sessions):
    results = {"CSP_LDA": {}, "CSP_SVM": {}}

    for target in range(1, N_SUBJECTS + 1):
        print(f"\n{'='*60}\n  LOSO fold: target = S{target}\n{'='*60}")
        source_subjects = [s for s in range(1, N_SUBJECTS + 1) if s != target]

        Xt_key = f"S{target}E"
        if Xt_key not in sessions:
            continue
        X_target, y_target = sessions[Xt_key]

        X_pool, y_pool = [], []
        for src in source_subjects:
            key = f"S{src}T"
            if key not in sessions:
                continue
            X_s, y_s = sessions[key]
            X_pool.append(X_s)
            y_pool.append(y_s)
        X_pool = np.concatenate(X_pool, axis=0)
        y_pool = np.concatenate(y_pool, axis=0)

        filters_per_class = fit_ovr_csp(X_pool, y_pool)
        F_train = csp_log_var_features(X_pool, filters_per_class)
        F_test  = csp_log_var_features(X_target, filters_per_class)

        # CSP + LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(F_train, y_pool)
        pred_lda = lda.predict(F_test)
        acc_lda = float((pred_lda == y_target).mean())
        f1_lda = float(f1_score(y_target, pred_lda, average='macro', zero_division=0))
        results["CSP_LDA"][f"S{target}"] = {"acc": round(acc_lda, 4), "f1": round(f1_lda, 4)}
        print(f"  CSP+LDA: acc={acc_lda:.4f} f1={f1_lda:.4f}")

        # CSP + SVM (linear kernel, standard for CSP log-var features)
        svm = SVC(kernel='linear', C=1.0)
        svm.fit(F_train, y_pool)
        pred_svm = svm.predict(F_test)
        acc_svm = float((pred_svm == y_target).mean())
        f1_svm = float(f1_score(y_target, pred_svm, average='macro', zero_division=0))
        results["CSP_SVM"][f"S{target}"] = {"acc": round(acc_svm, 4), "f1": round(f1_svm, 4)}
        print(f"  CSP+SVM: acc={acc_svm:.4f} f1={f1_svm:.4f}")

        with open(os.path.join(SAVE_DIR, "csp_baselines_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    return results


def print_summary(results):
    print(f"\n{'='*70}\n  CSP BASELINES — LOSO SUMMARY\n{'='*70}")
    for method in ("CSP_LDA", "CSP_SVM"):
        accs = [v["acc"] for v in results[method].values()]
        f1s  = [v["f1"] for v in results[method].values()]
        print(f"  {method:10s} mean_acc={np.mean(accs):.4f}  std={np.std(accs):.4f}  mean_f1={np.mean(f1s):.4f}")
    print(f"\n  Compare to paper's reported means: CSP+LDA 40.93%, CSP+SVM 36.54%")


def main():
    np.random.seed(42)
    sessions = load_all_sessions_csp()
    print(f"\nLoaded {len(sessions)} sessions.")
    results = run_csp_loso(sessions)
    print_summary(results)
    out_path = os.path.join(SAVE_DIR, "csp_baselines_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved: {out_path}")


if __name__ == "__main__":
    main()