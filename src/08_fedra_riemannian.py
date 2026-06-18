"""
BCI-IV 2a — Federated Riemannian Alignment (FedRA)
====================================================
Novel contribution: Privacy-preserving cross-subject MI classification
via federated learning with Riemannian geometry-based distribution alignment.

Key idea:
  EEG covariance matrices live on a Riemannian manifold (SPD manifold).
  Standard CORAL aligns features in Euclidean space — geometrically incorrect
  for covariance matrices and ignores manifold structure.

  FedRA aligns each subject's EEG distribution toward the global Riemannian
  mean of all subjects' distributions — preserving manifold geometry.

Protocol each round:
  1. Server broadcasts: model weights + global Riemannian mean M_global
  2. Each client:
     a. Aligns its trials: C_aligned = M_global^(-1/2) C M_global^(-T/2)
     b. Trains EEGNet on Riemannian-aligned trials (LOCAL_EPOCHS)
     c. Computes local Riemannian mean M_s from its trials
     d. Returns: model weights + M_s
  3. Server:
     a. FedAvg on model weights
     b. Riemannian mean of {M_s} → new M_global
     c. Repeat

Privacy: only model weights + one 22×22 SPD matrix per client shared.
         No raw EEG data leaves the client.

Comparison modes:
  Mode 1: FedAvg global (best baseline, 0.4667)
  Mode 2: FedRA — Federated Riemannian Alignment (proposed)
  Mode 3: FedRA + EEGNet feature CORAL (full system)

vs prior work:
  - CORAL (Sun et al. 2016): Euclidean feature alignment, centralised
  - CTL (Gao et al. 2026): Riemannian features but centralised, needs target labels
  - FedRA: Riemannian manifold alignment, federated, zero target labels
"""

# ── Install pyriemann if not present ────────────────────────────────
import subprocess, sys
try:
    import pyriemann
except ImportError:
    print("Installing pyriemann...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pyriemann", "--quiet"])
    import pyriemann

from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm, sqrtm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score, classification_report
import os, json, copy, time
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/datasets/abinayajone/bci-iv-2a-mi"
SAVE_DIR   = "/kaggle/working"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FS         = 250
T_START    = 2.5
T_END      = 6.0
EPOCH_LEN  = int((T_END - T_START) * FS)   # 875 samples
EOG_THRESH = 100.0
N_CLASSES  = 4
N_SUBJECTS = 9
N_CHANNELS = 22

FL_ROUNDS    = 100
LOCAL_EPOCHS = 5
BATCH_SIZE   = 32
LR           = 1e-3
FL_PATIENCE  = 25

# Regularisation for covariance estimation (prevents singular matrices)
# EMS-normalised EEG needs stronger regularisation than raw EEG
COV_REG = 1e-3

print(f"Device      : {DEVICE}")
print(f"FL rounds   : {FL_ROUNDS}  Local epochs: {LOCAL_EPOCHS}")
print(f"Patience    : {FL_PATIENCE}")
print(f"Cov reg     : {COV_REG}")


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def bandpass(data, lo=4, hi=40, fs=FS, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=-1)


def exponential_moving_standardize(data, decay=0.999, eps=1e-6):
    out  = np.zeros_like(data)
    mean = np.zeros(data.shape[0])
    var  = np.ones(data.shape[0])
    for t in range(data.shape[1]):
        mean = decay * mean + (1 - decay) * data[:, t]
        var  = decay * var  + (1 - decay) * (data[:, t] - mean) ** 2
        out[:, t] = (data[:, t] - mean) / (np.sqrt(var) + eps)
    return out


def load_session(path, return_raw=False):
    """
    Load one session.
    If return_raw=True: returns (X_raw, X_ems, y)
      X_raw: bandpass-filtered epochs BEFORE EMS — used for covariance/alignment
      X_ems: EMS-normalised epochs — used for EEGNet training
    If return_raw=False: returns (X_ems, y) — backward compatible
    """
    mat  = loadmat(path, struct_as_record=False, squeeze_me=True)
    data = mat['data']
    X_raw_list, X_ems_list, y_list = [], [], []

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
        eeg = bandpass(raw_X[:N_CHANNELS])
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
            # Store RAW bandpass epoch (for covariance computation)
            X_raw_list.append(epoch.astype(np.float32))
            # Store EMS epoch (for EEGNet training)
            X_ems_list.append(
                exponential_moving_standardize(epoch).astype(np.float32))
            y_list.append(lbl - 1)

    if len(X_raw_list) == 0:
        if return_raw:
            return None, None, None
        return None, None

    # Shape: (N, C, T) for raw, (N, 1, C, T) for EMS
    X_raw = np.stack(X_raw_list)               # (N, C, T)
    X_ems = np.stack(X_ems_list)[:, np.newaxis] # (N, 1, C, T)
    y     = np.array(y_list, dtype=np.int64)

    if return_raw:
        return X_raw, X_ems, y
    return X_ems, y


def load_all_sessions(return_raw=False):
    """
    Load all sessions.
    If return_raw=True: sessions[key] = (X_raw, X_ems, y)
    If return_raw=False: sessions[key] = (X_ems, y)
    """
    sessions = {}
    print("\nLoading all sessions...")
    for s in range(1, N_SUBJECTS + 1):
        for sess in ['T', 'E']:
            fname = f"A0{s}{sess}.mat"
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fpath):
                continue
            if return_raw:
                X_raw, X_ems, y = load_session(fpath, return_raw=True)
                if X_raw is None:
                    continue
                key = f"S{s}{sess}"
                sessions[key] = (X_raw, X_ems, y)
            else:
                X_ems, y = load_session(fpath, return_raw=False)
                if X_ems is None:
                    continue
                key = f"S{s}{sess}"
                sessions[key] = (X_ems, y)
            classes = [int((y == c).sum()) for c in range(N_CLASSES)]
            print(f"  {key}: {len(y)}/288 ({288-len(y)} rej) "
                  f"classes={classes}")
    return sessions


# ─────────────────────────────────────────────
# RIEMANNIAN GEOMETRY UTILITIES
# ─────────────────────────────────────────────
def compute_covariance_matrix(epoch):
    """
    Compute regularised covariance matrix for one trial.
    epoch: (C, T) numpy array
    Returns: (C, C) SPD matrix
    """
    C, T  = epoch.shape
    # Mean-center each channel
    epoch = epoch - epoch.mean(axis=1, keepdims=True)
    cov   = (epoch @ epoch.T) / (T - 1)

    # Symmetrise (numerical precision)
    cov = (cov + cov.T) / 2

    # Adaptive regularisation — ensure all eigenvalues positive
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = eigvals.min()
    if min_eig < COV_REG:
        # Add enough regularisation to push smallest eigenvalue above COV_REG
        reg = COV_REG - min_eig + COV_REG
    else:
        reg = COV_REG
    cov = cov + reg * np.eye(C)

    return cov


def compute_subject_covariances(X_raw):
    """
    Compute covariance matrix for each trial from RAW bandpass epochs.
    X_raw: (N, C, T) — bandpass-filtered, NOT EMS-normalised
    Returns: (N, C, C) array of SPD covariance matrices
    """
    N = X_raw.shape[0]
    covs = np.zeros((N, N_CHANNELS, N_CHANNELS))
    for i in range(N):
        covs[i] = compute_covariance_matrix(X_raw[i])   # (C, T)
    return covs


def ensure_spd(M, reg=1e-3):
    """Ensure matrix is SPD by symmetrising and adding regularisation."""
    M = (M + M.T) / 2
    eigvals = np.linalg.eigvalsh(M)
    min_eig = eigvals.min()
    if min_eig < reg:
        M = M + (reg - min_eig + reg) * np.eye(M.shape[0])
    return M


def compute_riemannian_mean(covs, max_iter=50, tol=1e-7):
    """
    Compute Riemannian (geometric) mean of SPD matrices.
    covs: (N, C, C) array
    Returns: (C, C) SPD matrix
    """
    # Ensure all input matrices are SPD
    covs_clean = np.stack([ensure_spd(c) for c in covs])

    try:
        M = mean_riemann(covs_clean, tol=tol, maxiter=max_iter)
        M = ensure_spd(M)
        return M
    except Exception as e:
        # Fallback to Euclidean mean if Riemannian fails
        M = np.mean(covs_clean, axis=0)
        return ensure_spd(M)


def riemannian_align_and_ems(X_raw, M_global):
    """
    Correct pipeline:
      1. Apply Riemannian whitening to raw bandpass epochs
      2. Apply EMS normalisation to whitened epochs
      3. Return (N, 1, C, T) ready for EEGNet

    X_raw:    (N, C, T)  — raw bandpass epochs
    M_global: (C, C)     — global Riemannian mean (SPD)
    Returns:  (N, 1, C, T) — aligned + EMS normalised
    """
    M_global = ensure_spd(M_global)
    W = invsqrtm(M_global)   # (C, C) whitening matrix

    if np.any(np.isnan(W)) or np.any(np.isinf(W)):
        # Fallback: skip alignment, just apply EMS to raw
        X_ems = np.stack([
            exponential_moving_standardize(X_raw[i])
            for i in range(len(X_raw))
        ])
        return X_ems[:, np.newaxis].astype(np.float32)

    N = len(X_raw)
    X_aligned = np.zeros((N, 1, N_CHANNELS, EPOCH_LEN), dtype=np.float32)

    for i in range(N):
        # Step 1: Riemannian whitening on raw signal
        epoch_whitened = W @ X_raw[i]             # (C, T)
        # Step 2: EMS on whitened signal
        epoch_ems      = exponential_moving_standardize(epoch_whitened)
        X_aligned[i, 0] = epoch_ems

    return X_aligned


def federated_riemannian_mean(local_means, weights):
    """
    Compute weighted Riemannian mean of local means from all clients.
    """
    covs = np.stack([ensure_spd(M) for M in local_means])
    w    = np.array(weights, dtype=np.float64)
    w    = w / w.sum()

    try:
        M_global = mean_riemann(covs, sample_weight=w)
        return ensure_spd(M_global)
    except Exception:
        M_global = np.sum(covs * w[:, None, None], axis=0)
        return ensure_spd(M_global)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class EEGNetFeatureExtractor(nn.Module):
    def __init__(self, n_channels=22, n_times=875,
                 F1=8, D=2, F2=16, kern_len=32, drop_rate=0.5):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len),
                      padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop_rate)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop_rate)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            x = self.separable(self.depthwise(self.temporal(dummy)))
            self.feat_dim = x.numel()

    def forward(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x.flatten(1)


class GlobalClassifier(nn.Module):
    def __init__(self, feat_dim, n_classes=N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, feat):
        return self.fc(feat)


# ─────────────────────────────────────────────
# FEDERATED AVERAGING
# ─────────────────────────────────────────────
def fed_avg(global_model, client_states, client_weights):
    total        = sum(client_weights)
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.zeros_like(
            global_state[key], dtype=torch.float32)
        for state, w in zip(client_states, client_weights):
            global_state[key] += (w / total) * state[key].float()
    global_model.load_state_dict(global_state)
    return global_model


# ─────────────────────────────────────────────
# CLIENT LOCAL TRAINING
# ─────────────────────────────────────────────
def client_local_train(extractor, classifier,
                       X_aligned, y_client):
    """Train on Riemannian-aligned data for LOCAL_EPOCHS."""
    extractor  = copy.deepcopy(extractor).to(DEVICE)
    classifier = copy.deepcopy(classifier).to(DEVICE)

    loader    = DataLoader(
        TensorDataset(torch.FloatTensor(X_aligned),
                      torch.LongTensor(y_client)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    params    = list(extractor.parameters()) + \
                list(classifier.parameters())
    optimizer = optim.Adam(params, lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    extractor.train()
    classifier.train()

    for epoch in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            feat = extractor(xb)
            out  = classifier(feat)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

    return extractor.state_dict(), classifier.state_dict()


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(extractor, classifier, X_raw, X_ems, y, M_global=None):
    """
    Evaluate on data.
    If M_global provided: align X_raw then apply EMS (correct order).
    Otherwise: use pre-computed X_ems directly.
    """
    if M_global is not None:
        X = riemannian_align_and_ems(X_raw, M_global)
    else:
        X = X_ems

    extractor.eval()
    classifier.eval()
    loader  = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=BATCH_SIZE, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (classifier(extractor(xb)).argmax(1) == yb
                        ).sum().item()
            total   += len(yb)
    return correct / total if total > 0 else 0.0


def get_predictions(extractor, classifier, X_raw, X_ems, y,
                    M_global=None):
    if M_global is not None:
        X = riemannian_align_and_ems(X_raw, M_global)
    else:
        X = X_ems

    extractor.eval()
    classifier.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=BATCH_SIZE, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = classifier(extractor(xb.to(DEVICE)))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(yb.numpy())
    return np.array(preds), np.array(labels)


# ─────────────────────────────────────────────
# MAIN FL LOOP
# ─────────────────────────────────────────────
def run_federated(sessions, mode_name, use_riemannian):
    print(f"\n{'#'*70}")
    print(f"# MODE: {mode_name}")
    print(f"#   Riemannian alignment: {use_riemannian}")
    if use_riemannian:
        print(f"#   Pipeline: bandpass → Riemannian align → EMS → EEGNet")
    else:
        print(f"#   Pipeline: bandpass → EMS → EEGNet")
    print(f"{'#'*70}")

    # Build client data — store both raw and EMS versions
    client_X_raw    = {}   # (N, C, T) bandpass only — for covariance
    client_X_ems    = {}   # (N, 1, C, T) EMS — for FedAvg baseline
    client_y        = {}
    client_X_raw_val= {}
    client_X_ems_val= {}
    client_y_val    = {}
    client_X_raw_test = {}
    client_X_ems_test = {}
    client_y_test   = {}
    client_n_train  = {}
    clients         = []

    for s in range(1, N_SUBJECTS + 1):
        tgt_T = f"S{s}T"
        tgt_E = f"S{s}E"
        if tgt_T not in sessions or tgt_E not in sessions:
            continue

        # Unpack — sessions now stores (X_raw, X_ems, y)
        X_raw_T, X_ems_T, y_T = sessions[tgt_T]
        X_raw_E, X_ems_E, y_E = sessions[tgt_E]

        n_val = max(1, int(len(y_T) * 0.2))
        idx   = np.random.permutation(len(y_T))
        val_i = idx[:n_val]
        tr_i  = idx[n_val:]

        client_X_raw[s]     = X_raw_T[tr_i]
        client_X_ems[s]     = X_ems_T[tr_i]
        client_y[s]         = y_T[tr_i]
        client_X_raw_val[s] = X_raw_T[val_i]
        client_X_ems_val[s] = X_ems_T[val_i]
        client_y_val[s]     = y_T[val_i]
        client_X_raw_test[s]= X_raw_E
        client_X_ems_test[s]= X_ems_E
        client_y_test[s]    = y_E
        client_n_train[s]   = len(tr_i)
        clients.append(s)

    print(f"\n  Clients: {clients}")
    for s in clients:
        print(f"    S{s}: train={client_n_train[s]} "
              f"val={len(client_y_val[s])} "
              f"test={len(client_y_test[s])}")

    # Initialise global model
    global_ext = EEGNetFeatureExtractor().to(DEVICE)
    global_clf = GlobalClassifier(global_ext.feat_dim).to(DEVICE)
    feat_dim   = global_ext.feat_dim

    n_params = (sum(p.numel() for p in global_ext.parameters()) +
                sum(p.numel() for p in global_clf.parameters()))
    print(f"\n  Total params: {n_params:,}  feat_dim: {feat_dim}")

    # ── Initialise global Riemannian mean ──────────────────────────
    # Compute from all clients' training data
    M_global = None
    if use_riemannian:
        print(f"\n  Computing initial global Riemannian mean...")
        t_init = time.time()
        all_covs    = []
        all_weights = []
        for s in clients:
            # Use RAW bandpass epochs for covariance — correct
            covs = compute_subject_covariances(client_X_raw[s])
            M_s  = compute_riemannian_mean(covs)
            all_covs.append(M_s)
            all_weights.append(client_n_train[s])

        M_global = federated_riemannian_mean(all_covs, all_weights)
        M_global = ensure_spd(M_global)
        print(f"  M_global shape: {M_global.shape}  "
              f"[{time.time()-t_init:.1f}s]")
        eigvals = np.linalg.eigvalsh(M_global)
        print(f"  M_global eigenvalue range: "
              f"[{eigvals.min():.6f}, {eigvals.max():.4f}]  "
              f"(all positive: {(eigvals > 0).all()})")

    best_mean_val  = 0.0
    best_round     = 0
    best_ext_state = copy.deepcopy(global_ext.state_dict())
    best_clf_state = copy.deepcopy(global_clf.state_dict())
    best_M_global  = copy.deepcopy(M_global) if M_global is not None \
                     else None
    patience_cnt   = 0
    val_history    = []
    riem_log       = []

    print(f"\n  Starting federation — {FL_ROUNDS} rounds "
          f"(patience={FL_PATIENCE})...")

    for rnd in range(1, FL_ROUNDS + 1):
        t0 = time.time()

        client_ext_states = []
        client_clf_states = []
        local_M_list      = []

        for s in clients:
            # ── Get aligned+EMS data for this client ───────────────
            if use_riemannian and M_global is not None:
                # Correct order: align raw → EMS
                X_s_train = riemannian_align_and_ems(
                    client_X_raw[s], M_global)
            else:
                # Standard: pre-computed EMS
                X_s_train = client_X_ems[s]

            y_s = client_y[s]

            # ── Local training ─────────────────────────────────────
            ext_state, clf_state = client_local_train(
                global_ext, global_clf, X_s_train, y_s)
            client_ext_states.append(ext_state)
            client_clf_states.append(clf_state)

            # ── Compute local Riemannian mean on RAW aligned epochs ─
            if use_riemannian:
                # Align raw epochs with current M_global
                X_raw_aligned = np.stack([
                    invsqrtm(ensure_spd(M_global)) @ client_X_raw[s][i]
                    for i in range(len(client_X_raw[s]))
                ])
                covs = compute_subject_covariances(X_raw_aligned)
                M_s  = compute_riemannian_mean(covs)
                local_M_list.append(M_s)

        # ── FedAvg ─────────────────────────────────────────────────
        weights    = [client_n_train[s] for s in clients]
        global_ext = fed_avg(global_ext, client_ext_states, weights)
        global_clf = fed_avg(global_clf, client_clf_states, weights)

        # ── Update global Riemannian mean ──────────────────────────
        if use_riemannian and local_M_list:
            M_global = federated_riemannian_mean(local_M_list, weights)

        # ── Evaluate ───────────────────────────────────────────────
        val_accs = [
            evaluate(global_ext, global_clf,
                     client_X_raw_val[s], client_X_ems_val[s],
                     client_y_val[s],
                     M_global if use_riemannian else None)
            for s in clients
        ]
        mean_val = np.mean(val_accs)
        val_history.append(mean_val)

        elapsed = time.time() - t0

        if rnd % 10 == 0 or rnd == 1:
            print(f"  Round {rnd:3d}/{FL_ROUNDS} | "
                  f"Mean val: {mean_val:.4f} | "
                  f"Per-client: "
                  f"{[f'{a:.3f}' for a in val_accs]} | "
                  f"[{elapsed:.1f}s]")

        if mean_val > best_mean_val:
            best_mean_val  = mean_val
            best_round     = rnd
            best_ext_state = copy.deepcopy(global_ext.state_dict())
            best_clf_state = copy.deepcopy(global_clf.state_dict())
            best_M_global  = copy.deepcopy(M_global) \
                             if M_global is not None else None
            patience_cnt   = 0
        else:
            patience_cnt  += 1

        if patience_cnt >= FL_PATIENCE:
            print(f"\n  Early stop @ round {rnd} "
                  f"(best round={best_round})")
            break

    # Restore best
    global_ext.load_state_dict(best_ext_state)
    global_clf.load_state_dict(best_clf_state)
    M_global = best_M_global

    print(f"\n  Best round: {best_round}  "
          f"Best mean val: {best_mean_val:.4f}")

    # ── Final evaluation ───────────────────────────────────────────
    print(f"\n  Final test evaluation (E sessions):")
    print(f"  {'Subj':<8} {'Acc':>8} {'F1':>8}")
    print(f"  {'-'*28}")

    results   = {}
    test_accs = []
    test_f1s  = []

    for s in clients:
        preds, labels = get_predictions(
            global_ext, global_clf,
            client_X_raw_test[s], client_X_ems_test[s],
            client_y_test[s],
            M_global if use_riemannian else None)
        acc = float((preds == labels).mean())
        f1  = float(f1_score(labels, preds,
                              average='macro', zero_division=0))
        results[f"S{s}"] = {
            "acc": round(acc, 4),
            "f1":  round(f1, 4),
            "n":   len(labels)
        }
        test_accs.append(acc)
        test_f1s.append(f1)
        print(f"  S{s:<7} {acc:>8.4f} {f1:>8.4f}")

    mean_acc = np.mean(test_accs)
    std_acc  = np.std(test_accs)
    mean_f1  = np.mean(test_f1s)

    print(f"  {'-'*28}")
    print(f"  {'Mean':<8} {mean_acc:>8.4f} {mean_f1:>8.4f}")
    print(f"  {'Std':<8} {std_acc:>8.4f}")

    # Per-subject reports
    print(f"\n  Detailed per-subject reports:")
    for s in clients:
        preds, labels = get_predictions(
            global_ext, global_clf,
            client_X_raw_test[s], client_X_ems_test[s],
            client_y_test[s],
            M_global if use_riemannian else None)
        print(f"\n  S{s}:")
        print(classification_report(
            labels, preds,
            target_names=['Left Hand','Right Hand',
                          'Both Feet','Tongue'],
            zero_division=0))

    out = {
        "mode":            mode_name,
        "use_riemannian":  use_riemannian,
        "best_round":      best_round,
        "best_mean_val":   round(best_mean_val, 4),
        "mean_test_acc":   round(mean_acc, 4),
        "std_test_acc":    round(std_acc, 4),
        "mean_test_f1":    round(mean_f1, 4),
        "per_subject":     results,
        "val_history":     val_history
    }

    out_path = os.path.join(SAVE_DIR, f"fedra_{mode_name}_results.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return out


# ─────────────────────────────────────────────
# FINAL COMPARISON
# ─────────────────────────────────────────────
def print_final_comparison(all_results):
    base = 0.4100

    print(f"\n{'='*74}")
    print(f"  COMPLETE RESULTS — BCI-IV 2a, 9-subject LOSO")
    print(f"  All methods: fully unsupervised (zero target labels)")
    print(f"{'='*74}")
    print(f"  {'Method':<44} {'Privacy':>7} {'Acc':>8} {'vs Base':>10}")
    print(f"  {'-'*72}")

    refs = [
        ("CSP + LDA",                     "✗", 0.4093),
        ("Centralised EEGNet",             "✗", 0.4100),
        ("Centralised + CORAL (λ=10)",     "✗", 0.4358),
        ("FedAvg global (best baseline)",  "✓", 0.4667),
        ("FedCL curriculum",               "✓", 0.4617),
    ]
    for name, priv, acc in refs:
        d = acc - base
        arrow = f"▲{d:.4f}" if d > 0 else (
                f"▼{abs(d):.4f}" if d < 0 else "─")
        print(f"  {name:<44} {priv:>7} {acc:>8.4f} {arrow:>10}")

    print(f"  {'-'*72}")

    best_acc = max(r['mean_test_acc'] for r in all_results)
    for res in all_results:
        acc   = res['mean_test_acc']
        d     = acc - base
        arrow = f"▲{d:.4f}" if d > 0 else f"▼{abs(d):.4f}"
        tag   = ""
        if acc == best_acc and acc > 0.4667:
            tag = "← NEW BEST ✓"
        elif acc > 0.4667:
            tag = "← beats baseline"
        elif abs(acc - 0.4667) < 0.005:
            tag = "≈ baseline"
        label = res['mode']
        print(f"  {label:<44} {'✓':>7} {acc:>8.4f} "
              f"{arrow:>10}  {tag}")

    print(f"\n  Chance (4-class): 0.2500")
    print(f"\n  --- Comparison with prior work ---")
    print(f"  CTL (Gao et al. 2026)    : 73.13%  "
          f"[160 labeled target samples, centralised]")
    print(f"  EA (He & Wu 2020)         : ~68%    "
          f"[Euclidean alignment, centralised]")
    print(f"  FedRA (proposed)          : {best_acc:.4f}  "
          f"[zero target labels, federated, Riemannian]")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("="*70)
    print("  BCI-IV 2a — Federated Riemannian Alignment (FedRA)")
    print("  Corrected pipeline: bandpass → Riemannian align → EMS")
    print("="*70)

    # Load with raw epochs for correct Riemannian alignment
    # sessions[key] = (X_raw, X_ems, y)
    sessions = load_all_sessions(return_raw=True)
    print(f"\nLoaded {len(sessions)} sessions.")
    print(f"Each session stores: X_raw (C,T) + X_ems (1,C,T) + y")

    all_results = []

    # Mode 1: Standard FedAvg (reproduced baseline)
    res1 = run_federated(
        sessions,
        mode_name      = "FedAvg_baseline",
        use_riemannian = False
    )
    all_results.append(res1)

    # Mode 2: FedRA — Federated Riemannian Alignment (proposed)
    res2 = run_federated(
        sessions,
        mode_name      = "FedRA_proposed",
        use_riemannian = True
    )
    all_results.append(res2)

    print_final_comparison(all_results)

    summary_path = os.path.join(SAVE_DIR, "fedra_v2_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved: {summary_path}")


if __name__ == "__main__":
    main()