"""
BCI-IV 2a — FEDRA RE-VERIFICATION (genuine LOSO protocol)
====================================================================
Purpose: FedRA (Riemannian alignment) was the most severe negative
result in the original ablation study (34.20%), but like FedAvg global,
FedCL, and the local-heads variants, it was computed under a
personalised protocol, not genuine cross-subject LOSO. This script
re-runs FedRA from scratch under the same verified LOSO scaffold used
throughout this project, with the corrected whitening formulation
(channel-level, not feature-level -- see docstring note below).

Protocol:
  - Each source client computes its own local mean covariance M_k
    (C x C, C=22 channels) from its own RAW T-session trial covariances,
    prior to any EEGNet feature extraction.
  - The server aggregates these into a global Riemannian mean estimate,
    M_global, via a trial-count-weighted average of the client
    covariances (a genuine cross-client COMMUNICATION step -- unlike
    Fed-EA, where each subject's whitening statistic never leaves that
    subject). This communication step is computed ONCE per LOSO fold
    (not once per round, since it depends only on raw signal statistics,
    not on the evolving model).
  - Every client whitens its own raw trials using W = M_global^(-1/2)
    BEFORE feature extraction: X_tilde = W @ X, X in R^{C x T}. This
    matches the corrected, channel-level formulation (the original
    submission's equation mistakenly described whitening a 432-dim
    feature vector with a 22x22 matrix -- a dimensional mismatch this
    script avoids by construction).
  - After whitening, standard EMS normalisation and FedAvg training
    proceed exactly as in fedavg_global_reverify.py.
  - The held-out TARGET subject is whitened using the SAME M_global
    (computed only from the 8 source clients -- the target's own data
    is never used to compute M_global). This makes FedRA, unlike
    Fed-EA, a FULLY INDUCTIVE alignment method: no target-subject data
    of any kind, labelled or unlabelled, is used to compute the
    whitening transform applied to the target.

Output: fedra_reverify_results.json with per-subject LOSO accuracy.
Compare against the corrected FedAvg-no-alignment baseline (39.84%,
2-seed average) and Fed-EA (50.37%, 2-seed average) to complete the
communication-dependent vs. communication-free contrast.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import f1_score
import os, json, copy

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/datasets/abinayajone/bci-iv-2a-mi"
SAVE_DIR   = "/kaggle/working"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FS         = 250
T_START    = 2.5
T_END      = 6.0
EPOCH_LEN  = int((T_END - T_START) * FS)
EOG_THRESH = 100.0
N_CLASSES  = 4
N_SUBJECTS = 9

FL_ROUNDS     = 100
LOCAL_EPOCHS  = 5
FL_PATIENCE   = 20
BATCH_SIZE    = 32
LR            = 1e-3
VAL_FRACTION  = 0.15

print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
# PREPROCESSING — bandpass only here; Riemannian whitening and EMS
# applied separately below
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

def load_session_raw(path):
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
        eeg = bandpass(raw_X[:22])
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
            X_list.append(epoch)   # NOTE: no EMS yet -- whitening applied first
            y_list.append(lbl - 1)
    if len(X_list) == 0:
        return None, None
    X = np.stack(X_list).astype(np.float64)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def load_all_sessions_raw():
    sessions = {}
    print("\nLoading all sessions (pre-whitening, pre-EMS)...")
    for s in range(1, N_SUBJECTS + 1):
        for sess in ['T', 'E']:
            fname = f"A0{s}{sess}.mat"
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fpath):
                continue
            X, y = load_session_raw(fpath)
            if X is None:
                continue
            sessions[f"S{s}{sess}"] = (X, y)
            print(f"  S{s}{sess}: {len(y)} trials")
    return sessions


# ─────────────────────────────────────────────
# RIEMANNIAN (FEDRA) WHITENING — computed once per fold via a
# genuine cross-client aggregation step (unlike Fed-EA's purely local
# statistic). This IS the communication-dependent step this experiment
# is designed to test.
# ─────────────────────────────────────────────
def client_local_covariance(X):
    """X: (n_trials, C, T) for one client's own trials.
    Returns the client's local mean covariance, normalised by trace
    (standard Riemannian-alignment convention) -- this is the
    statistic that gets COMMUNICATED to the server."""
    n, C, T = X.shape
    M = np.zeros((C, C))
    for i in range(n):
        cov = X[i] @ X[i].T
        cov /= np.trace(cov)   # normalise before averaging
        M += cov
    M /= n
    return M

def aggregate_global_mean(client_covs, client_weights):
    """Server-side aggregation: trial-count-weighted arithmetic mean of
    client covariances, approximating the global Riemannian mean. This
    communication step is what distinguishes FedRA from Fed-EA."""
    total = sum(client_weights)
    M_global = sum(cov * (w / total) for cov, w in zip(client_covs, client_weights))
    return M_global

def compute_whitening(M_global):
    return fractional_matrix_power(M_global, -0.5).real

def apply_whitening(X, W):
    return np.einsum('cd,ndt->nct', W, X)

def whiten_then_ems(X, W):
    X_whitened = apply_whitening(X, W)
    X_out = np.zeros_like(X_whitened, dtype=np.float32)
    for i in range(X_whitened.shape[0]):
        X_out[i] = exponential_moving_standardize(X_whitened[i])
    return X_out[:, np.newaxis].astype(np.float32)


# ─────────────────────────────────────────────
# MODEL — identical EEGNet used throughout this project
# ─────────────────────────────────────────────
class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_times=875, n_classes=N_CLASSES,
                 F1=8, D=2, F2=16, kern_len=32, drop_rate=0.5):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop_rate)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop_rate)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            x = self.separable(self.depthwise(self.temporal(dummy)))
            feat_dim = x.numel()
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.flatten(1)
        return self.fc(x)


def class_weights(y, n_classes=N_CLASSES):
    n = len(y)
    counts = np.array([max((y == c).sum(), 1) for c in range(n_classes)])
    w = n / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).to(DEVICE)
        preds = model(Xt).argmax(1).cpu().numpy()
    acc = float((preds == y).mean())
    f1 = float(f1_score(y, preds, average='macro', zero_division=0))
    return acc, f1


def client_local_train(global_state, X_train, y_train, local_epochs=LOCAL_EPOCHS):
    model = EEGNet().to(DEVICE)
    model.load_state_dict(global_state)
    opt = optim.Adam(model.parameters(), lr=LR)
    cw = class_weights(y_train).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model.state_dict()


def fed_avg_state(states, weights):
    total = sum(weights)
    avg = copy.deepcopy(states[0])
    for k in avg.keys():
        if avg[k].dtype.is_floating_point:
            avg[k] = sum(states[i][k] * (weights[i] / total) for i in range(len(states)))
        else:
            avg[k] = states[0][k]
    return avg


# ─────────────────────────────────────────────
# LOSO — FedRA (resume-aware)
# ─────────────────────────────────────────────
def load_existing_results():
    path = os.path.join(SAVE_DIR, "fedra_reverify_results.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            print(f"\n  Found existing results — resuming. {len(data)}/9 done.")
            return data
        except Exception:
            pass
    return {}


def run_loso(sessions):
    results = load_existing_results()

    for target in range(1, N_SUBJECTS + 1):
        if f"S{target}" in results:
            print(f"\n  S{target}: already done, skipping.")
            continue

        print(f"\n{'='*60}\n  LOSO fold: target = S{target}\n{'='*60}")
        clients = [s for s in range(1, N_SUBJECTS + 1) if s != target]

        Xt_key = f"S{target}E"
        if Xt_key not in sessions:
            continue
        X_target_raw, y_target = sessions[Xt_key]

        # --- Step 1: each client computes its own local covariance ---
        client_raw = {}
        client_covs, client_weights = [], []
        for s in clients:
            key = f"S{s}T"
            if key not in sessions:
                continue
            X_s_raw, y_s = sessions[key]
            client_raw[s] = (X_s_raw, y_s)
            client_covs.append(client_local_covariance(X_s_raw))
            client_weights.append(len(y_s))

        # --- Step 2: server aggregates -> global Riemannian mean ---
        M_global = aggregate_global_mean(client_covs, client_weights)
        eigvals = np.linalg.eigvalsh(M_global)
        print(f"  M_global eigenvalue range: [{eigvals.min():.4f}, {eigvals.max():.4f}] "
              f"(ratio {eigvals.max()/max(eigvals.min(), 1e-12):.1f}x)")
        W = compute_whitening(M_global)

        # --- Step 3: every client (and the target) whitens using W ---
        X_target = whiten_then_ems(X_target_raw, W)

        client_data, client_val = {}, {}
        for s in clients:
            X_s_raw, y_s = client_raw[s]
            X_s = whiten_then_ems(X_s_raw, W)
            n = len(y_s)
            idx = np.random.RandomState(42).permutation(n)
            n_val = max(int(n * VAL_FRACTION), 1)
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            client_data[s] = (X_s[train_idx], y_s[train_idx])
            client_val[s]  = (X_s[val_idx], y_s[val_idx])

        # --- Step 4: standard FedAvg training on whitened data ---
        torch.manual_seed(42)
        global_model = EEGNet().to(DEVICE)
        global_state = global_model.state_dict()
        weights = [len(client_data[s][0]) for s in clients]

        best_val = 0.0
        best_state = copy.deepcopy(global_state)
        patience_cnt = 0

        for rnd in range(1, FL_ROUNDS + 1):
            client_states = []
            for s in clients:
                X_tr, y_tr = client_data[s]
                st = client_local_train(global_state, X_tr, y_tr)
                client_states.append(st)

            global_state = fed_avg_state(client_states, weights)
            global_model.load_state_dict(global_state)

            val_accs = []
            for s in clients:
                Xv, yv = client_val[s]
                acc, _ = evaluate(global_model, Xv, yv)
                val_accs.append(acc)
            mean_val = np.mean(val_accs)

            if rnd % 20 == 0 or rnd == 1:
                print(f"    round {rnd:3d}/{FL_ROUNDS} | mean val: {mean_val:.4f}")

            if mean_val > best_val:
                best_val = mean_val
                best_state = copy.deepcopy(global_state)
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= FL_PATIENCE:
                print(f"    early stop @ round {rnd}")
                break

        global_model.load_state_dict(best_state)
        acc, f1 = evaluate(global_model, X_target, y_target)
        print(f"  >> S{target}: best_val={best_val:.4f}  test_acc={acc:.4f}  f1={f1:.4f}")

        results[f"S{target}"] = {
            "acc": round(acc, 4), "f1": round(f1, 4),
            "eig_min": float(eigvals.min()), "eig_max": float(eigvals.max()),
        }
        with open(os.path.join(SAVE_DIR, "fedra_reverify_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    return results


def print_summary(results):
    print(f"\n{'='*70}\n  FEDRA RE-VERIFICATION — LOSO SUMMARY\n{'='*70}")
    accs = [v["acc"] for v in results.values()]
    f1s = [v["f1"] for v in results.values()]
    for s, r in sorted(results.items(), key=lambda kv: int(kv[0][1:])):
        print(f"  {s:<6} acc={r['acc']:.4f}  f1={r['f1']:.4f}  "
              f"eig=[{r['eig_min']:.3f},{r['eig_max']:.3f}]")
    print(f"  {'-'*30}")
    print(f"  Mean   acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}")
    print(f"  Std    acc={np.std(accs):.4f}")
    print(f"\n  Compare to (all genuine LOSO, verified this project):")
    print(f"    FedAvg (no alignment), 2-seed avg : 39.84%")
    print(f"    Fed-EA (proposed), 2-seed avg      : 50.37%")
    print(f"    Original submission's FedRA claim  : 34.20% (personalised protocol, unverified)")
    print(f"\n  This completes the two-example communication-dependent")
    print(f"  contrast (CORAL, FedRA) against communication-free Fed-EA.")


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    sessions = load_all_sessions_raw()
    print(f"\nLoaded {len(sessions)} sessions.")
    results = run_loso(sessions)
    print_summary(results)
    print(f"\nFull results saved: {os.path.join(SAVE_DIR, 'fedra_reverify_results.json')}")


if __name__ == "__main__":
    main()