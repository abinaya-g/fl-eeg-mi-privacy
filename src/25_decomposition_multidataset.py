"""
ALIGNMENT DECOMPOSITION ON A SECOND DATASET (genuine LOSO protocol)
====================================================================
Purpose: replicate the paper's decomposition (FedAvg vs. transductive /
inductive Euclidean alignment vs. shared / per-subject Riemannian
alignment) on a second public MI dataset, with the same pipeline,
model and training protocol as the BCI-IV 2a scripts.

Arms (--arm):
  fedavg        No alignment; EMS only (04_/05_fedavg_global_reverify_*).
  ea            Fed-EA, transductive: every subject, including the
                held-out target, is whitened by W_s = R_s^(-1/2), R_s the
                mean of X X^T over its own trials (06_/07_fedea_*).
  ea_inductive  Fed-EA, inductive: sources as in 'ea'; the target uses the
                unweighted mean of the source clients' W_s (13_/14_*).
  ra_global     FedRA: federated Riemannian (Karcher) mean of all
                trace-normalised source-trial covariances, one shared
                whitening for sources and target (22_, --arm global).
  ra_local      FedRA-local: per-subject Riemannian re-centring, target
                on its own unlabelled trials (22_, --arm local).

Datasets (--dataset):
  bnci2014002   BNCI 2014-002 (Steyrl et al. 2016): 14 subjects, 15 ch,
                512 Hz, right hand vs. feet. T session = 5 runs (100
                trials), E session = 3 runs (60 trials). Window 3.5-7.0 s
                after trial start (cue at 3 s), i.e. the same 0.5-4.0 s
                post-cue, 3.5 s window as for BCI-IV 2a; resampled to
                250 Hz so EEGNet sees the same 875-sample input.
  bci2a         BCI Competition IV-2a (BNCI 2014-001), as in the paper.

Preprocessing, EEGNet, FedAvg, early stopping, EMS and validation split
are identical to the BCI-IV 2a scripts. Output: one JSON per fold,
<out-dir>/<dataset>_<arm>_seed<seed>_S<fold>.json.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.metrics import f1_score
import os, json, copy, time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FS_MODEL   = 250
EPOCH_LEN  = 875          # 3.5 s at 250 Hz
EOG_THRESH = 100.0

FL_ROUNDS     = 100
LOCAL_EPOCHS  = 5
FL_PATIENCE   = 20
BATCH_SIZE    = 32
LR            = 1e-3
VAL_FRACTION  = 0.15

KARCHER_MAX_ITER = 200
KARCHER_TOL      = 1e-8

DATASETS = {
    "bci2a": dict(n_subjects=9, n_channels=22, n_classes=4, fs=250,
                  t_start=2.5, t_end=6.0, fname=lambda s, k: f"A0{s}{k}.mat"),
    "bnci2014002": dict(n_subjects=14, n_channels=15, n_classes=2, fs=512,
                        t_start=3.5, t_end=7.0, fname=lambda s, k: f"S{s:02d}{k}.mat"),
}


# ─────────────────────────────────────────────
# PREPROCESSING (as in the BCI-IV 2a scripts; resampling added for 512 Hz)
# ─────────────────────────────────────────────
def bandpass(data, fs, lo=4, hi=40, order=4):
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

def load_session_raw(path, cfg):
    mat  = loadmat(path, struct_as_record=False, squeeze_me=True)
    data = mat['data']
    runs = data if hasattr(data, '__len__') else [data]
    X_list, y_list = [], []
    for run in runs:
        try:
            raw_X, raw_y, t_pos, fs_run = run.X.T, run.y, run.trial, run.fs
        except AttributeError:
            continue
        if not hasattr(raw_y, '__len__') or len(raw_y) == 0:
            continue
        eeg = bandpass(raw_X[:cfg["n_channels"]], fs_run)
        for onset, lbl in zip(t_pos, raw_y):
            if lbl < 1 or lbl > cfg["n_classes"]:
                continue
            s = int(onset + cfg["t_start"] * fs_run)
            e = int(onset + cfg["t_end"] * fs_run)
            if e > eeg.shape[1]:
                continue
            epoch = eeg[:, s:e]
            if fs_run != FS_MODEL:
                epoch = resample_poly(epoch, FS_MODEL, int(fs_run), axis=-1)
            if epoch.shape[1] != EPOCH_LEN:
                continue
            if np.max(np.abs(epoch)) > EOG_THRESH:
                continue
            X_list.append(epoch)        # no EMS yet -- alignment applied first
            y_list.append(int(lbl) - 1)
    if not X_list:
        return None, None
    return np.stack(X_list).astype(np.float64), np.array(y_list, dtype=np.int64)

def load_all_sessions_raw(data_dir, cfg):
    sessions = {}
    print("\nLoading all sessions (pre-alignment, pre-EMS)...")
    for s in range(1, cfg["n_subjects"] + 1):
        for k in ['T', 'E']:
            fpath = os.path.join(data_dir, cfg["fname"](s, k))
            if not os.path.exists(fpath):
                continue
            X, y = load_session_raw(fpath, cfg)
            if X is None:
                continue
            sessions[f"S{s}{k}"] = (X, y)
            print(f"  S{s}{k}: {len(y)} trials, classes {np.bincount(y).tolist()}")
    return sessions


# ─────────────────────────────────────────────
# ALIGNMENT
# ─────────────────────────────────────────────
def _eig_fn(M, fn):
    w, V = np.linalg.eigh((M + M.T) / 2)
    return (V * fn(w)) @ V.T

def spd_sqrt(M):     return _eig_fn(M, np.sqrt)
def spd_invsqrt(M):  return _eig_fn(M, lambda w: 1.0 / np.sqrt(w))
def spd_log(M):      return _eig_fn(M, np.log)
def spd_exp(M):      return _eig_fn(M, np.exp)

def ea_whitening(X):
    """Euclidean alignment (He & Wu 2020), as in 06_fedea_*: R = mean X X^T."""
    R = np.einsum('nct,ndt->cd', X, X) / len(X)
    return spd_invsqrt(R)

def trial_covariances(X):
    covs = np.einsum('nct,ndt->ncd', X, X)
    return covs / np.trace(covs, axis1=1, axis2=2)[:, None, None]

def federated_riemannian_mean(client_covs):
    """Karcher fixed-point iteration over client tangent sums (as in 22_)."""
    n_total = sum(len(c) for c in client_covs)
    G = sum(c.sum(0) for c in client_covs) / n_total
    nu, tau = 1.0, np.finfo(np.float64).max
    for it in range(1, KARCHER_MAX_ITER + 1):
        G_sqrt, G_isqrt = spd_sqrt(G), spd_invsqrt(G)
        T = sum(sum(spd_log(G_isqrt @ C @ G_isqrt) for C in c) for c in client_covs) / n_total
        G = G_sqrt @ spd_exp(nu * T) @ G_sqrt
        step = np.linalg.norm(T, 'fro')
        if step < KARCHER_TOL:
            break
        h = nu * step
        if h < tau:
            nu, tau = 0.95 * nu, h
        else:
            nu *= 0.5
    return G, it, step

def whiten_then_ems(X, W):
    Xw = np.einsum('cd,ndt->nct', W, X)
    out = np.zeros_like(Xw, dtype=np.float32)
    for i in range(Xw.shape[0]):
        out[i] = exponential_moving_standardize(Xw[i])
    return out[:, np.newaxis].astype(np.float32)

def alignment_matrices(arm, sessions, clients, target):
    """Returns ({source: W_s}, W_target, diagnostics)."""
    C = sessions[f"S{target}E"][0].shape[1]
    X_t = sessions[f"S{target}E"][0]
    src = {s: sessions[f"S{s}T"][0] for s in clients}
    diag = {}
    if arm == "fedavg":
        I = np.eye(C)
        return {s: I for s in clients}, I, diag
    if arm in ("ea", "ea_inductive"):
        W = {s: ea_whitening(X) for s, X in src.items()}
        W_t = ea_whitening(X_t) if arm == "ea" else np.mean([W[s] for s in clients], axis=0)
        return W, W_t, diag
    if arm == "ra_global":
        G, it, step = federated_riemannian_mean([trial_covariances(X) for X in src.values()])
        w = np.linalg.eigvalsh(G)
        diag.update(karcher_iters=it, karcher_final_step=float(step),
                    eig_ratio=float(w.max() / max(w.min(), 1e-12)))
        W_g = spd_invsqrt(G)
        return {s: W_g for s in clients}, W_g, diag
    if arm == "ra_local":
        W = {s: spd_invsqrt(federated_riemannian_mean([trial_covariances(X)])[0])
             for s, X in src.items()}
        G_t, it, step = federated_riemannian_mean([trial_covariances(X_t)])
        diag.update(target_karcher_iters=it, target_karcher_final_step=float(step))
        return W, spd_invsqrt(G_t), diag
    raise ValueError(arm)


# ─────────────────────────────────────────────
# MODEL AND FEDAVG (as in the BCI-IV 2a scripts)
# ─────────────────────────────────────────────
class EEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_times=EPOCH_LEN,
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
            feat_dim = self.separable(self.depthwise(self.temporal(dummy))).numel()
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.fc(self.separable(self.depthwise(self.temporal(x))).flatten(1))


def class_weights(y, n_classes):
    counts = np.array([max((y == c).sum(), 1) for c in range(n_classes)])
    return torch.tensor(len(y) / (n_classes * counts), dtype=torch.float32)

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).to(DEVICE)).argmax(1).cpu().numpy()
    return float((preds == y).mean()), float(f1_score(y, preds, average='macro', zero_division=0))

def client_local_train(global_state, X_train, y_train, cfg):
    model = EEGNet(cfg["n_channels"], cfg["n_classes"]).to(DEVICE)
    model.load_state_dict(global_state)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(weight=class_weights(y_train, cfg["n_classes"]).to(DEVICE))
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                        batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
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


def run_fold(sessions, cfg, target, arm, seed, max_rounds):
    np.random.seed(seed)
    torch.manual_seed(seed)
    clients = [s for s in range(1, cfg["n_subjects"] + 1)
               if s != target and f"S{s}T" in sessions]
    print(f"\n{'='*60}\n  [{arm}, seed {seed}] target S{target}, {len(clients)} clients\n{'='*60}")
    t0 = time.time()
    W, W_t, diag = alignment_matrices(arm, sessions, clients, target)
    X_target_raw, y_target = sessions[f"S{target}E"]
    X_target = whiten_then_ems(X_target_raw, W_t)

    client_data, client_val = {}, {}
    for s in clients:
        X_s_raw, y_s = sessions[f"S{s}T"]
        X_s = whiten_then_ems(X_s_raw, W[s])
        idx = np.random.RandomState(seed).permutation(len(y_s))
        n_val = max(int(len(y_s) * VAL_FRACTION), 1)
        client_data[s] = (X_s[idx[n_val:]], y_s[idx[n_val:]])
        client_val[s] = (X_s[idx[:n_val]], y_s[idx[:n_val]])

    torch.manual_seed(seed)
    global_model = EEGNet(cfg["n_channels"], cfg["n_classes"]).to(DEVICE)
    global_state = global_model.state_dict()
    weights = [len(client_data[s][0]) for s in clients]
    best_val, best_round, patience_cnt, rnd = 0.0, 0, 0, 0
    best_state = copy.deepcopy(global_state)
    for rnd in range(1, max_rounds + 1):
        states = [client_local_train(global_state, *client_data[s], cfg) for s in clients]
        global_state = fed_avg_state(states, weights)
        global_model.load_state_dict(global_state)
        mean_val = np.mean([evaluate(global_model, *client_val[s])[0] for s in clients])
        if rnd % 10 == 0 or rnd == 1:
            print(f"    round {rnd:3d} | mean val {mean_val:.4f} | {time.time() - t0:.0f}s")
        if mean_val > best_val:
            best_val, best_round, patience_cnt = mean_val, rnd, 0
            best_state = copy.deepcopy(global_state)
        else:
            patience_cnt += 1
        if patience_cnt >= FL_PATIENCE:
            break

    global_model.load_state_dict(best_state)
    acc, f1 = evaluate(global_model, X_target, y_target)
    print(f"  >> S{target}: best_val={best_val:.4f}  test_acc={acc:.4f}  f1={f1:.4f}")
    return {"arm": arm, "seed": seed, "target": f"S{target}",
            "acc": round(acc, 4), "f1": round(f1, 4),
            "best_val": round(float(best_val), 4), "best_round": best_round,
            "rounds_run": rnd, "n_target_trials": int(len(y_target)),
            "n_clients": len(clients), "runtime_s": round(time.time() - t0, 1), **diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="bnci2014002")
    ap.add_argument("--arm", choices=["fedavg", "ea", "ea_inductive", "ra_global", "ra_local"],
                    required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0, help="target subject; 0 = all")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="results/decomposition")
    ap.add_argument("--max-rounds", type=int, default=FL_ROUNDS,
                    help="smoke tests only; keep the default for real runs")
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]

    os.makedirs(args.out_dir, exist_ok=True)
    sessions = load_all_sessions_raw(args.data_dir, cfg)
    folds = range(1, cfg["n_subjects"] + 1) if args.fold == 0 else [args.fold]
    for target in folds:
        if f"S{target}E" not in sessions:
            raise SystemExit(f"Missing target session S{target}E in {args.data_dir}")
        res = run_fold(sessions, cfg, target, args.arm, args.seed, args.max_rounds)
        res["dataset"] = args.dataset
        out = os.path.join(args.out_dir, f"{args.dataset}_{args.arm}_seed{args.seed}_S{target}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
