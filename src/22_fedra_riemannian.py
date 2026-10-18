"""
BCI-IV 2a — FEDRA WITH A TRUE RIEMANNIAN MEAN (genuine LOSO protocol)
====================================================================
Purpose: correct the FedRA comparator. 09_fedra_reverification.py
aggregated client covariances with a trial-count-weighted ARITHMETIC
mean, which is not the Riemannian (affine-invariant / Karcher) mean of
the cited Riemannian-alignment method (Zanini et al., 2018). This
script re-runs FedRA with the genuine Riemannian mean, in two arms:

  --arm global   Federated Riemannian mean, shared whitening (inductive).
                 The server computes the Karcher mean of ALL source-trial
                 covariances by federated fixed-point iteration: it
                 broadcasts the current estimate G, each client returns
                 the sum of its trials' tangent vectors log_G(C_i) and
                 its trial count, and the server updates
                 G <- G^(1/2) expm(mean tangent) G^(1/2). Only C x C
                 summaries leave a client; this is the exact pooled
                 Riemannian mean (checked against pyriemann). Every
                 client AND the held-out target are whitened with
                 W = G^(-1/2). Target data is never used -> inductive.
                 This is 09's protocol with the mean fixed.

  --arm local    Per-subject Riemannian re-centring (Zanini et al. 2018
                 as originally formulated). Each subject is whitened by
                 the inverse square root of the Riemannian mean of its
                 OWN trials; the target uses its own unlabelled E-session
                 trials (transductive, exactly like Fed-EA). No
                 cross-client communication. This is the Riemannian
                 analogue of Fed-EA (which uses the arithmetic mean).

Everything else (preprocessing, EEGNet, FedAvg, early stopping, EMS,
validation split) is copied unchanged from 09_fedra_reverification.py.

Usage:
  python src/22_fedra_riemannian.py --arm global --seed 42 --fold 3 \
      --data-dir data/bci2a --out-dir results/fedra_riemann
  --fold 0 runs all nine folds sequentially.

Output: <out-dir>/<arm>_seed<seed>_S<fold>.json, one file per fold.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score
import os, json, copy, time

# ─────────────────────────────────────────────
# CONFIG (identical to 09_fedra_reverification.py)
# ─────────────────────────────────────────────
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

KARCHER_MAX_ITER = 200
KARCHER_TOL      = 1e-8


# ─────────────────────────────────────────────
# PREPROCESSING (unchanged from 09)
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
            X_list.append(epoch)
            y_list.append(lbl - 1)
    if len(X_list) == 0:
        return None, None
    X = np.stack(X_list).astype(np.float64)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def load_all_sessions_raw(data_dir):
    sessions = {}
    print("\nLoading all sessions (pre-whitening, pre-EMS)...")
    for s in range(1, N_SUBJECTS + 1):
        for sess in ['T', 'E']:
            fpath = os.path.join(data_dir, f"A0{s}{sess}.mat")
            if not os.path.exists(fpath):
                continue
            X, y = load_session_raw(fpath)
            if X is None:
                continue
            sessions[f"S{s}{sess}"] = (X, y)
            print(f"  S{s}{sess}: {len(y)} trials")
    return sessions


# ─────────────────────────────────────────────
# RIEMANNIAN GEOMETRY (affine-invariant metric, SPD matrices)
# ─────────────────────────────────────────────
def _eig_fn(M, fn):
    w, V = np.linalg.eigh((M + M.T) / 2)
    return (V * fn(w)) @ V.T

def spd_sqrt(M):     return _eig_fn(M, np.sqrt)
def spd_invsqrt(M):  return _eig_fn(M, lambda w: 1.0 / np.sqrt(w))
def spd_log(M):      return _eig_fn(M, np.log)
def spd_exp(M):      return _eig_fn(M, np.exp)

def trial_covariances(X):
    """X: (n_trials, C, T). Trace-normalised trial covariances, the same
    per-trial statistic 09 used before averaging."""
    covs = np.einsum('nct,ndt->ncd', X, X)
    return covs / np.trace(covs, axis1=1, axis2=2)[:, None, None]

def client_tangent_sum(covs, G_invsqrt):
    """CLIENT side: sum over own trials of log(G^-1/2 C_i G^-1/2).
    Only this C x C matrix (and the trial count) is sent to the server."""
    T = np.zeros_like(G_invsqrt)
    for C in covs:
        T += spd_log(G_invsqrt @ C @ G_invsqrt)
    return T

def federated_riemannian_mean(client_covs):
    """SERVER side: Karcher fixed-point iteration over federated tangent
    sums. Initialised at the arithmetic mean (09's estimate), with the
    same adaptive step size as pyriemann.utils.mean.mean_riemann. Returns
    the Riemannian mean of the pooled source trials."""
    n_total = sum(len(c) for c in client_covs)
    G = sum(c.sum(0) for c in client_covs) / n_total
    nu, tau = 1.0, np.finfo(np.float64).max
    for it in range(1, KARCHER_MAX_ITER + 1):
        G_sqrt, G_isqrt = spd_sqrt(G), spd_invsqrt(G)
        T = sum(client_tangent_sum(c, G_isqrt) for c in client_covs) / n_total
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

def riemannian_mean(covs):
    """Riemannian mean of one subject's own trials (local arm)."""
    return federated_riemannian_mean([covs])

def check_against_pyriemann(covs, G):
    """Optional sanity check: our mean vs pyriemann's mean_riemann."""
    try:
        from pyriemann.utils.mean import mean_riemann
    except ImportError:
        return None
    ref = mean_riemann(covs, tol=1e-10, maxiter=200)
    return float(np.linalg.norm(G - ref, 'fro') / np.linalg.norm(ref, 'fro'))

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
# ONE LOSO FOLD
# ─────────────────────────────────────────────
def eig_summary(M):
    w = np.linalg.eigvalsh(M)
    return {"eig_min": float(w.min()), "eig_max": float(w.max()),
            "eig_ratio": float(w.max() / max(w.min(), 1e-12))}

def run_fold(sessions, target, arm, seed, max_rounds=FL_ROUNDS):
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"\n{'='*60}\n  [{arm}, seed {seed}] LOSO fold: target = S{target}\n{'='*60}")
    clients = [s for s in range(1, N_SUBJECTS + 1)
               if s != target and f"S{s}T" in sessions]
    X_target_raw, y_target = sessions[f"S{target}E"]
    t0 = time.time()

    # --- Alignment statistics ---
    client_covs = {s: trial_covariances(sessions[f"S{s}T"][0]) for s in clients}
    diag = {}
    if arm == "global":
        G, n_it, step = federated_riemannian_mean([client_covs[s] for s in clients])
        W_global = spd_invsqrt(G)
        W = {s: W_global for s in clients}
        W_target = W_global
        diag.update(eig_summary(G))
        diag.update({"karcher_iters": n_it, "karcher_final_step": step})
        diag["rel_diff_vs_pyriemann"] = check_against_pyriemann(
            np.concatenate([client_covs[s] for s in clients]), G)
        arith = sum(c.sum(0) for c in client_covs.values()) / sum(len(c) for c in client_covs.values())
        diag["rel_diff_vs_arithmetic"] = float(
            np.linalg.norm(G - arith, 'fro') / np.linalg.norm(arith, 'fro'))
        print(f"  Federated Karcher mean: {n_it} iters, final step {step:.2e}, "
              f"eig ratio {diag['eig_ratio']:.1f}x, "
              f"vs pyriemann {diag['rel_diff_vs_pyriemann']}, "
              f"vs arithmetic {diag['rel_diff_vs_arithmetic']:.4f}")
    elif arm == "local":
        W = {}
        for s in clients:
            M_s, _, _ = riemannian_mean(client_covs[s])
            W[s] = spd_invsqrt(M_s)
        M_t, n_it, step = riemannian_mean(trial_covariances(X_target_raw))
        W_target = spd_invsqrt(M_t)
        diag.update({f"target_{k}": v for k, v in eig_summary(M_t).items()})
        diag.update({"target_karcher_iters": n_it, "target_karcher_final_step": step})
        print(f"  Per-subject Riemannian re-centring; target eig ratio "
              f"{diag['target_eig_ratio']:.1f}x ({n_it} Karcher iters)")
    else:
        raise ValueError(arm)

    X_target = whiten_then_ems(X_target_raw, W_target)

    client_data, client_val = {}, {}
    for s in clients:
        X_s_raw, y_s = sessions[f"S{s}T"]
        X_s = whiten_then_ems(X_s_raw, W[s])
        n = len(y_s)
        idx = np.random.RandomState(seed).permutation(n)
        n_val = max(int(n * VAL_FRACTION), 1)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        client_data[s] = (X_s[train_idx], y_s[train_idx])
        client_val[s]  = (X_s[val_idx], y_s[val_idx])

    # --- Standard FedAvg training on aligned data ---
    torch.manual_seed(seed)
    global_model = EEGNet().to(DEVICE)
    global_state = global_model.state_dict()
    weights = [len(client_data[s][0]) for s in clients]

    best_val, best_round = 0.0, 0
    best_state = copy.deepcopy(global_state)
    patience_cnt = 0
    rnd = 0
    for rnd in range(1, max_rounds + 1):
        client_states = [client_local_train(global_state, *client_data[s]) for s in clients]
        global_state = fed_avg_state(client_states, weights)
        global_model.load_state_dict(global_state)

        mean_val = np.mean([evaluate(global_model, *client_val[s])[0] for s in clients])
        if rnd % 10 == 0 or rnd == 1:
            print(f"    round {rnd:3d}/{max_rounds} | mean val: {mean_val:.4f} "
                  f"| {time.time() - t0:.0f}s")

        if mean_val > best_val:
            best_val, best_round = mean_val, rnd
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

    return {
        "arm": arm, "seed": seed, "target": f"S{target}",
        "acc": round(acc, 4), "f1": round(f1, 4),
        "best_val": round(float(best_val), 4), "best_round": best_round,
        "rounds_run": rnd, "n_target_trials": int(len(y_target)),
        "runtime_s": round(time.time() - t0, 1),
        **diag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["global", "local"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0, help="target subject 1-9; 0 = all")
    ap.add_argument("--data-dir", default="data/bci2a")
    ap.add_argument("--out-dir", default="results/fedra_riemann")
    ap.add_argument("--max-rounds", type=int, default=FL_ROUNDS,
                    help="smoke tests only; keep the default for real runs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sessions = load_all_sessions_raw(args.data_dir)
    folds = range(1, N_SUBJECTS + 1) if args.fold == 0 else [args.fold]
    for target in folds:
        if f"S{target}E" not in sessions:
            raise SystemExit(f"Missing target session S{target}E in {args.data_dir}")
        res = run_fold(sessions, target, args.arm, args.seed, args.max_rounds)
        out = os.path.join(args.out_dir, f"{args.arm}_seed{args.seed}_S{target}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
