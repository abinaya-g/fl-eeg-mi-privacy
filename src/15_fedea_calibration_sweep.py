"""
BCI-IV 2a — FED-EA TARGET-CALIBRATION-SIZE SWEEP (genuine LOSO protocol)
==========================================================================
CORRECTED VERSION: built directly on top of the verified
fed_ea_experiment.py / fed_ea_experiment_seed123.py pipeline (the
script that produced the paper's 50.37% headline number), NOT
reconstructed from scratch. A first attempt at this sweep, written
without the original source in hand, failed its own frac=1.00 sanity
check (42.71% instead of the verified 50.37%) -- root causes were an
incorrect EMS implementation (batch z-score instead of the real
per-trial exponential moving standardisation), LOCAL_EPOCHS=1 instead
of 5, no class-weighted loss, and a slightly wrong EEGNet architecture.
Every one of those pieces below is now copied verbatim from the
verified source; the ONLY change from the original script is in
run_loso(), where the target subject's EA whitening matrix is computed
from a FRACTION of its own E-session trials instead of all of them,
and evaluation happens on a disjoint held-out remainder for fractions
below 1.00.

Purpose: standard (transductive) Fed-EA uses ALL of the held-out
target's unlabelled E-session trials to compute its EA whitening
matrix (50.37% mean acc); the fully inductive variant uses NONE of the
target's data (37.82% mean acc). This sweeps the fraction of target
data used for calibration -- f in {0.25, 0.50, 0.75, 1.00} -- to get a
dose-response curve between those two endpoints.

Design:
  - EMS (exponential moving standardisation) is applied per-trial and
    does NOT depend on any reference statistic from other trials, so
    the calibration fraction affects ONLY the EA whitening matrix, not
    EMS. This matches the original pipeline's structure exactly.
  - Source-client training is IDENTICAL to the original script and is
    trained once per (fold, seed); only the target-side calibration
    step changes across fractions.
  - For f < 1.00: 5 independent random calibration/test splits per
    (fold, seed), calibration trials never overlap with test trials.
    Reported per-fold accuracy at f < 1.00 is the mean over splits.
  - For f = 1.00: calibration uses all E-session trials and evaluation
    also uses all E-session trials, EXACTLY matching the original
    script's protocol -- this is included as a sanity check and should
    reproduce ~50.37% (seed 42+123 average) before you trust anything
    else in this sweep.
  - Two seeds (42, 123), matching the paper's existing standard. Seed
    handling (RandomState for the per-client validation split, model
    init, cuda) mirrors the original script exactly: RandomState(seed)
    and torch.manual_seed(seed), not a separate fixed constant.

Output: fed_ea_calibration_sweep_results.json.

BEFORE RUNNING A FULL SWEEP: run with FRACTIONS = [1.00] only, for a
couple of folds, and confirm the accuracy lands close to your existing
per-subject Fed-EA numbers for those folds. Only proceed to the full
sweep once that check passes.
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
# CONFIG — identical to fed_ea_experiment.py
# ─────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/datasets/abinayajone/bci-iv-2a-mi"
SAVE_DIR   = "/kaggle/working"
RESULTS_JSON = os.path.join(SAVE_DIR, "fed_ea_calibration_sweep_results.json")
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

SEEDS       = [42, 123]
FRACTIONS   = [0.25, 0.50, 0.75, 1.00]   # set to [1.00] for the sanity-check run
N_SPLITS_PER_FRACTION = 5                 # only used when fraction < 1.00

print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
# PREPROCESSING — identical to fed_ea_experiment.py
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

def load_all_sessions_raw():
    sessions = {}
    print("\nLoading all sessions (pre-EA, pre-EMS)...")
    for s in range(1, N_SUBJECTS + 1):
        for sess in ['T', 'E']:
            fpath = os.path.join(DATA_DIR, f"A0{s}{sess}.mat")
            if not os.path.exists(fpath):
                continue
            X, y = load_session_raw(fpath)
            if X is None:
                continue
            sessions[f"S{s}{sess}"] = (X, y)
            print(f"  S{s}{sess}: {len(y)} trials")
    return sessions

# ─────────────────────────────────────────────
# EUCLIDEAN ALIGNMENT — identical to fed_ea_experiment.py
# ─────────────────────────────────────────────
def compute_ea_whitening(X):
    """X: (n_trials, C, T). Returns W = R^(-1/2) computed entirely from
    these trials -- no eps regularisation, matching the original."""
    n, C, T = X.shape
    R = np.zeros((C, C))
    for i in range(n):
        R += X[i] @ X[i].T
    R /= n
    W = fractional_matrix_power(R, -0.5).real
    return W

def apply_ea(X, W):
    return np.einsum('cd,ndt->nct', W, X)

def ea_apply_then_ems(X_raw, W):
    """Apply an EXTERNALLY-SUPPLIED whitening matrix W (e.g. computed
    from a calibration subset), then per-epoch EMS -- identical
    postprocessing to ea_then_ems() in the original script, but W is
    passed in rather than computed from X_raw itself. This is the only
    structural change from the original pipeline."""
    X_aligned = apply_ea(X_raw, W)
    X_out = np.zeros_like(X_aligned, dtype=np.float32)
    for i in range(X_aligned.shape[0]):
        X_out[i] = exponential_moving_standardize(X_aligned[i])
    return X_out[:, np.newaxis].astype(np.float32)

def ea_then_ems(X):
    """Original convenience wrapper: whitening computed from X itself,
    then applied to X itself. Used for source clients (unaffected by
    calibration fraction) and for the frac=1.00 sanity-check case."""
    W = compute_ea_whitening(X)
    return ea_apply_then_ems(X, W), W

# ─────────────────────────────────────────────
# MODEL — identical EEGNet to fed_ea_experiment.py
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
# CHECKPOINTING
# ─────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return {"per_run": {}}

def save_checkpoint(results):
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)

def run_key(subj, seed, frac, split_idx):
    return f"S{subj}_seed{seed}_frac{frac:.2f}_split{split_idx}"

# ─────────────────────────────────────────────
# LOSO — Fed-EA calibration sweep (resume-aware)
# ─────────────────────────────────────────────
def train_source_fedavg(sessions, target, seed):
    """Trains the FedAvg global model on the 8 source clients, EXACTLY
    matching fed_ea_experiment.py's run_loso() body (source side only).
    Returns the trained EEGNet with best-val weights loaded."""
    clients = [s for s in range(1, N_SUBJECTS + 1) if s != target]

    client_data, client_val = {}, {}
    for s in clients:
        key = f"S{s}T"
        if key not in sessions:
            continue
        X_s_raw, y_s = sessions[key]
        X_s, _ = ea_then_ems(X_s_raw)   # per-client EA, own T-session data only

        n = len(y_s)
        idx = np.random.RandomState(seed).permutation(n)
        n_val = max(int(n * VAL_FRACTION), 1)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        client_data[s] = (X_s[train_idx], y_s[train_idx])
        client_val[s]  = (X_s[val_idx], y_s[val_idx])

    torch.manual_seed(seed)
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
    return global_model, best_val

def run_sweep(sessions):
    results = load_checkpoint()

    for seed in SEEDS:
        for target in range(1, N_SUBJECTS + 1):
            Xt_key = f"S{target}E"
            if Xt_key not in sessions:
                continue
            X_target_raw, y_target = sessions[Xt_key]
            n_tgt = X_target_raw.shape[0]

            all_keys_needed = []
            for frac in FRACTIONS:
                n_splits = 1 if frac >= 1.0 else N_SPLITS_PER_FRACTION
                for sp in range(n_splits):
                    all_keys_needed.append(run_key(target, seed, frac, sp))
            if all(k in results["per_run"] for k in all_keys_needed):
                print(f"\n  S{target} seed={seed}: all fractions already done, skipping fold.")
                continue

            print(f"\n{'='*60}\n  LOSO fold: target = S{target}, seed = {seed}\n{'='*60}")
            model, best_val = train_source_fedavg(sessions, target, seed)

            for frac in FRACTIONS:
                n_splits = 1 if frac >= 1.0 else N_SPLITS_PER_FRACTION
                for split_idx in range(n_splits):
                    key = run_key(target, seed, frac, split_idx)
                    if key in results["per_run"]:
                        print(f"    frac={frac:.2f} split={split_idx}: already done, skipping.")
                        continue

                    rng = np.random.RandomState(1000 * split_idx + target)
                    idx = np.arange(n_tgt)
                    rng.shuffle(idx)

                    if frac >= 1.0:
                        calib_idx = idx
                        test_idx = idx        # matches original protocol exactly
                    else:
                        n_calib = max(1, int(round(frac * n_tgt)))
                        calib_idx = idx[:n_calib]
                        test_idx = idx[n_calib:]
                        if len(test_idx) == 0:
                            continue

                    W_tgt = compute_ea_whitening(X_target_raw[calib_idx])   # unlabelled calib subset only
                    X_test = ea_apply_then_ems(X_target_raw[test_idx], W_tgt)
                    y_test = y_target[test_idx]

                    acc, f1 = evaluate(model, X_test, y_test)
                    print(f"    frac={frac:.2f} split={split_idx} "
                          f"n_calib={len(calib_idx)} n_test={len(test_idx)} "
                          f"acc={acc:.4f} f1={f1:.4f}")

                    results["per_run"][key] = {
                        "subject": target, "seed": seed, "fraction": frac,
                        "split_idx": split_idx, "n_calib": int(len(calib_idx)),
                        "n_test": int(len(test_idx)), "acc": round(acc, 4), "f1": round(f1, 4),
                    }
                    save_checkpoint(results)

    return results

def print_summary(results):
    print(f"\n{'='*70}\n  FED-EA CALIBRATION-SIZE SWEEP — SUMMARY\n{'='*70}")
    summary = {}
    for frac in FRACTIONS:
        per_subject_seed = {}
        for run in results["per_run"].values():
            if run["fraction"] != frac:
                continue
            k = (run["subject"], run["seed"])
            per_subject_seed.setdefault(k, []).append(run["acc"])
        if not per_subject_seed:
            continue
        subj_seed_means = [float(np.mean(v)) for v in per_subject_seed.values()]
        mean_acc = float(np.mean(subj_seed_means))
        std_acc = float(np.std(subj_seed_means))
        summary[f"{frac:.2f}"] = {"mean_acc": mean_acc, "std_acc": std_acc,
                                   "n_subject_seed_cells": len(subj_seed_means)}
        print(f"  frac={frac:.2f}  mean_acc={mean_acc:.4f}  std={std_acc:.4f}  "
              f"(n={len(subj_seed_means)} subject×seed cells)")
    print("\n  Sanity check: frac=1.00 above should be close to 50.37%")
    print("  (the verified transductive Fed-EA 2-seed average). If it is")
    print("  not, STOP and do not trust the other fractions -- something")
    print("  in this script still diverges from the verified pipeline.")
    results["summary"] = summary
    save_checkpoint(results)

def main():
    sessions = load_all_sessions_raw()
    print(f"\nLoaded {len(sessions)} sessions.")
    results = run_sweep(sessions)
    print_summary(results)
    print(f"\nFull results saved: {RESULTS_JSON}")

if __name__ == "__main__":
    main()
