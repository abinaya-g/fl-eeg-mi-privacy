"""
BCI-IV 2a — FEDERATED EUCLIDEAN ALIGNMENT (Fed-EA), SEED 123 SANITY CHECK
============================================================================
The first Fed-EA run (seed=42) gave a mean of 50.68%, a +10.85pp
improvement over the no-EA FedAvg baseline (39.84%, itself a 2-seed
average), significant under both paired t-test (p=0.0014) and Wilcoxon
(p=0.0039) with a large effect size (dz=1.60) -- every one of the 9
subjects improved under Fed-EA relative to no-EA FedAvg. Given how
large and clean this effect is, and given that the paper's ORIGINAL
headline number (46.67%) turned out not to replicate under a seed/
protocol check, this script applies the same discipline here: identical
protocol, seed 123 instead of 42 (affecting weight init, client data
shuffling, and the train/val split -- EA whitening itself is
deterministic given the data, so it is NOT reseeded by this change).

If this run also lands well above the no-EA baselines (~40%) and
reasonably close to the seed=42 result (~50.68%), that is strong
evidence Fed-EA is a genuine, robust effect. If this run lands much
closer to the no-EA baselines, treat the seed=42 result with caution
and investigate further before reporting it as a finding.

Purpose otherwise unchanged: Euclidean Alignment (He & Wu, 2020),
applied per-subject with zero cross-client communication, computed
entirely from each subject's own trials (source clients: own
T-session; held-out target: own E-session, label-free but
transductive). Tests whether a communication-FREE alignment method
avoids the failure mode diagnosed for communication-DEPENDENT methods
(CORAL, FedRA) in Sections 5.2-5.3.

Output: fed_ea_seed123_results.json with per-subject LOSO accuracy.
Compare against BOTH the seed=42 Fed-EA result (50.68%) and the no-EA
FedAvg baselines (~40%).
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
# PREPROCESSING — bandpass only here; EA and EMS applied separately
# below so EA operates on bandpassed-but-not-yet-EMS-normalised trials
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
    """Loads bandpassed epochs WITHOUT EMS normalisation -- EA is
    applied first (below), then EMS is applied per-epoch afterward."""
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
            X_list.append(epoch)   # NOTE: no EMS yet
            y_list.append(lbl - 1)
    if len(X_list) == 0:
        return None, None
    X = np.stack(X_list).astype(np.float64)   # (n_trials, 22, T)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def load_all_sessions_raw():
    sessions = {}
    print("\nLoading all sessions (pre-EA, pre-EMS)...")
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
# EUCLIDEAN ALIGNMENT (He & Wu, 2020) — per-subject, zero communication
# ─────────────────────────────────────────────
def compute_ea_whitening(X):
    """X: (n_trials, C, T) for ONE subject's own trials.
    Returns W = R^(-1/2), the whitening matrix for THIS subject only,
    computed entirely from its own data -- no cross-subject statistic
    involved at any point."""
    n, C, T = X.shape
    R = np.zeros((C, C))
    for i in range(n):
        R += X[i] @ X[i].T
    R /= n
    W = fractional_matrix_power(R, -0.5).real
    return W

def apply_ea(X, W):
    """X: (n_trials, C, T), W: (C, C) -> whitened (n_trials, C, T)."""
    return np.einsum('cd,ndt->nct', W, X)

def ea_then_ems(X):
    """Apply this subject's own EA whitening, then per-epoch EMS
    normalisation (matching every other script's pipeline)."""
    W = compute_ea_whitening(X)
    X_aligned = apply_ea(X, W)
    X_out = np.zeros_like(X_aligned, dtype=np.float32)
    for i in range(X_aligned.shape[0]):
        X_out[i] = exponential_moving_standardize(X_aligned[i])
    return X_out[:, np.newaxis].astype(np.float32)   # add channel dim for EEGNet


# ─────────────────────────────────────────────
# MODEL — identical EEGNet used throughout this revision
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
# LOSO — Fed-EA (resume-aware)
# ─────────────────────────────────────────────
def load_existing_results():
    path = os.path.join(SAVE_DIR, "fed_ea_seed123_results.json")
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

        # EA for the target: computed from the target's OWN E-session
        # trials only (label-free, transductive -- see docstring).
        print(f"  Computing EA whitening for target S{target} "
              f"(from its own {len(y_target)} E-session trials)...")
        X_target = ea_then_ems(X_target_raw)

        client_data, client_val = {}, {}
        for s in clients:
            key = f"S{s}T"
            if key not in sessions:
                continue
            X_s_raw, y_s = sessions[key]
            # EA for this source client: computed from ITS OWN
            # T-session trials only -- zero communication with other
            # clients or the server at this stage.
            X_s = ea_then_ems(X_s_raw)

            n = len(y_s)
            idx = np.random.RandomState(123).permutation(n)
            n_val = max(int(n * VAL_FRACTION), 1)
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            client_data[s] = (X_s[train_idx], y_s[train_idx])
            client_val[s]  = (X_s[val_idx], y_s[val_idx])

        torch.manual_seed(123)
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

        results[f"S{target}"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
        with open(os.path.join(SAVE_DIR, "fed_ea_seed123_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    return results


def print_summary(results):
    print(f"\n{'='*70}\n  FED-EA, SEED 123 — LOSO SUMMARY\n{'='*70}")
    accs = [v["acc"] for v in results.values()]
    f1s = [v["f1"] for v in results.values()]
    for s, r in sorted(results.items(), key=lambda kv: int(kv[0][1:])):
        print(f"  {s:<6} acc={r['acc']:.4f}  f1={r['f1']:.4f}")
    print(f"  {'-'*30}")
    print(f"  Mean   acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}")
    print(f"  Std    acc={np.std(accs):.4f}")
    print(f"\n  Compare to:")
    print(f"    Fed-EA, seed 42 (first run)        : 50.68%")
    print(f"    FedAvg global, no EA (2-seed avg)  : 39.84%")
    print(f"    Centralised EEGNet                  : 40.26%")
    print(f"    Local-only ensemble                 : 36.67%")
    print(f"\n  If this seed also lands well above ~40% and reasonably close")
    print(f"  to 50.68%: Fed-EA is a robust, genuine effect.")
    print(f"  If this seed lands much closer to ~40%: treat the seed=42")
    print(f"  result with caution -- investigate before reporting as a finding.")


def main():
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    sessions = load_all_sessions_raw()
    print(f"\nLoaded {len(sessions)} sessions.")
    results = run_loso(sessions)
    print_summary(results)
    print(f"\nFull results saved: {os.path.join(SAVE_DIR, 'fed_ea_seed123_results.json')}")


if __name__ == "__main__":
    main()