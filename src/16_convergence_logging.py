"""
BCI-IV 2a — CONVERGENCE LOGGING (FedAvg vs Fed-EA, genuine LOSO)
====================================================================
Purpose: No verified script in this project has logged round-by-round
validation accuracy (only final best-val / test-acc were saved), so no
real convergence-curve figure currently exists. This script re-runs
FedAvg (no alignment) and Fed-EA under the identical verified LOSO
scaffold, but logs mean validation accuracy at EVERY communication
round for all 9 LOSO folds, then averages across folds at each round
for a clean convergence curve.

Difference from fedavg_global_reverify.py / fed_ea_experiment.py:
training runs the FULL R=100 rounds every fold (early stopping is
DISABLED for this logging run, so every fold has a complete 100-round
trajectory to average) -- the final reported test accuracy still uses
the best-validation-round checkpoint, exactly as in the original
verified scripts, so the headline numbers from this run should closely
match the already-verified 39.84% (FedAvg) and 50.37% (Fed-EA) 2-seed
averages; only the per-round logging and disabled early stopping are
new. This makes the run more expensive (no early exit) but produces a
complete, comparable curve for every fold.

Output: convergence_log.json with, for each condition (fedavg, fedea):
  - per-fold round-by-round validation accuracy (9 folds x 100 rounds)
  - final test accuracy per fold (sanity check against verified numbers)
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

FL_ROUNDS     = 100     # NOTE: always runs to completion, no early stopping
LOCAL_EPOCHS  = 5
BATCH_SIZE    = 32
LR            = 1e-3
VAL_FRACTION  = 0.15

print(f"Device: {DEVICE}")

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
    print("\nLoading all sessions...")
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

def ems_only(X):
    X_out = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        X_out[i] = exponential_moving_standardize(X[i])
    return X_out[:, np.newaxis].astype(np.float32)

def compute_ea_whitening(X):
    n, C, T = X.shape
    R = np.zeros((C, C))
    for i in range(n):
        R += X[i] @ X[i].T
    R /= n
    return fractional_matrix_power(R, -0.5).real

def apply_ea(X, W):
    return np.einsum('cd,ndt->nct', W, X)

def ea_then_ems(X, W):
    return ems_only(apply_ea(X, W))

# ─────────────────────────────────────────────
# MODEL
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


def run_condition_with_logging(sessions, use_ea, results):
    label = "Fed-EA" if use_ea else "FedAvg (no alignment)"
    key = "fedea" if use_ea else "fedavg"
    print(f"\n{'#'*70}\n  {label} — CONVERGENCE LOGGING\n{'#'*70}")
    fold_results = results.get(key, {})

    for target in range(1, N_SUBJECTS + 1):
        if f"S{target}" in fold_results:
            print(f"\n  S{target}: already done, skipping.")
            continue

        print(f"\n  LOSO fold: target = S{target}")
        clients = [s for s in range(1, N_SUBJECTS + 1) if s != target]
        Xt_key = f"S{target}E"
        if Xt_key not in sessions:
            continue
        X_target_raw, y_target = sessions[Xt_key]

        client_data, client_val = {}, {}
        if use_ea:
            W_target = compute_ea_whitening(X_target_raw)
            X_target = ea_then_ems(X_target_raw, W_target)
        else:
            X_target = ems_only(X_target_raw)

        for s in clients:
            key_t = f"S{s}T"
            if key_t not in sessions:
                continue
            X_s_raw, y_s = sessions[key_t]
            if use_ea:
                W_s = compute_ea_whitening(X_s_raw)
                X_s = ea_then_ems(X_s_raw, W_s)
            else:
                X_s = ems_only(X_s_raw)
            n = len(y_s)
            idx = np.random.RandomState(42).permutation(n)
            n_val = max(int(n * VAL_FRACTION), 1)
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            client_data[s] = (X_s[train_idx], y_s[train_idx])
            client_val[s]  = (X_s[val_idx], y_s[val_idx])

        torch.manual_seed(42)
        global_model = EEGNet().to(DEVICE)
        global_state = global_model.state_dict()
        weights = [len(client_data[s][0]) for s in clients]

        round_val_accs = []
        best_val, best_state = 0.0, copy.deepcopy(global_state)

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
            mean_val = float(np.mean(val_accs))
            round_val_accs.append(mean_val)

            if mean_val > best_val:
                best_val = mean_val
                best_state = copy.deepcopy(global_state)

            if rnd % 20 == 0:
                print(f"    round {rnd:3d}/{FL_ROUNDS} | mean val: {mean_val:.4f}")

        global_model.load_state_dict(best_state)
        test_acc, test_f1 = evaluate(global_model, X_target, y_target)
        print(f"  >> S{target}: best_val={best_val:.4f}  test_acc={test_acc:.4f}")

        fold_results[f"S{target}"] = {
            "round_val_accs": round_val_accs,
            "test_acc": round(test_acc, 4),
            "test_f1": round(test_f1, 4),
        }
        results[key] = fold_results
        with open(os.path.join(SAVE_DIR, "convergence_log.json"), 'w') as f:
            json.dump(results, f)

    return fold_results


def print_summary(results):
    print(f"\n{'='*70}\n  CONVERGENCE LOGGING — SUMMARY\n{'='*70}")
    for key, label in [("fedavg", "FedAvg (no alignment)"), ("fedea", "Fed-EA")]:
        if key not in results:
            continue
        test_accs = [v["test_acc"] for v in results[key].values()]
        ref = "39.84%" if key == "fedavg" else "50.37%"
        print(f"  {label}: mean test_acc={np.mean(test_accs):.4f} (sanity check vs. verified: {ref})")
        curves = np.array([v["round_val_accs"] for v in results[key].values()])
        avg_curve = curves.mean(axis=0)
        print(f"    Round  1 avg val: {avg_curve[0]:.4f}")
        print(f"    Round 20 avg val: {avg_curve[19]:.4f}")
        print(f"    Round 50 avg val: {avg_curve[49]:.4f}")
        print(f"    Round 100 avg val: {avg_curve[99]:.4f}")
        print(f"    Best round: {int(np.argmax(avg_curve))+1} (avg val={avg_curve.max():.4f})")


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    sessions = load_all_sessions_raw()
    print(f"\nLoaded {len(sessions)} sessions.")

    path = os.path.join(SAVE_DIR, "convergence_log.json")
    results = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                results = json.load(f)
            print("Resuming from existing convergence_log.json")
        except Exception:
            pass

    run_condition_with_logging(sessions, use_ea=False, results=results)
    run_condition_with_logging(sessions, use_ea=True, results=results)

    print_summary(results)
    print(f"\nFull results saved: {path}")


if __name__ == "__main__":
    main()