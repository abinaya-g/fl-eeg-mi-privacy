"""
BCI-IV 2a — FED-EA, FULLY INDUCTIVE VARIANT
================================================================
Purpose: The standard Fed-EA design (fed_ea_experiment.py) computes the
held-out target subject's whitening matrix from the TARGET'S OWN
unlabelled E-session trials -- a transductive assumption. This leaves
open whether Fed-EA's advantage comes from (a) being communication-free,
or (b) having batch access to target-specific statistics that CORAL and
FedRA do not use in the same way. This script isolates factor (a) by
constructing a FULLY INDUCTIVE version of Fed-EA: the target subject's
whitening matrix is derived ENTIRELY from source-client statistics,
never touching any target-subject data, labelled or unlabelled.

Design: each of the 8 source clients computes its own local EA
whitening matrix exactly as in standard Fed-EA (zero communication
during their own training). For the target subject specifically
(who has no training data of its own), we use the UNWEIGHTED AVERAGE of
the 8 source clients' individual whitening matrices as a source-derived
proxy transform. This average is computed once per LOSO fold and
constitutes a single, lightweight aggregation step (8 matrices of size
22x22) used only to handle the unseen target -- it does not affect how
source clients train, and it never involves target data. This is
comparable in spirit to FedRA's aggregation but computed from EA
matrices rather than raw covariances, and unlike FedRA's per-round
communication, this is a ONE-TIME step needed only because the target
has no way to compute its own inductive-context-free transform.

If this inductive variant performs close to standard (transductive)
Fed-EA, that is evidence Fed-EA's advantage is NOT primarily driven by
its transductive target-data access. If it performs much worse, that
suggests the transductive assumption is doing real work, and the
"communication-free" framing alone does not fully explain standard
Fed-EA's advantage.

Output: fed_ea_inductive_results.json with per-subject LOSO accuracy.
Compare against standard (transductive) Fed-EA (50.37%, 2-seed average)
and against FedAvg (no alignment, 39.84%) and FedRA (40.72%, fully
inductive but different math) to complete the inductive/transductive
and communication-free/dependent 2x2 comparison.
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
# PREPROCESSING — bandpass only; EA and EMS applied separately below
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
# EUCLIDEAN ALIGNMENT — identical maths to fed_ea_experiment.py
# ─────────────────────────────────────────────
def compute_ea_whitening(X):
    n, C, T = X.shape
    R = np.zeros((C, C))
    for i in range(n):
        R += X[i] @ X[i].T
    R /= n
    W = fractional_matrix_power(R, -0.5).real
    return W

def apply_ea(X, W):
    return np.einsum('cd,ndt->nct', W, X)

def ea_then_ems(X, W):
    X_aligned = apply_ea(X, W)
    X_out = np.zeros_like(X_aligned, dtype=np.float32)
    for i in range(X_aligned.shape[0]):
        X_out[i] = exponential_moving_standardize(X_aligned[i])
    return X_out[:, np.newaxis].astype(np.float32)


# ─────────────────────────────────────────────
# MODEL — identical EEGNet
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
# LOSO — Fed-EA, fully inductive variant (resume-aware)
# ─────────────────────────────────────────────
def load_existing_results():
    path = os.path.join(SAVE_DIR, "fed_ea_inductive_results.json")
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

        # Each source client computes its OWN local EA whitening and
        # applies it to its OWN training data (identical to standard
        # Fed-EA) -- zero communication for training purposes.
        client_data, client_val = {}, {}
        source_whitening_matrices = []
        for s in clients:
            key = f"S{s}T"
            if key not in sessions:
                continue
            X_s_raw, y_s = sessions[key]
            W_s = compute_ea_whitening(X_s_raw)
            source_whitening_matrices.append(W_s)
            X_s = ea_then_ems(X_s_raw, W_s)

            n = len(y_s)
            idx = np.random.RandomState(42).permutation(n)
            n_val = max(int(n * VAL_FRACTION), 1)
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            client_data[s] = (X_s[train_idx], y_s[train_idx])
            client_val[s]  = (X_s[val_idx], y_s[val_idx])

        # FULLY INDUCTIVE step: the target's whitening matrix is the
        # unweighted average of the 8 source clients' own EA matrices.
        # No target data of any kind is used to compute this.
        W_target_inductive = np.mean(source_whitening_matrices, axis=0)
        X_target = ea_then_ems(X_target_raw, W_target_inductive)

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

        results[f"S{target}"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
        with open(os.path.join(SAVE_DIR, "fed_ea_inductive_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    return results


def print_summary(results):
    print(f"\n{'='*70}\n  FED-EA, FULLY INDUCTIVE — LOSO SUMMARY\n{'='*70}")
    accs = [v["acc"] for v in results.values()]
    f1s = [v["f1"] for v in results.values()]
    for s, r in sorted(results.items(), key=lambda kv: int(kv[0][1:])):
        print(f"  {s:<6} acc={r['acc']:.4f}  f1={r['f1']:.4f}")
    print(f"  {'-'*30}")
    print(f"  Mean   acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}")
    print(f"  Std    acc={np.std(accs):.4f}")
    print(f"\n  Compare to:")
    print(f"    Fed-EA, transductive (2-seed avg) : 50.37%")
    print(f"    FedAvg (no alignment)              : 39.84%")
    print(f"    FedRA (fully inductive, different math): 40.72%")
    print(f"\n  If this lands close to 50.37%: Fed-EA's advantage is NOT")
    print(f"  primarily driven by transductive target-data access.")
    print(f"  If this drops toward ~40%: the transductive assumption is")
    print(f"  doing real work, and communication-freedom alone does not")
    print(f"  fully explain standard Fed-EA's advantage.")


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    sessions = load_all_sessions_raw()
    print(f"\nLoaded {len(sessions)} sessions.")
    results = run_loso(sessions)
    print_summary(results)
    print(f"\nFull results saved: {os.path.join(SAVE_DIR, 'fed_ea_inductive_results.json')}")


if __name__ == "__main__":
    main()