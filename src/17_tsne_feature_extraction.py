"""
BCI-IV 2a — t-SNE FEATURE EXTRACTION (FedAvg vs Fed-EA)
====================================================================
Purpose: No verified script has saved a trained model checkpoint, so no
real t-SNE figure currently exists from the verified runs. This script
trains FedAvg (no alignment) and Fed-EA for two specific LOSO folds --
target=S3 (the best-performing subject throughout this project) and
target=S2 (a BCI-illiterate subject) -- under the identical verified
protocol, then extracts the 432-dimensional EEGNet feature vector (the
penultimate layer, before the final classification head) for every
trial in the target subject's E-session, saving features + true labels
for offline t-SNE plotting.

Only 2 folds x 2 conditions = 4 trainings are run (not all 9 folds),
since t-SNE visualisation only needs representative examples.

Output: tsne_features.json with, for each of the 4 (subject, condition)
combinations: the 432-dim feature vector and true label for every trial
in that subject's E-session. Run the companion script figure_tsne.py
locally afterward to project with t-SNE and plot.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.linalg import fractional_matrix_power
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

TARGET_SUBJECTS = [2, 3]   # S2 = BCI-illiterate example, S3 = best-performing example

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
# MODEL — split so we can extract the 432-dim feature vector directly
# ─────────────────────────────────────────────
class EEGNetFeatureExtractor(nn.Module):
    def __init__(self, n_channels=22, n_times=875, F1=8, D=2, F2=16, kern_len=32, drop_rate=0.5):
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
            self.feat_dim = x.numel()

    def forward(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x.flatten(1)


class Classifier(nn.Module):
    def __init__(self, feat_dim, n_classes=N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, z):
        return self.fc(z)


def class_weights(y, n_classes=N_CLASSES):
    n = len(y)
    counts = np.array([max((y == c).sum(), 1) for c in range(n_classes)])
    w = n / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)

def evaluate(extractor, clf, X, y):
    extractor.eval(); clf.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).to(DEVICE)
        preds = clf(extractor(Xt)).argmax(1).cpu().numpy()
    acc = float((preds == y).mean())
    return acc

def extract_features(extractor, X):
    extractor.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).to(DEVICE)
        z = extractor(Xt)
    return z.cpu().numpy()

def client_local_train(ext_state, clf_state, X_train, y_train, local_epochs=LOCAL_EPOCHS):
    extractor = EEGNetFeatureExtractor().to(DEVICE)
    extractor.load_state_dict(ext_state)
    clf = Classifier(extractor.feat_dim).to(DEVICE)
    clf.load_state_dict(clf_state)
    params = list(extractor.parameters()) + list(clf.parameters())
    opt = optim.Adam(params, lr=LR)
    cw = class_weights(y_train).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    extractor.train(); clf.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(clf(extractor(xb)), yb)
            loss.backward()
            opt.step()
    return extractor.state_dict(), clf.state_dict()

def fed_avg_state(states, weights):
    total = sum(weights)
    avg = copy.deepcopy(states[0])
    for k in avg.keys():
        if avg[k].dtype.is_floating_point:
            avg[k] = sum(states[i][k] * (weights[i] / total) for i in range(len(states)))
        else:
            avg[k] = states[0][k]
    return avg


def train_and_extract(sessions, target, use_ea):
    label = "Fed-EA" if use_ea else "FedAvg"
    print(f"\n  --- Training {label} for target=S{target} ---")
    clients = [s for s in range(1, N_SUBJECTS + 1) if s != target]
    Xt_key = f"S{target}E"
    X_target_raw, y_target = sessions[Xt_key]

    if use_ea:
        W_target = compute_ea_whitening(X_target_raw)
        X_target = ea_then_ems(X_target_raw, W_target)
    else:
        X_target = ems_only(X_target_raw)

    client_data, client_val = {}, {}
    for s in clients:
        X_s_raw, y_s = sessions[f"S{s}T"]
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
    global_ext = EEGNetFeatureExtractor().to(DEVICE)
    global_clf = Classifier(global_ext.feat_dim).to(DEVICE)
    ext_state, clf_state = global_ext.state_dict(), global_clf.state_dict()
    weights = [len(client_data[s][0]) for s in clients]

    best_val, best_ext, best_clf = 0.0, copy.deepcopy(ext_state), copy.deepcopy(clf_state)
    patience_cnt = 0

    for rnd in range(1, FL_ROUNDS + 1):
        ext_states, clf_states = [], []
        for s in clients:
            X_tr, y_tr = client_data[s]
            e_st, c_st = client_local_train(ext_state, clf_state, X_tr, y_tr)
            ext_states.append(e_st); clf_states.append(c_st)
        ext_state = fed_avg_state(ext_states, weights)
        clf_state = fed_avg_state(clf_states, weights)
        global_ext.load_state_dict(ext_state)
        global_clf.load_state_dict(clf_state)

        val_accs = [evaluate(global_ext, global_clf, *client_val[s]) for s in clients]
        mean_val = np.mean(val_accs)
        if mean_val > best_val:
            best_val, best_ext, best_clf = mean_val, copy.deepcopy(ext_state), copy.deepcopy(clf_state)
            patience_cnt = 0
        else:
            patience_cnt += 1
        if rnd % 20 == 0:
            print(f"    round {rnd:3d} | mean val: {mean_val:.4f}")
        if patience_cnt >= FL_PATIENCE:
            print(f"    early stop @ round {rnd}")
            break

    global_ext.load_state_dict(best_ext)
    global_clf.load_state_dict(best_clf)
    test_acc = evaluate(global_ext, global_clf, X_target, y_target)
    features = extract_features(global_ext, X_target)
    print(f"  >> S{target} {label}: test_acc={test_acc:.4f}, extracted {features.shape[0]} feature vectors")

    return {
        "features": features.tolist(),
        "labels": y_target.tolist(),
        "test_acc": round(float(test_acc), 4),
    }


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    sessions = load_all_sessions_raw()
    print(f"\nLoaded {len(sessions)} sessions.")

    path = os.path.join(SAVE_DIR, "tsne_features.json")
    results = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                results = json.load(f)
            print("Resuming from existing tsne_features.json")
        except Exception:
            pass

    for target in TARGET_SUBJECTS:
        for use_ea in [False, True]:
            key = f"S{target}_{'fedea' if use_ea else 'fedavg'}"
            if key in results:
                print(f"\n  {key}: already done, skipping.")
                continue
            r = train_and_extract(sessions, target, use_ea)
            results[key] = r
            with open(path, 'w') as f:
                json.dump(results, f)

    print(f"\nAll done. Full results saved: {path}")
    print("Run figure_tsne.py locally next and plot.")


if __name__ == "__main__":
    main()