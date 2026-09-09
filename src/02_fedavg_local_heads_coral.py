"""
BCI-IV 2a — FEDAVG + LOCAL HEADS (with and without CORAL)
============================================================
Purpose: Figure 4's heatmap currently shows APPROXIMATED per-subject
values for "FedAvg + local heads" and "FedAvg + local heads + CORAL"
(interpolated from the mean accuracy only). This script produces REAL
per-subject LOSO accuracy for both, resolving that gap in Figure 4.

Architecture: EEGNet is split into a FEATURE EXTRACTOR (temporal +
depthwise + separable conv, producing the 432-dim feature vector) and a
LOCAL CLASSIFIER HEAD (a single linear layer, 432 -> 4 classes) per
client. Only the feature extractor is federated via FedAvg; each
client's classification head is trained locally every round but is
NEVER aggregated, shared, or reset — it persists and specialises to
that client across all communication rounds. This is a standard
personalised-FL design and is architecturally distinct from FedAvg
global (which shares BOTH the feature extractor and the classifier).

Evaluation on the held-out target subject: since the target has no
local head of its own (it never participated in training), we evaluate
by applying the final aggregated global feature extractor together with
EACH of the 8 source clients' local heads, then averaging the resulting
softmax probabilities across all 8 heads (unweighted ensemble) — the
same ensembling convention used for the local-only baseline elsewhere
in this paper, applied here at the classifier-head level only (the
feature extractor itself IS federated, unlike the local-only baseline).

CORAL variant — explicit protocol (resolving the ambiguity flagged in
review): at the START of each communication round, every client computes
its local feature covariance C_k (from the CURRENT global feature
extractor's output on its own T-session trials — no raw data leaves the
client). Each client sends ONLY this covariance matrix (a 432x432
summary statistic, not raw data or gradients) to the server. The server
computes a source-population reference covariance C_ref as the
sample-size-weighted average of the 8 clients' C_k (i.e. FedAvg-style
aggregation applied to covariance matrices instead of model parameters),
then broadcasts C_ref back to all clients. Each client then trains
locally for E=5 epochs using cross-entropy plus a CORAL term pulling its
own running feature covariance toward C_ref. CRITICALLY: C_ref is a
source-population statistic only -- it never includes any trial, label,
gradient, or covariance information from the held-out target subject,
so the protocol remains fully unsupervised and target-free throughout.

Output: fedavg_localheads_results.json with per-subject accuracy for
both conditions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score
import os, json, copy

# ─────────────────────────────────────────────
# CONFIG (identical preprocessing/architecture to prior EEGNet scripts)
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
CORAL_LAMBDA  = 10   # matches the paper's best centralised CORAL lambda (Section 4.5.1)

print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
# PREPROCESSING — identical to EEGNet runs (4-40 Hz + EMS)
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

def load_session(path):
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
            epoch = exponential_moving_standardize(epoch)
            X_list.append(epoch)
            y_list.append(lbl - 1)
    if len(X_list) == 0:
        return None, None
    X = np.stack(X_list).astype(np.float32)[:, np.newaxis]
    y = np.array(y_list, dtype=np.int64)
    return X, y

def load_all_sessions():
    sessions = {}
    print("\nLoading all sessions...")
    for s in range(1, N_SUBJECTS + 1):
        for sess in ['T', 'E']:
            fname = f"A0{s}{sess}.mat"
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fpath):
                continue
            X, y = load_session(fpath)
            if X is None:
                continue
            sessions[f"S{s}{sess}"] = (X, y)
            print(f"  S{s}{sess}: {len(y)} trials")
    return sessions

# ─────────────────────────────────────────────
# MODEL — EEGNet split: feature extractor + local classifier head
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
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_rate)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop_rate)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            x = self.separable(self.depthwise(self.temporal(dummy)))
            self.feat_dim = x.numel()

    def forward(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x.flatten(1)   # (B, 432)


class LocalHead(nn.Module):
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


def evaluate_ensemble(extractor, heads, X, y):
    """Average softmax across all client heads, using the shared
    (aggregated) feature extractor."""
    extractor.eval()
    Xt = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad():
        z = extractor(Xt)
        probs_sum = None
        for head in heads:
            head.eval()
            logits = head(z)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        probs_avg = probs_sum / len(heads)
        preds = probs_avg.argmax(1).cpu().numpy()
    acc = float((preds == y).mean())
    f1 = float(f1_score(y, preds, average='macro', zero_division=0))
    return acc, f1


def feature_covariance(extractor, X):
    extractor.eval()
    Xt = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad():
        z = extractor(Xt)                        # (n, 432)
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / max(z.shape[0] - 1, 1)  # (432, 432)
    return cov


def coral_loss(z, C_ref):
    """z: (B, 432) batch features. Aligns this batch's covariance to
    the broadcast source-population reference covariance C_ref."""
    z_c = z - z.mean(dim=0, keepdim=True)
    n = z.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=z.device)
    C_batch = (z_c.T @ z_c) / (n - 1)
    d = z.shape[1]
    return ((C_batch - C_ref) ** 2).sum() / (4 * d * d)


def client_local_train(extractor_state, head, X_train, y_train,
                        use_coral=False, C_ref=None, local_epochs=LOCAL_EPOCHS):
    """Trains a fresh extractor (init'd from the current global state)
    and this client's persistent local head for `local_epochs`, jointly.
    Returns the updated extractor state_dict (to be FedAvg-aggregated)
    and leaves `head` mutated in-place (never aggregated)."""
    extractor = EEGNetFeatureExtractor().to(DEVICE)
    extractor.load_state_dict(extractor_state)
    opt = optim.Adam(list(extractor.parameters()) + list(head.parameters()), lr=LR)
    cw = class_weights(y_train).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    extractor.train()
    head.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            z = extractor(xb)
            logits = head(z)
            loss = crit(logits, yb)
            if use_coral and C_ref is not None:
                loss = loss + CORAL_LAMBDA * coral_loss(z, C_ref)
            loss.backward()
            opt.step()

    return extractor.state_dict()


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
# LOSO — FedAvg + local heads (optionally + CORAL)
# ─────────────────────────────────────────────
def run_condition(sessions, use_coral):
    label = "FedAvg + local heads + CORAL" if use_coral else "FedAvg + local heads"
    print(f"\n{'#'*70}\n  {label}\n{'#'*70}")
    results = {}

    for target in range(1, N_SUBJECTS + 1):
        print(f"\n  LOSO fold: target = S{target}")
        clients = [s for s in range(1, N_SUBJECTS + 1) if s != target]
        Xt_key = f"S{target}E"
        if Xt_key not in sessions:
            continue
        X_target, y_target = sessions[Xt_key]

        client_data, client_val = {}, {}
        for s in clients:
            key = f"S{s}T"
            if key not in sessions:
                continue
            X_s, y_s = sessions[key]
            n = len(y_s)
            idx = np.random.RandomState(42).permutation(n)
            n_val = max(int(n * VAL_FRACTION), 1)
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            client_data[s] = (X_s[train_idx], y_s[train_idx])
            client_val[s]  = (X_s[val_idx], y_s[val_idx])

        torch.manual_seed(42)
        global_extractor = EEGNetFeatureExtractor().to(DEVICE)
        global_state = global_extractor.state_dict()
        feat_dim = global_extractor.feat_dim

        # Persistent local heads, one per client, initialised once and
        # never reset or aggregated across rounds.
        torch.manual_seed(43)
        local_heads = {s: LocalHead(feat_dim).to(DEVICE) for s in clients}

        weights = [len(client_data[s][0]) for s in clients]

        best_val = 0.0
        best_extractor_state = copy.deepcopy(global_state)
        best_head_states = {s: copy.deepcopy(local_heads[s].state_dict()) for s in clients}
        patience_cnt = 0

        for rnd in range(1, FL_ROUNDS + 1):
            # --- CORAL: compute + broadcast source-population reference covariance ---
            C_ref = None
            if use_coral:
                covs, cov_weights = [], []
                for s in clients:
                    X_tr, _ = client_data[s]
                    covs.append(feature_covariance(global_extractor, X_tr))
                    cov_weights.append(len(X_tr))
                total_w = sum(cov_weights)
                C_ref = sum(c * (w / total_w) for c, w in zip(covs, cov_weights))
                C_ref = C_ref.detach()

            client_states = []
            for s in clients:
                X_tr, y_tr = client_data[s]
                st = client_local_train(
                    global_state, local_heads[s], X_tr, y_tr,
                    use_coral=use_coral, C_ref=C_ref
                )
                client_states.append(st)

            global_state = fed_avg_state(client_states, weights)
            global_extractor.load_state_dict(global_state)

            val_accs = []
            for s in clients:
                Xv, yv = client_val[s]
                acc, _ = evaluate_ensemble(global_extractor, [local_heads[s]], Xv, yv)
                val_accs.append(acc)
            mean_val = np.mean(val_accs)

            if rnd % 20 == 0 or rnd == 1:
                print(f"    round {rnd:3d}/{FL_ROUNDS} | mean val: {mean_val:.4f}")

            if mean_val > best_val:
                best_val = mean_val
                best_extractor_state = copy.deepcopy(global_state)
                best_head_states = {s: copy.deepcopy(local_heads[s].state_dict()) for s in clients}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= FL_PATIENCE:
                print(f"    early stop @ round {rnd}")
                break

        global_extractor.load_state_dict(best_extractor_state)
        for s in clients:
            local_heads[s].load_state_dict(best_head_states[s])

        acc, f1 = evaluate_ensemble(global_extractor, list(local_heads.values()), X_target, y_target)
        print(f"  >> S{target}: best_val={best_val:.4f}  test_acc={acc:.4f}  f1={f1:.4f}")
        results[f"S{target}"] = {"acc": round(acc, 4), "f1": round(f1, 4)}

        path = os.path.join(SAVE_DIR, "fedavg_localheads_results.json")
        existing = {}
        if os.path.exists(path):
            try:
                with open(path) as rf:
                    existing = json.load(rf)
            except Exception:
                pass
        key = "with_coral" if use_coral else "no_coral"
        existing[key] = results
        with open(path, 'w') as f:
            json.dump(existing, f, indent=2)

    return results


def print_summary(all_results):
    print(f"\n{'='*70}\n  FEDAVG + LOCAL HEADS — SUMMARY\n{'='*70}")
    for key, label in [("no_coral", "FedAvg+local heads"), ("with_coral", "FedAvg+local heads+CORAL")]:
        if key not in all_results:
            continue
        accs = [v["acc"] for v in all_results[key].values()]
        print(f"  {label:30s} mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
    print(f"\n  Compare to paper's reported means: 43.46% (no CORAL), 43.16% (CORAL)")


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    sessions = load_all_sessions()
    print(f"\nLoaded {len(sessions)} sessions.")

    all_results = {}
    all_results["no_coral"]   = run_condition(sessions, use_coral=False)
    all_results["with_coral"] = run_condition(sessions, use_coral=True)

    print_summary(all_results)
    out_path = os.path.join(SAVE_DIR, "fedavg_localheads_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved: {out_path}")


if __name__ == "__main__":
    main()