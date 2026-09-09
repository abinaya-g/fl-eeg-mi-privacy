"""
BCI-IV 2a — FedCL WARMUP CURRICULUM (genuine LOSO protocol)
====================================================================
Purpose: The original ablation study reported two linear-warmup FedCL
variants (30%->100% and 50%->100%), at 45.40% and 42.73% respectively,
but like FedAvg global, FedCL-fixed, and the local-heads variants, both
were computed under a personalised protocol, not genuine cross-subject
LOSO. This script re-runs both warmup variants under the same verified
LOSO scaffold already used for fedcl_fixed_stages.py, changing only the
curriculum schedule from fixed stages (40/70/100% at rounds 1/40/70) to
a linear warmup from an initial fraction rho_0 to 100% over the first
50 rounds, exactly as originally specified in the paper's Methodology
(Section 3.5.2): "the proportion increases linearly from an initial
value rho_0 in {0.30, 0.50} to 100% over the first 50 rounds."

Patience: 25 rounds for warmup variants (vs. 20 for fixed-stage), per
the original methodology description, to allow the gradual schedule
time to take effect.

Runs BOTH warmup variants (rho_0 = 0.30 and rho_0 = 0.50) sequentially
in one script, since they share the same infrastructure; each is
independently checkpointed and resumable.

Output: fedcl_warmup_results.json with per-subject LOSO accuracy for
both variants. Compare against FedCL-fixed (38.56%, already verified)
and plain FedAvg (39.84%, 2-seed average) to complete the curriculum
ablation.
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

FL_ROUNDS       = 100
LOCAL_EPOCHS    = 5
FL_PATIENCE     = 25   # warmup variants use 25, not fixed-stage's 20 (paper Sec 3.5.2)
BATCH_SIZE      = 32
LR              = 1e-3
VAL_FRACTION    = 0.15
WARMUP_ROUNDS   = 50    # linear ramp from rho_0 to 100% over first 50 rounds

# Which warmup variants to run this session (edit if resuming a subset)
VARIANTS_TO_RUN = [0.30, 0.50]

print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
# PREPROCESSING — identical to fedcl_fixed_stages.py
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
# MODEL — identical split EEGNet as fedcl_fixed_stages.py
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


class GlobalClassifier(nn.Module):
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


def get_warmup_fraction(rnd, rho_0, warmup_rounds=WARMUP_ROUNDS):
    """Linear ramp from rho_0 at round 1 to 1.0 at round warmup_rounds,
    then held at 1.0 for all subsequent rounds."""
    if rnd >= warmup_rounds:
        return 1.0
    return rho_0 + (1.0 - rho_0) * (rnd - 1) / (warmup_rounds - 1)


def compute_entropy_scores(extractor, classifier, X, y):
    extractor.eval(); classifier.eval()
    entropies = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                         batch_size=64, shuffle=False)
    with torch.no_grad():
        for xb, _ in loader:
            logits = classifier(extractor(xb.to(DEVICE)))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs = np.clip(probs, 1e-8, 1.0)
            ent = -np.sum(probs * np.log(probs), axis=1)
            entropies.extend(ent.tolist())
    return np.array(entropies)


def select_curriculum_indices(entropies, fraction):
    n_select = max(4, int(len(entropies) * fraction))
    order = np.argsort(entropies)   # ascending entropy = easiest first
    return order[:n_select]


def evaluate(extractor, classifier, X, y):
    extractor.eval(); classifier.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).to(DEVICE)
        preds = classifier(extractor(Xt)).argmax(1).cpu().numpy()
    acc = float((preds == y).mean())
    f1 = float(f1_score(y, preds, average='macro', zero_division=0))
    return acc, f1


def client_local_train(ext_state, clf_state, X_train, y_train, curr_idx):
    extractor = EEGNetFeatureExtractor().to(DEVICE)
    extractor.load_state_dict(ext_state)
    classifier = GlobalClassifier(extractor.feat_dim).to(DEVICE)
    classifier.load_state_dict(clf_state)

    X_c, y_c = X_train[curr_idx], y_train[curr_idx]
    cw = class_weights(y_c).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)
    params = list(extractor.parameters()) + list(classifier.parameters())
    opt = optim.Adam(params, lr=LR, weight_decay=1e-4)

    loader = DataLoader(TensorDataset(torch.from_numpy(X_c), torch.from_numpy(y_c)),
                         batch_size=BATCH_SIZE, shuffle=True)
    extractor.train(); classifier.train()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(classifier(extractor(xb)), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

    return extractor.state_dict(), classifier.state_dict(), len(curr_idx)


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
# LOSO — FedCL warmup (resume-aware, both variants in one file)
# ─────────────────────────────────────────────
def load_existing_results():
    path = os.path.join(SAVE_DIR, "fedcl_warmup_results.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            print(f"\n  Found existing results — resuming.")
            for rho in VARIANTS_TO_RUN:
                key = f"warmup_{int(rho*100)}"
                n_done = len(data.get(key, {}))
                print(f"    {key}: {n_done}/9 done")
            return data
        except Exception:
            pass
    return {}


def save_results(all_results):
    path = os.path.join(SAVE_DIR, "fedcl_warmup_results.json")
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2)


def run_loso_for_variant(sessions, rho_0, all_results):
    key = f"warmup_{int(rho_0*100)}"
    print(f"\n{'#'*70}\n  FEDCL WARMUP {int(rho_0*100)}% -> 100%\n{'#'*70}")
    results = all_results.get(key, {})

    for target in range(1, N_SUBJECTS + 1):
        if f"S{target}" in results:
            print(f"\n  S{target}: already done, skipping.")
            continue

        print(f"\n{'='*60}\n  LOSO fold: target = S{target}\n{'='*60}")
        clients = [s for s in range(1, N_SUBJECTS + 1) if s != target]
        Xt_key = f"S{target}E"
        if Xt_key not in sessions:
            continue
        X_target, y_target = sessions[Xt_key]

        client_data, client_val = {}, {}
        for s in clients:
            key_t = f"S{s}T"
            if key_t not in sessions:
                continue
            X_s, y_s = sessions[key_t]
            n = len(y_s)
            idx = np.random.RandomState(42).permutation(n)
            n_val = max(int(n * VAL_FRACTION), 1)
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            client_data[s] = (X_s[train_idx], y_s[train_idx])
            client_val[s]  = (X_s[val_idx], y_s[val_idx])

        torch.manual_seed(42)
        global_ext = EEGNetFeatureExtractor().to(DEVICE)
        global_clf = GlobalClassifier(global_ext.feat_dim).to(DEVICE)
        ext_state = global_ext.state_dict()
        clf_state = global_clf.state_dict()

        best_val = 0.0
        best_ext_state = copy.deepcopy(ext_state)
        best_clf_state = copy.deepcopy(clf_state)
        patience_cnt = 0

        for rnd in range(1, FL_ROUNDS + 1):
            curr_frac = get_warmup_fraction(rnd, rho_0)

            client_ext_states, client_clf_states, n_selected = [], [], []
            for s in clients:
                X_tr, y_tr = client_data[s]
                if curr_frac < 1.0:
                    ent = compute_entropy_scores(global_ext, global_clf, X_tr, y_tr)
                    curr_idx = select_curriculum_indices(ent, curr_frac)
                else:
                    curr_idx = np.arange(len(y_tr))
                e_st, c_st, n_sel = client_local_train(ext_state, clf_state, X_tr, y_tr, curr_idx)
                client_ext_states.append(e_st)
                client_clf_states.append(c_st)
                n_selected.append(n_sel)

            ext_state = fed_avg_state(client_ext_states, n_selected)
            clf_state = fed_avg_state(client_clf_states, n_selected)
            global_ext.load_state_dict(ext_state)
            global_clf.load_state_dict(clf_state)

            val_accs = []
            for s in clients:
                Xv, yv = client_val[s]
                acc, _ = evaluate(global_ext, global_clf, Xv, yv)
                val_accs.append(acc)
            mean_val = np.mean(val_accs)

            if rnd % 20 == 0 or rnd == 1 or rnd == WARMUP_ROUNDS:
                print(f"    round {rnd:3d}/{FL_ROUNDS} | frac={curr_frac:.2f} "
                      f"(~{int(np.mean(n_selected))} trials/client) | mean val: {mean_val:.4f}")

            if mean_val > best_val:
                best_val = mean_val
                best_ext_state = copy.deepcopy(ext_state)
                best_clf_state = copy.deepcopy(clf_state)
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= FL_PATIENCE:
                print(f"    early stop @ round {rnd}")
                break

        global_ext.load_state_dict(best_ext_state)
        global_clf.load_state_dict(best_clf_state)
        acc, f1 = evaluate(global_ext, global_clf, X_target, y_target)
        print(f"  >> S{target}: best_val={best_val:.4f}  test_acc={acc:.4f}  f1={f1:.4f}")

        results[f"S{target}"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
        all_results[key] = results
        save_results(all_results)

    return results


def print_summary(all_results):
    print(f"\n{'='*70}\n  FEDCL WARMUP — SUMMARY\n{'='*70}")
    for rho in VARIANTS_TO_RUN:
        key = f"warmup_{int(rho*100)}"
        if key not in all_results:
            continue
        accs = [v["acc"] for v in all_results[key].values()]
        f1s = [v["f1"] for v in all_results[key].values()]
        print(f"  Warmup {int(rho*100)}%->100%: mean_acc={np.mean(accs):.4f}  "
              f"std={np.std(accs):.4f}  mean_f1={np.mean(f1s):.4f}")
    print(f"\n  Compare to (all genuine LOSO, verified this project):")
    print(f"    FedCL fixed-stage : 38.56%")
    print(f"    FedAvg (no alignment), 2-seed avg : 39.84%")
    print(f"    Original submission's warmup claims: 45.40% / 42.73% (personalised protocol, unverified)")


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    sessions = load_all_sessions()
    print(f"\nLoaded {len(sessions)} sessions.")

    all_results = load_existing_results()
    for rho_0 in VARIANTS_TO_RUN:
        run_loso_for_variant(sessions, rho_0, all_results)

    print_summary(all_results)
    print(f"\nFull results saved: {os.path.join(SAVE_DIR, 'fedcl_warmup_results.json')}")


if __name__ == "__main__":
    main()