"""
BCI-IV 2a — FedAvg Global Model + CORAL
=========================================
Tests the one combination not yet tried:
  FedAvg global model (shared classifier) + CORAL alignment

Previous results:
  FedAvg global (no CORAL)     : 0.4613  ← best so far
  FedAvg local heads + CORAL   : 0.4316
  Centralised + CORAL          : 0.4358

Hypothesis: adding CORAL to the best-performing setup (FedAvg global)
may push accuracy above 0.4613 by aligning client feature distributions
toward the global centroid during local training.

Architecture:
  - Global feature extractor  → FedAvg aggregated
  - Global shared classifier  → FedAvg aggregated
  - CORAL                     → class-conditional alignment during local train
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score, classification_report
import os, json, copy, time
from collections import defaultdict

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

FL_ROUNDS    = 100
LOCAL_EPOCHS = 5
BATCH_SIZE   = 32
LR           = 1e-3
FL_PATIENCE  = 20

# Run both with and without CORAL for clean comparison
LAMBDAS = [0, 10]   # 0 = no CORAL (reproduce baseline), 10 = CORAL

print(f"Device      : {DEVICE}")
print(f"FL rounds   : {FL_ROUNDS}  Local epochs: {LOCAL_EPOCHS}")
print(f"Testing λ   : {LAMBDAS}")


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
            e = int(onset + T_END   * fs_run)
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
            key = f"S{s}{sess}"
            sessions[key] = (X, y)
            classes = [int((y == c).sum()) for c in range(N_CLASSES)]
            print(f"  {key}: {len(y)}/288 ({288-len(y)} rej) classes={classes}")
    return sessions


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class EEGNetFeatureExtractor(nn.Module):
    def __init__(self, n_channels=22, n_times=875,
                 F1=8, D=2, F2=16, kern_len=32, drop_rate=0.5):
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
        return x.flatten(1)


class GlobalClassifier(nn.Module):
    def __init__(self, feat_dim, n_classes=N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, feat):
        return self.fc(feat)


# ─────────────────────────────────────────────
# CORAL LOSS (class-conditional)
# ─────────────────────────────────────────────
def class_conditional_coral_loss(local_feat, local_labels,
                                  global_centroids, lam):
    if not global_centroids or lam == 0:
        return torch.tensor(0.0, device=local_feat.device)

    total_loss       = torch.tensor(0.0, device=local_feat.device)
    n_classes_present = 0

    for c in range(N_CLASSES):
        if c not in global_centroids:
            continue
        mask = (local_labels == c)
        if mask.sum() < 2:
            continue

        local_c  = local_feat[mask]
        global_c = torch.FloatTensor(
            global_centroids[c]).to(local_feat.device)
        global_c = global_c.unsqueeze(0).expand(local_c.size(0), -1)

        noise    = torch.randn_like(local_c) * 0.05
        target_c = global_c + noise

        n = local_c.size(0)
        d = local_c.size(1)

        src = local_c  - local_c.mean(0, keepdim=True)
        tgt = target_c - target_c.mean(0, keepdim=True)

        Cs   = (src.T @ src) / max(n - 1, 1)
        Ct   = (tgt.T @ tgt) / max(n - 1, 1)
        loss = torch.norm(Cs - Ct, p='fro') ** 2 / (4 * d * d)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss        = total_loss + loss
            n_classes_present += 1

    if n_classes_present > 0:
        total_loss = total_loss / n_classes_present

    return lam * total_loss


# ─────────────────────────────────────────────
# FEDERATED AVERAGING
# ─────────────────────────────────────────────
def fed_avg_model(global_model, client_states, client_weights):
    total        = sum(client_weights)
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.zeros_like(
            global_state[key], dtype=torch.float32)
        for state, w in zip(client_states, client_weights):
            global_state[key] += (w / total) * state[key].float()
    global_model.load_state_dict(global_state)
    return global_model


def aggregate_centroids(client_centroids_list, client_weights):
    total    = sum(client_weights)
    combined = {}
    counts   = defaultdict(float)
    for centroids, w in zip(client_centroids_list, client_weights):
        for c, centroid in centroids.items():
            if c not in combined:
                combined[c] = np.zeros_like(centroid, dtype=np.float64)
            combined[c] += (w / total) * centroid
            counts[c]   += w / total
    for c in combined:
        if counts[c] > 0:
            combined[c] /= counts[c]
    return combined


# ─────────────────────────────────────────────
# COMPUTE LOCAL CLASS CENTROIDS
# ─────────────────────────────────────────────
def compute_local_centroids(extractor, train_loader):
    extractor.eval()
    class_feats = defaultdict(list)
    with torch.no_grad():
        for xb, yb in train_loader:
            feat = extractor(xb.to(DEVICE)).cpu().numpy()
            for f, label in zip(feat, yb.numpy()):
                class_feats[int(label)].append(f)
    return {c: np.mean(feats, axis=0) for c, feats in class_feats.items()}


# ─────────────────────────────────────────────
# CLIENT LOCAL TRAINING
# ─────────────────────────────────────────────
def client_local_train(extractor, classifier, train_loader,
                       global_centroids, lam):
    extractor  = copy.deepcopy(extractor).to(DEVICE)
    classifier = copy.deepcopy(classifier).to(DEVICE)

    params    = list(extractor.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    extractor.train()
    classifier.train()

    for epoch in range(LOCAL_EPOCHS):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()

            feat = extractor(xb)
            out  = classifier(feat)
            ce   = criterion(out, yb)

            loss = ce
            if lam > 0 and global_centroids:
                c_loss = class_conditional_coral_loss(
                    feat, yb, global_centroids, lam)
                if not (torch.isnan(c_loss) or torch.isinf(c_loss)):
                    loss = ce + c_loss

            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

    local_centroids = compute_local_centroids(extractor, train_loader)
    return extractor.state_dict(), classifier.state_dict(), local_centroids


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(extractor, classifier, loader):
    extractor.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (classifier(extractor(xb)).argmax(1) == yb).sum().item()
            total   += len(yb)
    return correct / total if total > 0 else 0.0


def get_predictions(extractor, classifier, loader):
    extractor.eval()
    classifier.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = classifier(extractor(xb.to(DEVICE)))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(yb.numpy())
    return np.array(preds), np.array(labels)


# ─────────────────────────────────────────────
# MAIN FL LOOP
# ─────────────────────────────────────────────
def run_federated_global(sessions, lam):
    mode_name = f"FedAvg_global_CORAL_lam{lam}"
    coral_str = f"λ={lam}" if lam > 0 else "disabled"

    print(f"\n{'#'*70}")
    print(f"# MODE: FedAvg Global Model")
    print(f"#   CORAL: {coral_str}")
    print(f"{'#'*70}")

    # Build data loaders
    client_train_loaders = {}
    client_val_loaders   = {}
    client_test_loaders  = {}
    client_n_train       = {}
    clients              = []

    for s in range(1, N_SUBJECTS + 1):
        tgt_T = f"S{s}T"
        tgt_E = f"S{s}E"
        if tgt_T not in sessions or tgt_E not in sessions:
            continue

        X_T, y_T = sessions[tgt_T]
        X_E, y_E = sessions[tgt_E]

        n_val = max(1, int(len(y_T) * 0.2))
        idx   = np.random.permutation(len(y_T))
        val_i = idx[:n_val]
        tr_i  = idx[n_val:]

        client_train_loaders[s] = DataLoader(
            TensorDataset(torch.FloatTensor(X_T[tr_i]),
                          torch.LongTensor(y_T[tr_i])),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )
        client_val_loaders[s] = DataLoader(
            TensorDataset(torch.FloatTensor(X_T[val_i]),
                          torch.LongTensor(y_T[val_i])),
            batch_size=BATCH_SIZE, shuffle=False
        )
        client_test_loaders[s] = DataLoader(
            TensorDataset(torch.FloatTensor(X_E), torch.LongTensor(y_E)),
            batch_size=BATCH_SIZE, shuffle=False
        )
        client_n_train[s] = len(tr_i)
        clients.append(s)

    print(f"\n  Clients: {clients}")
    for s in clients:
        print(f"    S{s}: train={client_n_train[s]} "
              f"val={len(client_val_loaders[s].dataset)} "
              f"test={len(client_test_loaders[s].dataset)}")

    # Initialise global model (extractor + shared classifier)
    global_extractor  = EEGNetFeatureExtractor().to(DEVICE)
    global_classifier = GlobalClassifier(global_extractor.feat_dim).to(DEVICE)
    feat_dim          = global_extractor.feat_dim

    n_params = (sum(p.numel() for p in global_extractor.parameters()) +
                sum(p.numel() for p in global_classifier.parameters()))
    print(f"\n  Total params: {n_params:,}  feat_dim: {feat_dim}")

    global_centroids = {}

    best_mean_val     = 0.0
    best_round        = 0
    best_ext_state    = copy.deepcopy(global_extractor.state_dict())
    best_clf_state    = copy.deepcopy(global_classifier.state_dict())
    patience_cnt      = 0
    val_history       = []

    print(f"\n  Starting federation — {FL_ROUNDS} rounds...")

    for rnd in range(1, FL_ROUNDS + 1):
        t0 = time.time()

        client_ext_states  = []
        client_clf_states  = []
        client_centroids_l = []

        for s in clients:
            ext_state, clf_state, local_centroids = client_local_train(
                extractor        = global_extractor,
                classifier       = global_classifier,
                train_loader     = client_train_loaders[s],
                global_centroids = global_centroids,
                lam              = lam
            )
            client_ext_states.append(ext_state)
            client_clf_states.append(clf_state)
            client_centroids_l.append(local_centroids)

        # FedAvg on both extractor and classifier
        weights = [client_n_train[s] for s in clients]
        global_extractor  = fed_avg_model(
            global_extractor, client_ext_states, weights)
        global_classifier = fed_avg_model(
            global_classifier, client_clf_states, weights)

        # Update global centroids
        if lam > 0:
            global_centroids = aggregate_centroids(
                client_centroids_l, weights)

        # Evaluate
        val_accs = [
            evaluate(global_extractor, global_classifier,
                     client_val_loaders[s]) for s in clients
        ]
        mean_val = np.mean(val_accs)
        val_history.append(mean_val)

        elapsed = time.time() - t0

        if rnd % 10 == 0 or rnd == 1:
            print(f"  Round {rnd:3d}/{FL_ROUNDS} | "
                  f"Mean val: {mean_val:.4f} | "
                  f"Per-client: {[f'{a:.3f}' for a in val_accs]} | "
                  f"[{elapsed:.1f}s]")

        if mean_val > best_mean_val:
            best_mean_val  = mean_val
            best_round     = rnd
            best_ext_state = copy.deepcopy(global_extractor.state_dict())
            best_clf_state = copy.deepcopy(global_classifier.state_dict())
            patience_cnt   = 0
        else:
            patience_cnt += 1

        if patience_cnt >= FL_PATIENCE:
            print(f"\n  Early stop @ round {rnd} (best round={best_round})")
            break

    # Restore best
    global_extractor.load_state_dict(best_ext_state)
    global_classifier.load_state_dict(best_clf_state)

    print(f"\n  Best round: {best_round}  Best mean val: {best_mean_val:.4f}")

    # Final evaluation
    print(f"\n  Final test evaluation (E sessions):")
    print(f"  {'Subj':<8} {'Acc':>8} {'F1':>8} {'N':>6}")
    print(f"  {'-'*34}")

    results   = {}
    test_accs = []
    test_f1s  = []

    for s in clients:
        preds, labels = get_predictions(
            global_extractor, global_classifier, client_test_loaders[s])
        acc = float((preds == labels).mean())
        f1  = float(f1_score(labels, preds, average='macro', zero_division=0))
        n   = len(labels)
        results[f"S{s}"] = {"acc": round(acc, 4), "f1": round(f1, 4), "n": n}
        test_accs.append(acc)
        test_f1s.append(f1)
        print(f"  S{s:<7} {acc:>8.4f} {f1:>8.4f} {n:>6}")

    mean_acc = np.mean(test_accs)
    std_acc  = np.std(test_accs)
    mean_f1  = np.mean(test_f1s)

    print(f"  {'-'*34}")
    print(f"  {'Mean':<8} {mean_acc:>8.4f} {mean_f1:>8.4f}")
    print(f"  {'Std':<8} {std_acc:>8.4f}")

    # Detailed reports
    print(f"\n  Detailed per-subject reports:")
    for s in clients:
        preds, labels = get_predictions(
            global_extractor, global_classifier, client_test_loaders[s])
        print(f"\n  S{s}:")
        print(classification_report(
            labels, preds,
            target_names=['Left Hand', 'Right Hand', 'Both Feet', 'Tongue'],
            zero_division=0
        ))

    out = {
        "mode":          mode_name,
        "coral_lambda":  lam,
        "best_round":    best_round,
        "best_mean_val": round(best_mean_val, 4),
        "mean_test_acc": round(mean_acc, 4),
        "std_test_acc":  round(std_acc, 4),
        "mean_test_f1":  round(mean_f1, 4),
        "per_subject":   results,
        "val_history":   val_history
    }

    out_path = os.path.join(SAVE_DIR, f"fl_{mode_name}_results.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return out


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
def print_final_summary(all_results):
    print(f"\n{'='*72}")
    print(f"  COMPLETE RESULTS — BCI-IV 2a, 9-subject LOSO")
    print(f"{'='*72}")
    print(f"  {'Method':<42} {'Privacy':>7} {'Acc':>8} {'vs Base':>10}")
    print(f"  {'-'*70}")

    base = 0.4100
    refs = [
        ("Centralised EEGNet (no adapt)",           "✗", 0.4100),
        ("Centralised + CORAL (λ=10)",               "✗", 0.4358),
        ("FedAvg global (prev run)",                 "✓", 0.4613),
        ("FedAvg local heads + class CORAL",         "✓", 0.4316),
    ]
    for name, priv, acc in refs:
        d = acc - base
        arrow = f"▲{d:.4f}" if d > 0 else (f"▼{abs(d):.4f}" if d < 0 else "─")
        print(f"  {name:<42} {priv:>7} {acc:>8.4f} {arrow:>10}")

    print(f"  {'-'*70}")
    for res in all_results:
        d     = res['mean_test_acc'] - base
        arrow = f"▲{d:.4f}" if d > 0 else f"▼{abs(d):.4f}"
        coral = f"CORAL λ={res['coral_lambda']}" if res['coral_lambda'] > 0 \
                else "no CORAL"
        tag   = "← BEST" if res['mean_test_acc'] > 0.4613 else ""
        print(f"  {'FedAvg global + '+coral:<42} {'✓':>7} "
              f"{res['mean_test_acc']:>8.4f} {arrow:>10}  {tag}")

    print(f"\n  Chance (4-class): 0.2500")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("="*70)
    print("  BCI-IV 2a — FedAvg Global + CORAL")
    print("="*70)

    sessions = load_all_sessions()
    print(f"\nLoaded {len(sessions)} sessions.")

    all_results = []

    for lam in LAMBDAS:
        res = run_federated_global(sessions, lam)
        all_results.append(res)

    print_final_summary(all_results)

    summary_path = os.path.join(SAVE_DIR, "fl_global_coral_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved: {summary_path}")


if __name__ == "__main__":
    main()