"""
PhysioNet EEG Motor Movement/Imagery — FedAvg Global Model
===========================================================
Dataset: PhysioNet EEG Motor Movement/Imagery (109 subjects, 64 channels, 160Hz)
Task:    2-class motor imagery — Left Hand (T1) vs Right Hand (T2)
Runs:    R04, R08, R12 (imagined fist movement, 3 runs per subject)

Protocol:
  - Each subject = one FL client
  - Train on runs R04, R08 (split 80/20 train/val)
  - Test on run R12
  - FedAvg global model (best configuration from BCI-IV 2a experiments)
  - LOSO-style: final test accuracy averaged across all 109 subjects

Known bad subjects excluded:
  38, 88, 89, 92, 100, 104 — documented recording artifacts

Comparison:
  Mode 1: Centralised EEGNet (no FL)
  Mode 2: FedAvg global (proposed)
"""

# ── install MNE if not present ──────────────────────────────────────
import subprocess, sys
try:
    import mne
except ImportError:
    print("Installing MNE...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "mne", "--quiet"])
    import mne
mne.set_log_level('WARNING')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score, classification_report, accuracy_score
import os, json, copy, time
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "/kaggle/input/datasets/gamalasran/physionet-eeg-motor-movement-imagery/files"
SAVE_DIR    = "/kaggle/working"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FS          = 160          # PhysioNet sampling rate
T_START     = 0.0          # seconds after cue onset
T_END       = 4.0          # seconds (trial length)
EPOCH_LEN   = int((T_END - T_START) * FS)   # 640 samples

# Runs to use: imagined fist movement
MI_RUNS     = [4, 8, 12]   # R04, R08, R12
TRAIN_RUNS  = [4, 8]       # used for train/val
TEST_RUNS   = [12]         # held out for test

N_CLASSES   = 2            # T1=left hand, T2=right hand
N_CHANNELS  = 64

# Subjects with known recording problems — exclude
BAD_SUBJECTS = {38, 88, 89, 92, 100, 104}

# Training
FL_ROUNDS    = 100
LOCAL_EPOCHS = 5
BATCH_SIZE   = 32
LR           = 1e-3
FL_PATIENCE  = 25

print(f"Device      : {DEVICE}")
print(f"FS          : {FS} Hz")
print(f"Epoch len   : {EPOCH_LEN} samples ({T_START}–{T_END}s)")
print(f"FL rounds   : {FL_ROUNDS}  Local epochs: {LOCAL_EPOCHS}")
print(f"Bad subjects excluded: {sorted(BAD_SUBJECTS)}")


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def bandpass(data, lo=8, hi=30, fs=FS, order=4):
    """8-30 Hz mu + beta bands."""
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=-1)


def exponential_moving_standardize(data, decay=0.999, eps=1e-6):
    """Channel-wise EMS. data: (C, T)"""
    out  = np.zeros_like(data)
    mean = np.zeros(data.shape[0])
    var  = np.ones(data.shape[0])
    for t in range(data.shape[1]):
        mean = decay * mean + (1 - decay) * data[:, t]
        var  = decay * var  + (1 - decay) * (data[:, t] - mean) ** 2
        out[:, t] = (data[:, t] - mean) / (np.sqrt(var) + eps)
    return out


def load_subject_runs(subj_id, run_ids):
    """
    Load specified runs for one subject.
    Returns (X, y) where:
      X: (N, 1, 64, 640)
      y: (N,) with 0=left hand, 1=right hand
    """
    subj_str = f"S{subj_id:03d}"
    subj_dir = os.path.join(DATA_DIR, subj_str)

    X_list, y_list = [], []

    for run_id in run_ids:
        fname = f"{subj_str}R{run_id:02d}.edf"
        fpath = os.path.join(subj_dir, fname)

        if not os.path.exists(fpath):
            continue

        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
        except Exception as e:
            print(f"  WARNING: Could not read {fname}: {e}")
            continue

        # Get events from annotations
        events, event_id = mne.events_from_annotations(
            raw, verbose=False)

        # T1 = left fist imagery, T2 = right fist imagery
        # Event IDs vary by run — map to 0/1
        t1_id = event_id.get('T1', None)
        t2_id = event_id.get('T2', None)

        if t1_id is None or t2_id is None:
            continue

        # Get EEG data — all 64 channels
        data = raw.get_data()  # (64, T)

        # Bandpass filter
        data = bandpass(data, lo=8, hi=30)

        # Extract epochs around each event
        sfreq = raw.info['sfreq']

        for event in events:
            onset_sample = event[0]
            event_code   = event[2]

            if event_code not in [t1_id, t2_id]:
                continue

            s = int(onset_sample + T_START * sfreq)
            e = int(onset_sample + T_END   * sfreq)

            if e > data.shape[1]:
                continue

            epoch = data[:, s:e]

            # Ensure correct length
            if epoch.shape[1] < EPOCH_LEN:
                continue
            epoch = epoch[:, :EPOCH_LEN]

            # EMS normalisation
            epoch = exponential_moving_standardize(epoch)

            X_list.append(epoch.astype(np.float32))
            y_list.append(0 if event_code == t1_id else 1)

    if len(X_list) == 0:
        return None, None

    X = np.stack(X_list)[:, np.newaxis]   # (N, 1, 64, 640)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def load_all_subjects():
    """Load all valid subjects. Returns dict {subj_id: (X_train, y_train, X_test, y_test)}"""
    print(f"\nLoading {109 - len(BAD_SUBJECTS)} subjects "
          f"(excluding {sorted(BAD_SUBJECTS)})...")

    subjects = {}
    failed   = []

    for sid in range(1, 110):
        if sid in BAD_SUBJECTS:
            continue

        X_tr, y_tr = load_subject_runs(sid, TRAIN_RUNS)
        X_te, y_te = load_subject_runs(sid, TEST_RUNS)

        if X_tr is None or X_te is None:
            failed.append(sid)
            continue
        if len(y_tr) < 10 or len(y_te) < 5:
            failed.append(sid)
            continue

        subjects[sid] = (X_tr, y_tr, X_te, y_te)

        c0 = int((y_tr == 0).sum())
        c1 = int((y_tr == 1).sum())
        print(f"  S{sid:03d}: train={len(y_tr)} "
              f"(L:{c0} R:{c1})  test={len(y_te)}")

    print(f"\nLoaded {len(subjects)} subjects. "
          f"Failed/skipped: {failed}")
    return subjects


# ─────────────────────────────────────────────
# MODEL — EEGNet adapted for PhysioNet
# 64 channels, 640 time points, 2 classes
# ─────────────────────────────────────────────
class EEGNetPhysioNet(nn.Module):
    def __init__(self, n_classes=2, n_channels=64, n_times=640,
                 F1=8, D=2, F2=16, kern_len=80, drop_rate=0.5):
        """
        kern_len=80 = 0.5s at 160Hz (half sampling rate, standard EEGNet)
        """
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len),
                      padding=(0, kern_len // 2), bias=False),
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

        self.classifier = nn.Linear(self.feat_dim, n_classes)
        print(f"  EEGNetPhysioNet: feat_dim={self.feat_dim}  "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return self.classifier(x.flatten(1))


# ─────────────────────────────────────────────
# FEDERATED AVERAGING
# ─────────────────────────────────────────────
def fed_avg(global_model, client_states, client_weights):
    total        = sum(client_weights)
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.zeros_like(
            global_state[key], dtype=torch.float32)
        for state, w in zip(client_states, client_weights):
            global_state[key] += (w / total) * state[key].float()
    global_model.load_state_dict(global_state)
    return global_model


# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────
def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
    return correct / total if total > 0 else 0.0


def get_predictions(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(yb.numpy())
    return np.array(preds), np.array(labels)


def client_local_train(global_model, train_loader):
    model     = copy.deepcopy(global_model).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(LOCAL_EPOCHS):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model.state_dict()


# ─────────────────────────────────────────────
# CENTRALISED BASELINE
# ─────────────────────────────────────────────
def run_centralised(subjects):
    print(f"\n{'='*60}")
    print(f"  CENTRALISED EEGNET BASELINE")
    print(f"{'='*60}")

    # Pool all training data
    all_X = np.concatenate([v[0] for v in subjects.values()])
    all_y = np.concatenate([v[1] for v in subjects.values()])

    print(f"  Total training trials: {len(all_y)}")

    # 90/10 split
    idx   = np.random.permutation(len(all_y))
    n_val = max(1, int(len(all_y) * 0.1))
    val_i = idx[:n_val]
    tr_i  = idx[n_val:]

    train_loader = make_loader(all_X[tr_i], all_y[tr_i], BATCH_SIZE)
    val_loader   = make_loader(all_X[val_i], all_y[val_i],
                               BATCH_SIZE, shuffle=False)

    model     = EEGNetPhysioNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200)

    best_val    = 0.0
    best_state  = None
    patience_cnt = 0
    EPOCHS      = 200
    PATIENCE    = 30

    print(f"  Training for up to {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_acc = evaluate(model, val_loader)
        if val_acc > best_val:
            best_val   = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 25 == 0:
            print(f"    Ep {epoch}/{EPOCHS} | Val: {val_acc:.4f}")

        if patience_cnt >= PATIENCE:
            print(f"    Early stop @ ep {epoch}")
            break

    model.load_state_dict(best_state)

    # Evaluate per subject
    print(f"\n  Per-subject test accuracy:")
    accs, f1s = [], []
    for sid, (_, _, X_te, y_te) in sorted(subjects.items()):
        test_loader = make_loader(X_te, y_te, BATCH_SIZE, shuffle=False)
        preds, labels = get_predictions(model, test_loader)
        acc = float((preds == labels).mean())
        f1  = float(f1_score(labels, preds,
                              average='macro', zero_division=0))
        accs.append(acc)
        f1s.append(f1)

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    mean_f1  = np.mean(f1s)
    print(f"  Mean acc: {mean_acc:.4f} ± {std_acc:.4f}  F1: {mean_f1:.4f}")
    return {"mean_acc": round(mean_acc, 4),
            "std_acc":  round(std_acc, 4),
            "mean_f1":  round(mean_f1, 4)}


# ─────────────────────────────────────────────
# FEDAVG GLOBAL
# ─────────────────────────────────────────────
def run_fedavg(subjects):
    print(f"\n{'='*60}")
    print(f"  FEDAVG GLOBAL — {len(subjects)} clients")
    print(f"{'='*60}")

    # Build per-client loaders
    client_train = {}
    client_val   = {}
    client_test  = {}
    client_n     = {}
    clients      = sorted(subjects.keys())

    for sid in clients:
        X_tr, y_tr, X_te, y_te = subjects[sid]

        n_val = max(1, int(len(y_tr) * 0.2))
        idx   = np.random.permutation(len(y_tr))
        val_i = idx[:n_val]
        tr_i  = idx[n_val:]

        client_train[sid] = make_loader(
            X_tr[tr_i], y_tr[tr_i], BATCH_SIZE)
        client_val[sid]   = make_loader(
            X_tr[val_i], y_tr[val_i], BATCH_SIZE, shuffle=False)
        client_test[sid]  = make_loader(
            X_te, y_te, BATCH_SIZE, shuffle=False)
        client_n[sid]     = len(tr_i)

    print(f"  Clients: {len(clients)}")
    print(f"  Avg trials per client: "
          f"{np.mean([client_n[s] for s in clients]):.0f}")

    global_model = EEGNetPhysioNet().to(DEVICE)

    best_mean_val  = 0.0
    best_round     = 0
    best_state     = copy.deepcopy(global_model.state_dict())
    patience_cnt   = 0
    val_history    = []

    print(f"\n  Starting federation — {FL_ROUNDS} rounds...")

    for rnd in range(1, FL_ROUNDS + 1):
        t0 = time.time()

        client_states = []
        weights       = []

        for sid in clients:
            state = client_local_train(
                global_model, client_train[sid])
            client_states.append(state)
            weights.append(client_n[sid])

        global_model = fed_avg(global_model, client_states, weights)

        # Evaluate on subset of clients for speed (every round)
        # Use all clients every 10 rounds, subset otherwise
        if rnd % 10 == 0 or rnd == 1:
            val_accs = [evaluate(global_model, client_val[s])
                        for s in clients]
        else:
            # Sample 20 clients for speed
            sample  = np.random.choice(clients, min(20, len(clients)),
                                        replace=False)
            val_accs = [evaluate(global_model, client_val[s])
                        for s in sample]

        mean_val = np.mean(val_accs)
        val_history.append(mean_val)
        elapsed  = time.time() - t0

        if rnd % 10 == 0 or rnd == 1:
            print(f"  Round {rnd:3d}/{FL_ROUNDS} | "
                  f"Mean val: {mean_val:.4f} | [{elapsed:.1f}s]")

        if mean_val > best_mean_val:
            best_mean_val = mean_val
            best_round    = rnd
            best_state    = copy.deepcopy(global_model.state_dict())
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if patience_cnt >= FL_PATIENCE:
            print(f"\n  Early stop @ round {rnd} "
                  f"(best round={best_round})")
            break

    global_model.load_state_dict(best_state)
    print(f"\n  Best round: {best_round}  "
          f"Best mean val: {best_mean_val:.4f}")

    # Final evaluation on all subjects
    print(f"\n  Final test evaluation:")
    accs, f1s = [], []
    per_subject = {}

    for sid in clients:
        preds, labels = get_predictions(
            global_model, client_test[sid])
        acc = float((preds == labels).mean())
        f1  = float(f1_score(labels, preds,
                              average='macro', zero_division=0))
        accs.append(acc)
        f1s.append(f1)
        per_subject[f"S{sid:03d}"] = {
            "acc": round(acc, 4), "f1": round(f1, 4)}

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    mean_f1  = np.mean(f1s)

    print(f"  Mean acc: {mean_acc:.4f} ± {std_acc:.4f}  "
          f"F1: {mean_f1:.4f}")
    print(f"\n  Per-subject breakdown (first 20):")
    print(f"  {'Subj':<8} {'Acc':>8} {'F1':>8}")
    print(f"  {'-'*28}")
    for sid in sorted(clients)[:20]:
        r = per_subject[f"S{sid:03d}"]
        print(f"  S{sid:03d}    {r['acc']:>8.4f} {r['f1']:>8.4f}")
    if len(clients) > 20:
        print(f"  ... ({len(clients)-20} more subjects)")

    return {
        "mean_acc":    round(mean_acc, 4),
        "std_acc":     round(std_acc, 4),
        "mean_f1":     round(mean_f1, 4),
        "best_round":  best_round,
        "per_subject": per_subject,
        "val_history": val_history
    }


# ─────────────────────────────────────────────
# FINAL COMPARISON
# ─────────────────────────────────────────────
def print_comparison(cent_results, fedavg_results):
    print(f"\n{'='*70}")
    print(f"  PHYSIONET RESULTS — 109-subject, 2-class MI")
    print(f"  Left Hand vs Right Hand imagery")
    print(f"{'='*70}")
    print(f"  {'Method':<35} {'Privacy':>7} {'Acc':>10} {'F1':>8}")
    print(f"  {'-'*65}")

    rows = [
        ("Centralised EEGNet",   "✗", cent_results),
        ("FedAvg global (proposed)", "✓", fedavg_results),
    ]
    for name, priv, r in rows:
        print(f"  {name:<35} {priv:>7} "
              f"{r['mean_acc']:>8.4f}±{r['std_acc']:.3f} "
              f"{r['mean_f1']:>8.4f}")

    delta = fedavg_results['mean_acc'] - cent_results['mean_acc']
    arrow = "▲" if delta >= 0 else "▼"
    print(f"\n  FedAvg vs Centralised: {arrow}{abs(delta):.4f}")
    print(f"  Chance (2-class): 0.5000")
    print(f"\n  --- BCI-IV 2a reference (4-class) ---")
    print(f"  Centralised EEGNet : 0.4100")
    print(f"  FedAvg global      : 0.4667  (+5.7%)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("="*60)
    print("  PhysioNet EEG — FedAvg Global Model")
    print(f"  Device: {DEVICE}")
    print("="*60)

    # Load all subjects
    subjects = load_all_subjects()
    print(f"\nTotal valid subjects: {len(subjects)}")

    # Run centralised baseline
    cent_results = run_centralised(subjects)

    # Run FedAvg
    fedavg_results = run_fedavg(subjects)

    # Print comparison
    print_comparison(cent_results, fedavg_results)

    # Save
    out = {
        "dataset":        "PhysioNet EEG Motor Imagery",
        "n_subjects":     len(subjects),
        "task":           "2-class MI (left hand vs right hand)",
        "centralised":    cent_results,
        "fedavg_global":  fedavg_results
    }
    out_path = os.path.join(SAVE_DIR, "physionet_results.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()