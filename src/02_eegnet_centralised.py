"""
BCI-IV 2a EEGNet v3 — Clean Baseline (EOG Bug Fixed)
======================================================
Fixes from diagnostic:
  - EOG rejection applied AFTER bandpass (correct amplitude range 15–90 µV)
  - Both T and E sessions used in source pool (consistent with lambda sweep)
  - Full 288 trials per session (no spurious rejections)
  - Post-EMS amplitude noted but not changed (consistent across all runs)

Protocol:
  - LOSO cross-subject: test on each subject's E session
  - Source pool: all other subjects' T + E sessions
  - No domain adaptation (pure baseline)
  - Saves per-fold checkpoint — can resume if interrupted
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score, classification_report
import os, json, time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "/kaggle/input/datasets/abinayajone/bci-iv-2a-mi"
SAVE_DIR    = "/kaggle/working"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FS          = 250
T_START     = 2.5        # seconds after cue onset
T_END       = 6.0
EPOCH_LEN   = int((T_END - T_START) * FS)   # 875 samples
EOG_THRESH  = 100.0      # µV — applied AFTER bandpass
N_CLASSES   = 4
N_SUBJECTS  = 9

# Training
EPOCHS      = 300
BATCH_SIZE  = 32
LR          = 1e-3
PATIENCE    = 40         # early stopping

CHECKPOINT  = os.path.join(SAVE_DIR, "v3_clean_baseline_checkpoint.json")

print(f"Device     : {DEVICE}")
print(f"Epoch len  : {EPOCH_LEN} samples ({T_START}–{T_END}s @ {FS}Hz)")
print(f"EOG thresh : {EOG_THRESH} µV (post-bandpass)")
print(f"Epochs     : {EPOCHS}  Patience: {PATIENCE}  BS: {BATCH_SIZE}")


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def bandpass(data, lo=4, hi=40, fs=FS, order=4):
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


def load_session(path):
    """
    Load one .mat session.
    Returns (X, y) where X: (N, 1, 22, 875), y: (N,)
    EOG rejection applied AFTER bandpass — correct amplitude range.
    """
    mat  = loadmat(path, struct_as_record=False, squeeze_me=True)
    data = mat['data']

    X_list, y_list = [], []
    n_rejected = 0

    for run_idx in range(len(data)):
        run = data[run_idx]
        try:
            raw_X  = run.X.T        # (channels, time)
            raw_y  = run.y
            t_pos  = run.trial
            fs_run = run.fs
        except AttributeError:
            continue

        if not hasattr(raw_y, '__len__') or len(raw_y) == 0:
            continue

        # Bandpass FIRST (only EEG channels), THEN check amplitude
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

            # EOG rejection on bandpassed signal
            if np.max(np.abs(epoch)) > EOG_THRESH:
                n_rejected += 1
                continue

            epoch = exponential_moving_standardize(epoch)
            X_list.append(epoch)
            y_list.append(lbl - 1)   # 0-indexed

    if len(X_list) == 0:
        return None, None

    X = np.stack(X_list).astype(np.float32)[:, np.newaxis]  # (N,1,C,T)
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
                print(f"  WARNING: {fname} not found")
                continue
            X, y = load_session(fpath)
            if X is None:
                print(f"  WARNING: {fname} — no valid trials")
                continue
            key = f"S{s}{sess}"
            sessions[key] = (X, y)
            classes = [int((y == c).sum()) for c in range(N_CLASSES)]
            print(f"  {key}: {len(y)}/288 ({288 - len(y)} rej) classes={classes}")
    return sessions


# ─────────────────────────────────────────────
# MODEL — EEGNet v3 (identical to lambda sweep)
# ─────────────────────────────────────────────
class EEGNetV3(nn.Module):
    def __init__(self, n_classes=4, n_channels=22, n_times=875,
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

        self.classifier = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return self.classifier(x.flatten(1))


# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────
def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


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
            preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())
    return np.array(preds), np.array(labels)


def train_one_fold(model, train_loader, val_loader, fold_label):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc  = 0.0
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = nn.CrossEntropyLoss()(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_total   += len(yb)

        scheduler.step()

        val_acc = evaluate(model, val_loader)
        tr_acc  = tr_correct / tr_total
        tr_loss /= tr_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 25 == 0:
            print(f"    [{fold_label}] Ep {epoch:3d}/{EPOCHS} | "
                  f"Tr:{tr_loss:.4f}/{tr_acc:.4f} Vl:{val_acc:.4f}")

        if patience_cnt >= PATIENCE:
            print(f"    [{fold_label}] Early stop @ ep {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


# ─────────────────────────────────────────────
# LOSO
# ─────────────────────────────────────────────
def run_loso(sessions):
    # Load checkpoint
    completed = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            completed = json.load(f)
        print(f"\nResuming — completed folds: {list(completed.keys())}")

    results = dict(completed)

    for test_subj in range(1, N_SUBJECTS + 1):
        key = f"S{test_subj}"
        if key in results:
            print(f"  Skipping {key} (already done)")
            continue

        tgt_T = f"S{test_subj}T"
        tgt_E = f"S{test_subj}E"
        if tgt_T not in sessions or tgt_E not in sessions:
            print(f"  Skipping {key} — missing session")
            continue

        print(f"\n{'='*60}")
        print(f"Fold {test_subj}: Test={key}")

        # Build source pool — all other subjects T + E
        src_X_parts, src_y_parts = [], []
        for s in range(1, N_SUBJECTS + 1):
            if s == test_subj:
                continue
            for sess in ['T', 'E']:
                k = f"S{s}{sess}"
                if k in sessions:
                    src_X_parts.append(sessions[k][0])
                    src_y_parts.append(sessions[k][1])

        src_X = np.concatenate(src_X_parts)
        src_y = np.concatenate(src_y_parts)

        # 90/10 source train/val split
        n_val   = max(1, int(len(src_y) * 0.1))
        idx     = np.random.permutation(len(src_y))
        val_idx = idx[:n_val]
        tr_idx  = idx[n_val:]

        train_loader = make_loader(src_X[tr_idx], src_y[tr_idx], BATCH_SIZE)
        val_loader   = make_loader(src_X[val_idx], src_y[val_idx], BATCH_SIZE, shuffle=False)
        test_loader  = make_loader(sessions[tgt_E][0], sessions[tgt_E][1], BATCH_SIZE, shuffle=False)

        n_tgt_T = len(sessions[tgt_T][1])
        n_tgt_E = len(sessions[tgt_E][1])
        print(f"  Src train:{len(tr_idx)} val:{n_val} | Tgt T:{n_tgt_T} E:{n_tgt_E}")

        # Build model
        model = EEGNetV3().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  EEGNetV3: {n_params:,} params | feat_dim={model.feat_dim}")

        t0 = time.time()
        model, best_val_acc = train_one_fold(model, train_loader, val_loader, key)
        elapsed = time.time() - t0

        # Evaluate on test set
        test_acc = evaluate(model, test_loader)
        preds, labels = get_predictions(model, test_loader)
        f1  = f1_score(labels, preds, average='macro', zero_division=0)

        print(f"\n  {key}: Test Acc={test_acc:.4f}  F1={f1:.4f}  "
              f"(best val={best_val_acc:.4f})  [{elapsed:.0f}s]")
        print(classification_report(
            labels, preds,
            target_names=['Left Hand', 'Right Hand', 'Both Feet', 'Tongue'],
            zero_division=0
        ))

        results[key] = {
            "acc":      round(test_acc, 4),
            "f1":       round(f1, 4),
            "best_val": round(best_val_acc, 4),
            "n_test":   int(n_tgt_E),
            "elapsed":  round(elapsed, 1)
        }

        with open(CHECKPOINT, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Checkpoint saved ({len(results)} folds done)")

    return results


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def print_summary(results):
    print(f"\n{'='*60}")
    print(f"  EEGNet v3 CLEAN BASELINE — LOSO SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Subj':<8} {'Acc':>8} {'F1':>8} {'N':>6}")
    print(f"  {'-'*36}")

    accs, f1s = [], []
    for s in range(1, N_SUBJECTS + 1):
        k = f"S{s}"
        if k not in results:
            continue
        r = results[k]
        print(f"  {k:<8} {r['acc']:>8.4f} {r['f1']:>8.4f} {r['n_test']:>6}")
        accs.append(r['acc'])
        f1s.append(r['f1'])

    print(f"  {'-'*36}")
    if accs:
        print(f"  {'Mean':<8} {np.mean(accs):>8.4f} {np.mean(f1s):>8.4f}")
        print(f"  {'Std':<8} {np.std(accs):>8.4f} {np.std(f1s):>8.4f}")

    print(f"\n  --- Version comparison ---")
    print(f"  v1 EEGNet LOSO (old)       : 0.4780  [EOG bug status unknown]")
    print(f"  v3 EEGNet LOSO (old)       : 0.4277  [INVALID — EOG bug]")
    print(f"  v3 EEGNet LOSO (THIS RUN)  : {np.mean(accs):.4f}  [VALID]")
    print(f"  CORAL λ=10 LOSO            : 0.4358  [VALID]")
    print(f"  CORAL λ=1  LOSO            : 0.4335  [VALID]")

    print(f"\n  NOTE: Stage1 (no-adapt) averages from lambda sweep:")
    print(f"    λ=1  no-adapt avg : 0.4150")
    print(f"    λ=10 no-adapt avg : 0.4270")
    print(f"    λ=100 no-adapt avg: 0.3949")
    print(f"    λ=1000 no-adapt avg: 0.4180")
    print(f"  This run should align with those — confirms pipeline consistency.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("="*60)
    print("  BCI-IV 2a EEGNet v3 — CLEAN BASELINE")
    print("="*60)

    sessions = load_all_sessions()
    print(f"\nLoaded {len(sessions)} sessions.")

    results = run_loso(sessions)
    print_summary(results)

    # Save final results
    out_path = os.path.join(SAVE_DIR, "v3_clean_baseline_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()