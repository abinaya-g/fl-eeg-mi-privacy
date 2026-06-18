"""
BCI-IV 2a EEGNet v3 + CORAL — Lambda Sweep
Tests λ ∈ {1, 10, 100, 1000} to find optimal CORAL weight.
Runs full LOSO for each λ. Saves checkpoint per λ so can resume if interrupted.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import os, time, json
from sklearn.metrics import f1_score, classification_report

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/datasets/abinayajone/bci-iv-2a-mi"
SAVE_DIR   = "/kaggle/working"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDAS    = [1, 10, 100, 1000]   # sweep values

FS         = 250
T_START    = 2.5   # seconds after cue
T_END      = 6.0
EPOCH_LEN  = int((T_END - T_START) * FS)   # 875 samples

EOG_THRESH = 100.0   # µV — reject trial if any channel exceeds this
N_CLASSES  = 4
N_SUBJECTS = 9

# Training
STAGE1_EPOCHS  = 200
STAGE2_EPOCHS  = 150
BATCH_SIZE_SRC = 32
BATCH_SIZE_TGT = 32
LR             = 1e-3
PATIENCE       = 30   # early stopping patience

print(f"Device: {DEVICE}")
print(f"Lambda sweep: {LAMBDAS}")


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def bandpass(data, lo=4, hi=40, fs=FS, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, data, axis=-1)


def exponential_moving_standardize(data, decay=0.999, eps=1e-6):
    """Channel-wise EMS. data: (C, T)"""
    out = np.zeros_like(data)
    mean = np.zeros(data.shape[0])
    var  = np.ones(data.shape[0])
    for t in range(data.shape[1]):
        mean = decay * mean + (1 - decay) * data[:, t]
        var  = decay * var  + (1 - decay) * (data[:, t] - mean)**2
        out[:, t] = (data[:, t] - mean) / (np.sqrt(var) + eps)
    return out


def load_session(path):
    """Load one .mat session → (X, y) after preprocessing + EOG rejection."""
    mat   = loadmat(path, struct_as_record=False, squeeze_me=True)
    data  = mat['data']
    X_list, y_list = [], []

    for run_idx in range(len(data)):
        run = data[run_idx]
        try:
            raw_X  = run.X.T          # (C, T)  — transpose to channels-first
            raw_y  = run.y            # trial labels
            t_pos  = run.trial        # trial onset samples
            fs_run = run.fs
        except AttributeError:
            continue

        if not hasattr(raw_y, '__len__') or len(raw_y) == 0:
            continue

        raw_X = bandpass(raw_X[:22])  # keep only EEG channels (drop EOG)

        for i, (onset, label) in enumerate(zip(t_pos, raw_y)):
            if label < 1 or label > 4:
                continue
            s = int(onset + T_START * fs_run)
            e = int(onset + T_END   * fs_run)
            if e > raw_X.shape[1]:
                continue
            epoch = raw_X[:, s:e]
            if epoch.shape[1] != EPOCH_LEN:
                continue
            if np.max(np.abs(epoch)) > EOG_THRESH:
                continue   # EOG artifact rejection
            epoch = exponential_moving_standardize(epoch)
            X_list.append(epoch)
            y_list.append(label - 1)   # 0-indexed

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
                print(f"  WARNING: {fname} returned no trials")
                continue
            key = f"S{s}{sess}"
            sessions[key] = (X, y)
            classes = [int((y==c).sum()) for c in range(N_CLASSES)]
            print(f"  {key}: {len(y)}/288 ({288-len(y)} rej) classes={classes}")
    return sessions


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class EEGNetV3(nn.Module):
    def __init__(self, n_classes=4, n_channels=22, n_times=875,
                 F1=8, D=2, F2=16, kern_len=32, drop_rate=0.5):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len//2), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_rate)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop_rate)
        )
        # compute feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            x = self.temporal(dummy)
            x = self.depthwise(x)
            x = self.separable(x)
            self.feat_dim = x.numel()

        self.classifier = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x, return_features=False):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        feat = x.flatten(1)
        if return_features:
            return feat
        return self.classifier(feat)

    def forward_both(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        feat = x.flatten(1)
        return feat, self.classifier(feat)


# ─────────────────────────────────────────────
# CORAL LOSS
# ─────────────────────────────────────────────
def coral_loss(source_feat, target_feat):
    """Proper CORAL: aligns second-order statistics (covariance matrices)."""
    ns = source_feat.size(0)
    nt = target_feat.size(0)
    d  = source_feat.size(1)

    # Mean-center
    src = source_feat - source_feat.mean(0, keepdim=True)
    tgt = target_feat - target_feat.mean(0, keepdim=True)

    # Covariance matrices
    Cs = (src.T @ src) / (ns - 1)
    Ct = (tgt.T @ tgt) / (nt - 1)

    loss = torch.norm(Cs - Ct, p='fro') ** 2 / (4 * d * d)

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=source_feat.device, requires_grad=True)
    return loss


# ─────────────────────────────────────────────
# CORAL LOSS SELF-TEST
# ─────────────────────────────────────────────
def test_coral_loss():
    print("\nTesting CORAL loss function...")
    src = torch.randn(32, 64)
    tgt = torch.randn(32, 64) * 2 + 1
    loss = coral_loss(src, tgt)
    val  = loss.item()
    status = "OK — non-zero" if val > 1e-6 else "WARNING — near zero"
    print(f"  CORAL loss test: {val:.6f} ({status})")
    assert val > 1e-6, "CORAL loss is zero — check implementation"
    print("  CORAL loss verified non-zero\n")


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
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total   += len(yb)
    return correct / total if total > 0 else 0.0


def get_predictions(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            pred = model(xb).argmax(1).cpu().numpy()
            preds.extend(pred)
            labels.extend(yb.numpy())
    return np.array(preds), np.array(labels)


# ─────────────────────────────────────────────
# STAGE 1: SOURCE PRETRAINING
# ─────────────────────────────────────────────
def stage1_pretrain(model, src_train_loader, src_val_loader, subject_label):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS)

    best_val_acc  = 0.0
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, STAGE1_EPOCHS + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in src_train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_total   += len(yb)
        scheduler.step()

        val_acc = evaluate(model, src_val_loader)
        tr_acc  = tr_correct / tr_total
        tr_loss /= tr_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 25 == 0:
            print(f"    [{subject_label}] Ep {epoch:3d}/{STAGE1_EPOCHS} | "
                  f"Tr:{tr_loss:.4f}/{tr_acc:.4f} Vl:{val_acc:.4f}")

        if patience_cnt >= PATIENCE:
            print(f"    [{subject_label}] Early stop @ ep {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────
# STAGE 2: CORAL ADAPTATION
# ─────────────────────────────────────────────
def stage2_coral(model, src_train_loader, src_val_loader,
                 tgt_train_loader, lam, subject_label):
    criterion = nn.CrossEntropyLoss()

    # Freeze classifier, only adapt feature extractor
    for param in model.classifier.parameters():
        param.requires_grad = False
    feat_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = optim.Adam(feat_params, lr=LR * 0.1, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state   = None
    patience_cnt = 0

    tgt_iter = iter(tgt_train_loader)

    for epoch in range(1, STAGE2_EPOCHS + 1):
        model.train()
        ep_total, ep_ce, ep_coral = 0.0, 0.0, 0.0
        n_batches = 0

        for xb_src, yb_src in src_train_loader:
            xb_src, yb_src = xb_src.to(DEVICE), yb_src.to(DEVICE)

            # Get target batch — cycle if exhausted
            try:
                xb_tgt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter   = iter(tgt_train_loader)
                xb_tgt, _ = next(tgt_iter)
            xb_tgt = xb_tgt.to(DEVICE)

            optimizer.zero_grad()

            src_feat, src_out = model.forward_both(xb_src)
            tgt_feat          = model(xb_tgt, return_features=True)

            ce_loss   = criterion(src_out, yb_src)
            c_loss    = coral_loss(src_feat, tgt_feat)

            # Guard against NaN
            if torch.isnan(c_loss) or torch.isinf(c_loss):
                c_loss = torch.tensor(0.0, device=DEVICE)

            total_loss = ce_loss + lam * c_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(feat_params, 1.0)
            optimizer.step()

            ep_total  += total_loss.item()
            ep_ce     += ce_loss.item()
            ep_coral  += c_loss.item()
            n_batches += 1

        val_acc = evaluate(model, src_val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 25 == 0:
            print(f"    [{subject_label}] Ep {epoch:3d}/{STAGE2_EPOCHS} | "
                  f"Total:{ep_total/n_batches:.4f} "
                  f"CE:{ep_ce/n_batches:.4f} "
                  f"CORAL:{ep_coral/n_batches:.6f} | "
                  f"Vl:{val_acc:.4f}")

        if patience_cnt >= PATIENCE:
            print(f"    [{subject_label}] Early stop @ ep {epoch}")
            break

    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────
# LOSO FOR ONE LAMBDA
# ─────────────────────────────────────────────
def run_loso_for_lambda(sessions, lam):
    checkpoint_path = os.path.join(SAVE_DIR, f"coral_lambda_{lam}_checkpoint.json")

    # Load checkpoint if exists
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            completed = json.load(f)
        print(f"  Resuming λ={lam} — completed folds: {list(completed.keys())}")

    results = dict(completed)

    for test_subj in range(1, N_SUBJECTS + 1):
        subj_key = f"S{test_subj}"
        if subj_key in results:
            print(f"  Skipping {subj_key} (already done)")
            continue

        tgt_T_key = f"S{test_subj}T"
        tgt_E_key = f"S{test_subj}E"
        if tgt_T_key not in sessions or tgt_E_key not in sessions:
            print(f"  Skipping {subj_key} — missing session data")
            continue

        print(f"\n  ── Fold {test_subj}: Test={subj_key} ──")

        # Build source pool (all other subjects)
        src_X_all, src_y_all = [], []
        for s in range(1, N_SUBJECTS + 1):
            if s == test_subj:
                continue
            for sess in ['T', 'E']:
                k = f"S{s}{sess}"
                if k in sessions:
                    src_X_all.append(sessions[k][0])
                    src_y_all.append(sessions[k][1])

        src_X = np.concatenate(src_X_all)
        src_y = np.concatenate(src_y_all)

        # 90/10 source train/val split
        n_val    = max(1, int(len(src_y) * 0.1))
        idx      = np.random.permutation(len(src_y))
        val_idx  = idx[:n_val]
        tr_idx   = idx[n_val:]

        src_tr_loader  = make_loader(src_X[tr_idx], src_y[tr_idx], BATCH_SIZE_SRC)
        src_val_loader = make_loader(src_X[val_idx], src_y[val_idx], BATCH_SIZE_SRC, shuffle=False)

        tgt_X_T, tgt_y_T = sessions[tgt_T_key]
        tgt_X_E, tgt_y_E = sessions[tgt_E_key]
        tgt_tr_loader  = make_loader(tgt_X_T, tgt_y_T, BATCH_SIZE_TGT)
        tgt_test_loader = make_loader(tgt_X_E, tgt_y_E, BATCH_SIZE_TGT, shuffle=False)

        print(f"    Src train:{len(tr_idx)} val:{n_val} | "
              f"Tgt T:{len(tgt_y_T)} E:{len(tgt_y_E)}")

        # Build fresh model
        model = EEGNetV3().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    EEGNetV3: {n_params:,} params | feat_dim={model.feat_dim}")

        # Stage 1
        print(f"\n    Stage 1: Source pretraining...")
        model = stage1_pretrain(model, src_tr_loader, src_val_loader, "S1")
        stage1_acc = evaluate(model, tgt_test_loader)
        print(f"    After Stage 1: Test Acc={stage1_acc:.4f}")

        # Stage 2
        print(f"\n    Stage 2: CORAL adaptation (λ={lam})...")
        model = stage2_coral(model, src_tr_loader, src_val_loader,
                             tgt_tr_loader, lam, "S2")
        coral_acc = evaluate(model, tgt_test_loader)

        preds, labels = get_predictions(model, tgt_test_loader)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)

        delta = coral_acc - stage1_acc
        arrow = "▲" if delta >= 0 else "▼"
        print(f"\n    {subj_key}: Stage1={stage1_acc:.4f} → CORAL={coral_acc:.4f} "
              f"{arrow}{abs(delta):.4f}")
        print(classification_report(labels, preds,
              target_names=['Left Hand','Right Hand','Both Feet','Tongue'],
              zero_division=0))

        results[subj_key] = {
            "stage1": round(stage1_acc, 4),
            "coral":  round(coral_acc, 4),
            "delta":  round(delta, 4),
            "f1":     round(f1, 4),
            "n_test": int(len(tgt_y_E))
        }

        # Save checkpoint after each fold
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"    Checkpoint saved ({len(results)} folds done for λ={lam})")

    return results


# ─────────────────────────────────────────────
# PRINT SUMMARY TABLE FOR ONE LAMBDA
# ─────────────────────────────────────────────
def print_lambda_summary(results, lam):
    print(f"\n{'='*60}")
    print(f"  λ={lam} LOSO Summary")
    print(f"{'='*60}")
    print(f"  {'Subj':<8} {'Stage1':>8} {'CORAL':>8} {'Δ':>8} {'F1':>8} {'N':>6}")
    print(f"  {'-'*50}")

    stage1_vals, coral_vals = [], []
    for s in range(1, N_SUBJECTS + 1):
        k = f"S{s}"
        if k not in results:
            continue
        r = results[k]
        arrow = "▲" if r['delta'] >= 0 else "▼"
        print(f"  {k:<8} {r['stage1']:>8.4f} {r['coral']:>8.4f} "
              f"  {arrow}{abs(r['delta']):.4f} {r['f1']:>8.4f} {r['n_test']:>6}")
        stage1_vals.append(r['stage1'])
        coral_vals.append(r['coral'])

    print(f"  {'-'*50}")
    if coral_vals:
        print(f"  {'No-adapt avg':<16}: {np.mean(stage1_vals):.4f}")
        print(f"  {'CORAL avg':<16}: {np.mean(coral_vals):.4f} ± {np.std(coral_vals):.4f}")
        print(f"  {'Improvement':<16}: {np.mean(coral_vals)-np.mean(stage1_vals):+.4f}")


# ─────────────────────────────────────────────
# FINAL CROSS-LAMBDA COMPARISON
# ─────────────────────────────────────────────
def print_final_comparison(all_results):
    print(f"\n{'='*70}")
    print(f"  FINAL LAMBDA SWEEP COMPARISON")
    print(f"{'='*70}")
    print(f"  {'λ':>8} | {'LOSO Avg':>10} | {'Std':>8} | {'vs No-Adapt':>12} | {'Best subj'}")
    print(f"  {'-'*60}")

    best_lam, best_avg = None, 0.0
    for lam, results in all_results.items():
        if not results:
            continue
        coral_vals  = [r['coral']  for r in results.values()]
        stage1_vals = [r['stage1'] for r in results.values()]
        avg   = np.mean(coral_vals)
        std   = np.std(coral_vals)
        delta = avg - np.mean(stage1_vals)
        best_s = max(results, key=lambda k: results[k]['delta'])
        arrow  = "▲" if delta >= 0 else "▼"
        print(f"  {lam:>8} | {avg:>10.4f} | {std:>8.4f} | "
              f"  {arrow}{abs(delta):.4f}      | {best_s} ({results[best_s]['delta']:+.4f})")
        if avg > best_avg:
            best_avg = avg
            best_lam = lam

    print(f"\n  >> Best λ: {best_lam} (avg={best_avg:.4f})")

    print(f"\n  --- vs all versions ---")
    print(f"  v1 EEGNet LOSO     : 0.4780")
    print(f"  v3 EEGNet LOSO     : 0.4277")
    print(f"  v3+CORAL (λ=1)     : 0.4557")
    for lam, results in all_results.items():
        if lam == 1 or not results:
            continue
        coral_vals = [r['coral'] for r in results.values()]
        print(f"  v3+CORAL (λ={lam:<5}): {np.mean(coral_vals):.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("BCI-IV 2a EEGNet v3 + CORAL Lambda Sweep")
    print(f"Device: {DEVICE}\n")

    test_coral_loss()

    sessions = load_all_sessions()
    print(f"\nLoaded {len(sessions)} sessions.\n")

    all_results = {}

    for lam in LAMBDAS:
        print(f"\n{'#'*70}")
        print(f"# RUNNING λ = {lam}")
        print(f"{'#'*70}")

        # Check if this lambda is fully done already
        checkpoint_path = os.path.join(SAVE_DIR, f"coral_lambda_{lam}_checkpoint.json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                existing = json.load(f)
            if len(existing) == N_SUBJECTS:
                print(f"  λ={lam} already fully complete — loading from checkpoint.")
                all_results[lam] = existing
                print_lambda_summary(existing, lam)
                continue

        results = run_loso_for_lambda(sessions, lam)
        all_results[lam] = results
        print_lambda_summary(results, lam)

    # Final comparison across all λ values
    print_final_comparison(all_results)

    # Save full results
    summary_path = os.path.join(SAVE_DIR, "lambda_sweep_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nFull summary saved to: {summary_path}")


if __name__ == "__main__":
    main()