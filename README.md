# Privacy-Preserving Cross-Subject Motor Imagery EEG Classification via Federated Learning

**When Simple Averaging Beats Explicit Domain Adaptation**

Abinaya G — Department of Information Technology, Saveetha Engineering College, Chennai, Tamil Nadu, India

---

## Overview

This repository contains the full experimental code for the paper:

> *Privacy-Preserving Cross-Subject Motor Imagery EEG Classification via Federated Learning: When Simple Averaging Beats Explicit Domain Adaptation*  
> Submitted to Expert Systems with Applications (Elsevier)

We investigate federated learning (FL) as a privacy-preserving alternative for cross-subject motor imagery EEG classification. Our key finding is that standard FedAvg with a shared global model outperforms centralised training (46.67% vs 41.00% LOSO accuracy on BCI-IV 2a) while preserving complete data privacy — and that explicit domain adaptation strategies consistently fail to improve upon this.

---

## Key Results

| Method | Privacy | LOSO Acc | Macro F1 |
|--------|---------|----------|----------|
| CSP + LDA | ✗ | 40.93% | 0.361 |
| CSP + SVM | ✗ | 36.54% | 0.319 |
| Centralised EEGNet | ✗ | 41.00% | 0.367 |
| Centralised + CORAL (λ=10) | ✗ | 43.58% | — |
| FedAvg + local heads | ✓ | 43.46% | 0.411 |
| FedAvg + local heads + CORAL | ✓ | 43.16% | 0.410 |
| FedCL fixed stages (40/70/100%) | ✓ | 46.17% | 0.411 |
| FedCL warmup 30→100% | ✓ | 45.40% | 0.409 |
| FedCL warmup 50→100% | ✓ | 42.73% | 0.362 |
| FedRA (Riemannian alignment) | ✓ | 34.20% | 0.221 |
| **FedAvg global (proposed)** | **✓** | **46.67%** | **0.422** |

---

## Repository Structure

```
fl-eeg-repo/
├── src/
│   ├── 01_csp_baseline.py          # CSP + LDA and CSP + SVM baselines
│   ├── 02_eegnet_centralised.py    # Centralised EEGNet baseline
│   ├── 03_coral_lambda_sweep.py    # CORAL λ ∈ {1,10,100,1000} sweep
│   ├── 04_fedavg_global.py         # FedAvg global + local heads + CORAL variants
│   ├── 05_fedcl_fixed_stages.py    # Federated Curriculum Learning (fixed stages)
│   ├── 06_fedcl_warmup.py          # FedCL linear warmup variants
│   ├── 07_physionet_fedavg.py      # PhysioNet dataset — FedAvg vs centralised
│   ├── 08_fedra_riemannian.py      # Federated Riemannian Alignment (FedRA)
│   └── 09_generate_figures.py      # All paper figures (Figures 1–8)
├── figures/                        # Output directory for generated figures
├── requirements.txt
└── README.md
```

---

## Datasets

### BCI Competition IV Dataset 2a (primary)
- 9 subjects, 22 channels, 250 Hz, 4-class motor imagery
- Download: https://www.bbci.de/competition/iv/
- Expected path: `/kaggle/input/datasets/abinayajone/bci-iv-2a-mi/`

### PhysioNet EEG Motor Movement/Imagery (secondary)
- 109 subjects, 64 channels, 160 Hz, 2-class motor imagery
- Download: https://physionet.org/content/eegmmidb/1.0.0/
- Expected path: `/kaggle/input/datasets/gamalasran/physionet-eeg-motor-movement-imagery/files/`

---

## Evaluation Protocol

All experiments use a **strictly unsupervised leave-one-subject-out (LOSO)** protocol:
- Source subjects: T session only (training)
- Target subject: E session only (test)
- Zero target subject labels used at any stage

This is stricter than many published baselines which use both T and E sessions of source subjects.

---

## Preprocessing Pipeline

```
Raw EEG
  → Bandpass filter (4–40 Hz, 4th-order Butterworth)   [EEGNet methods]
  → Bandpass filter (8–30 Hz, 4th-order Butterworth)   [CSP methods + PhysioNet]
  → EOG rejection (threshold 100 µV, post-bandpass)
  → Epoch extraction (2.5–6.0 s, 875 samples at 250 Hz)
  → EMS normalisation (decay=0.999) per epoch
  → EEGNet input: (B, 1, 22, 875)
```

---

## Model

**EEGNet** (Lawhern et al. 2018): F1=8, D=2, F2=16, kernel_length=32, dropout=0.5  
Total parameters: 6,516

**FL hyperparameters:** 100 rounds, 5 local epochs, batch=32, LR=1e-3, Adam optimiser, patience=25

---

## Running the Code

All scripts are designed to run on **Kaggle GPU** (P100 or T4). They can also run on local CPU, though significantly slower.

Run scripts in order:

```bash
# 1. Classical baselines
python src/01_csp_baseline.py

# 2. Centralised EEGNet
python src/02_eegnet_centralised.py

# 3. CORAL lambda sweep
python src/03_coral_lambda_sweep.py

# 4. FedAvg variants
python src/04_fedavg_global.py

# 5. Federated Curriculum Learning
python src/05_fedcl_fixed_stages.py
python src/06_fedcl_warmup.py

# 6. PhysioNet experiment
python src/07_physionet_fedavg.py

# 7. FedRA
python src/08_fedra_riemannian.py

# 8. Generate all figures
python src/09_generate_figures.py
```

Update the `DATA_DIR` variable at the top of each script to match your local dataset path.

---

## Requirements

See `requirements.txt`. Key dependencies:

- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn
- MNE (for PhysioNet EDF loading)
- pyriemann (for FedRA Riemannian alignment)
- matplotlib (for figure generation)

---

## Three Core Contributions

**1. FedAvg outperforms centralised training**  
46.67% LOSO vs 41.00% centralised EEGNet, with full privacy preservation. Mechanism: implicit cross-subject distribution alignment through iterative parameter aggregation.

**2. Systematic negative result on explicit domain adaptation**  
CORAL, curriculum learning (3 variants), and Riemannian alignment all fail to improve upon plain FedAvg in the federated setting. Each has a distinct, documented failure mechanism.

**3. Data sufficiency boundary**  
FL outperforms centralised at ~231 trials/client (BCI-IV 2a) but fails at ~30 trials/client (PhysioNet). Practical threshold: ~200 trials/client.

---

## Citation

If you use this code, please cite:

```bibtex
@article{abinaya2025fedavgeeg,
  title={Privacy-Preserving Cross-Subject Motor Imagery {EEG} Classification 
         via Federated Learning: When Simple Averaging Beats Explicit Domain Adaptation},
  author={Abinaya, G},
  journal={Expert Systems with Applications},
  year={2025},
  note={Under review}
}
```

---

## License

MIT License. See LICENSE file.

---

## Acknowledgements

The BCI Competition IV Dataset 2a was provided by the Institute for Knowledge Discovery, Graz University of Technology, Austria. The PhysioNet EEG Motor Movement/Imagery Dataset was obtained from PhysioNet (Goldberger et al. 2000).
