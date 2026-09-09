# Cross-Subject Motor Imagery EEG Decoding under Federated Learning

**A Controlled Benchmark of Euclidean Alignment and Target-Statistic Access**

Abinaya G — Department of Information Technology, Saveetha Engineering College, Chennai, Tamil Nadu, India

---

## Overview

This repository contains the experimental code for:

> *Cross-Subject Motor Imagery EEG Decoding under Federated Learning: A Controlled Benchmark of Euclidean Alignment and Target-Statistic Access*
> Abinaya G. Submitted to *Biomedical Signal Processing and Control* (Elsevier).

We evaluate federated learning (FL) for cross-subject motor imagery
EEG classification under a strictly unsupervised, genuine
leave-one-subject-out (LOSO) protocol on BCI Competition IV-2a, with
paired statistical testing (paired t-test, Wilcoxon signed-rank,
Cohen's d_z, bootstrap confidence intervals, Holm–Bonferroni
correction) across two independently designed backbones (EEGNet,
ShallowConvNet).

**Headline finding:** plain FedAvg does **not** provide a statistically
robust advantage over centralised training (p=0.966) or a matched
local-only ensemble. **Federated Euclidean Alignment (Fed-EA)** — a
per-client whitening transform computed with zero cross-client
communication — achieves a large, statistically robust improvement
over every baseline (50.37% vs. 39.84% for FedAvg, p=0.0009,
d_z=1.71). A controlled ablation then shows this advantage is
attributable primarily to Fed-EA's **transductive** access to the
target subject's own unlabelled data, not to communication-freedom
alone: a fully inductive variant drops to 38.02%, statistically
indistinguishable from FedAvg. A follow-up calibration-size sweep
shows just 25% of the target subject's unlabelled data is enough to
recover Fed-EA's full advantage, with no further benefit from more.

## Key Results (EEGNet backbone, genuine LOSO, verified)

| Method | Raw data local? | Seeds | LOSO Acc (%) | Macro F1 |
|---|---|---|---|---|
| CSP + LDA | ✗ | 1 | 39.66 | 0.353 |
| CSP + SVM | ✗ | 1 | 40.55 | 0.358 |
| Centralised | ✗ | 1 | 40.26 | 0.367 |
| Centralised + CORAL (λ=10) | ✗ | 1 | 42.23 | 0.399 |
| Local-only ensemble | ✓* | 1 | 36.67 | 0.294 |
| FedAvg + local heads | ✓ | 1 | 37.21 | 0.316 |
| FedAvg + local heads + CORAL | ✓ | 1 | 38.02 | — |
| FedRA (Riemannian alignment) | ✓ | 1 | 40.72 | 0.354 |
| FedCL (fixed 40/70/100% curriculum) | ✓ | 1 | 38.56 | 0.336 |
| FedCL (warmup 30%→100%) | ✓ | 1 | 40.61 | 0.357 |
| FedCL (warmup 50%→100%) | ✓ | 1 | 39.76 | 0.345 |
| FedAvg (no alignment) | ✓ | 2 | 39.84 | 0.340 |
| **Fed-EA (proposed)** | ✓ | 2 | **50.37** | **0.470** |
| Fed-EA, fully inductive (ablation) | ✓ | 2 | 38.02 | — |

*Chance level (4-class): 25.00%. See the manuscript for ShallowConvNet
backbone results, the full significance testing tables, and the
target-calibration-size sweep (25/50/75/100% of target data).*

## Repository Structure

```
fl-eeg-mi-privacy/
├── src/
│   ├── 01_csp_baselines.py                    # CSP+LDA / CSP+SVM, genuine per-subject LOSO
│   ├── 02_fedavg_local_heads_coral.py         # FedAvg+local-heads, with/without CORAL
│   ├── 03_fedcl_fixed_stage.py                # FedCL fixed-stage curriculum
│   ├── 04_fedavg_global_reverify_seed42.py    # FedAvg (no alignment), seed 42
│   ├── 05_fedavg_global_reverify_seed123.py   # FedAvg (no alignment), seed 123
│   ├── 06_fedea_experiment_seed42.py          # Fed-EA, EEGNet, seed 42
│   ├── 07_fedea_experiment_seed123.py         # Fed-EA, EEGNet, seed 123
│   ├── 08_fedea_shallowconvnet_seed42.py      # Fed-EA, ShallowConvNet, seed 42
│   ├── 09_fedra_reverification.py             # FedRA (Riemannian alignment)
│   ├── 10_fedcl_warmup.py                     # FedCL warmup curricula (2 variants)
│   ├── 11_centralised_coral_reverification.py # Centralised + CORAL, λ grid
│   ├── 12_fedea_shallowconvnet_seed123.py     # Fed-EA, ShallowConvNet, seed 123
│   ├── 13_fedea_inductive_seed42.py           # Fed-EA, fully inductive variant, seed 42
│   ├── 14_fedea_inductive_seed123.py          # Fed-EA, fully inductive variant, seed 123
│   ├── 15_fedea_calibration_sweep.py          # Target-calibration-size sweep (25/50/75/100%)
│   ├── 16_convergence_logging.py              # Round-by-round validation accuracy
│   ├── 17_tsne_feature_extraction.py          # Penultimate-layer features, S2 & S3
│   ├── 18_figure_tsne.py                      # t-SNE feature-space figure
│   ├── 19_figure_calibration_sweep.py         # Calibration-sweep figure
│   ├── 20_calibration_sweep_significance.py   # Calibration-sweep significance testing
│   └── 21_final_significance_n9_consistent.py # Exact stats reported in the manuscript (start here to verify)
├── results/            Raw JSON result files
├── LICENSE
├── requirements.txt
└── README.md
```

**To verify the manuscript's statistics directly, start with
`src/21_final_significance_n9_consistent.py`** — it reads the JSON
files in `results/` and reproduces every p-value, effect size, and
Holm-adjusted p-value quoted for the inductive-vs-transductive
ablation and the calibration-sweep significance table.

## Dataset

**BCI Competition IV Dataset 2a** (the only dataset used in the
current manuscript):
- 9 subjects, 22 channels, 250 Hz, 4-class motor imagery
- Download: https://www.bbci.de/competition/iv/
- Expected path: `/kaggle/input/datasets/abinayajone/bci-iv-2a-mi/`

## Evaluation Protocol

All experiments use a **genuine, leakage-audited leave-one-subject-out
(LOSO)** protocol:
- Source subjects: T session, used for training; an inner
  validation split is carved from source-only data for early
  stopping/model selection (no target data in this split).
- Target subject: E session, used for evaluation only. No target
  label is used at any stage. For the transductive Fed-EA condition
  specifically, unlabelled target-session signals are additionally
  used to estimate the target whitening transform at evaluation time
  (see the manuscript's protocol-summary table for the exact
  target-data-access profile of every method).

## Preprocessing Pipeline

```
Raw EEG
  → Bandpass filter (4–40 Hz, 4th-order Butterworth)
  → Euclidean Alignment (EA) whitening, where applicable
  → Exponential moving standardisation (decay=0.999), per epoch
  → EEGNet / ShallowConvNet input: (B, 1, 22, 875)
```

## Models

**EEGNet**: F1=8, D=2, F2=16, kernel_length=32, dropout=0.5
**ShallowConvNet**: as described in Schirrmeister et al. (2017)

**FL hyperparameters:** 100 communication rounds (fixed budget,
validation-based early stopping, patience=20), 5 local epochs,
batch=32, LR=1e-3, Adam optimiser, class-weighted cross-entropy.

## Running the Code

All scripts were run on **Kaggle (NVIDIA T4 GPU)** and include
per-fold checkpointing to JSON with resume logic — an interrupted
session can be resumed by simply rerunning the same script.

```bash
# Baselines
python src/01_csp_baselines.py
python src/02_fedavg_local_heads_coral.py
python src/03_fedcl_fixed_stage.py
python src/10_fedcl_warmup.py
python src/11_centralised_coral_reverification.py
python src/09_fedra_reverification.py

# Main comparison (FedAvg, 2 seeds)
python src/04_fedavg_global_reverify_seed42.py
python src/05_fedavg_global_reverify_seed123.py

# Fed-EA, both backbones, both seeds
python src/06_fedea_experiment_seed42.py
python src/07_fedea_experiment_seed123.py
python src/08_fedea_shallowconvnet_seed42.py
python src/12_fedea_shallowconvnet_seed123.py

# Inductive-vs-transductive ablation, both seeds
python src/13_fedea_inductive_seed42.py
python src/14_fedea_inductive_seed123.py

# Target-calibration-size sweep
python src/15_fedea_calibration_sweep.py

# Statistical verification (reproduces every number in the paper)
python src/21_final_significance_n9_consistent.py

# Supporting figures
python src/16_convergence_logging.py
python src/17_tsne_feature_extraction.py
python src/18_figure_tsne.py
python src/19_figure_calibration_sweep.py
```

Update the `DATA_DIR` variable at the top of each script if running
outside Kaggle.

## A note on statistical convention

Two different multiple-comparison framings appear for the same
underlying inductive-vs-transductive comparison, and this is
intentional, not an inconsistency:
- Reported **standalone** (uncorrected), matching how other single
  targeted ablations are treated (e.g. Fed-EA vs. FedRA): **significant**
  (p=0.0174 paired t-test, p=0.0078 Wilcoxon, d_z=-1.00).
- Reported **within the calibration-sweep's own 10-comparison
  Holm-corrected family**: does not survive that correction (Holm p=0.151).

Both framings are computed explicitly by
`src/21_final_significance_n9_consistent.py`. All paired significance
tests in this project use n=9 (one value per subject; where two random
seeds exist, the seeds are averaged per subject first).

## Requirements

See `requirements.txt`. Key dependencies: Python 3.9+, PyTorch,
NumPy, SciPy, scikit-learn, matplotlib.

## Citation

If you use this code, please cite the current manuscript (update once
publication details are final):

```
@article{abinaya2026fedea,
  title={Cross-Subject Motor Imagery {EEG} Decoding under Federated Learning:
         A Controlled Benchmark of Euclidean Alignment and Target-Statistic Access},
  author={Abinaya, G},
  journal={Biomedical Signal Processing and Control},
  year={2026},
  note={Under review}
}
```

## License

MIT License. See LICENSE file.

## Acknowledgements

The BCI Competition IV Dataset 2a was provided by the Institute for
Knowledge Discovery, Graz University of Technology, Austria.
