# Skin Lesion Metadata vs. Multimodal Deep Learning

**Clinical Metadata versus Multimodal Deep Learning for Skin Lesion Classification: A Comparative and Interpretable Machine Learning Study**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-PAD--UFES--20-informational)](https://doi.org/10.17632/zr7vgbcyr2.1)
[![Status](https://img.shields.io/badge/Status-Research%20Project-lightgrey)](#citation)

This repository contains the analysis code and notebooks for a study comparing two approaches to automated skin lesion classification on the [PAD-UFES-20](https://doi.org/10.17632/zr7vgbcyr2.1) dataset:

1. **Metadata-only machine learning** — twelve classical machine learning and deep learning models trained exclusively on structured clinical metadata (age, lesion morphology, patient history, environmental factors, etc.), requiring no imaging hardware.
2. **Hybrid multimodal deep learning** — a reproduced architecture fusing dermoscopic images (MobileNetV2) with clinical metadata (Random Forest) via late fusion, used as a benchmark for (1).

The central question: **how much diagnostic value is already present in the clinical metadata alone, before a single image is analysed?**

---

## Table of Contents

- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Running on Google Colab](#running-on-google-colab-recommended)
  - [Running Locally](#running-locally)
- [Notebooks](#notebooks)
- [Methodology at a Glance](#methodology-at-a-glance)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

---

## Key Results

<div align="center">

| Approach | Model | Test Accuracy | Macro F1 | Macro AUC |
|:---:|:---:|:---:|:---:|:---:|
| Metadata-only | **Random Forest** (best) | **84.64%** (95% CI: 80.3–88.7%) | **0.78** | **0.9537** |
| Metadata-only | Voting Classifier | 84.06% | N/R | N/R |
| Hybrid multimodal (reproduced) | MobileNetV2 + Random Forest, late fusion | ~84.78% | 0.70 | N/R |
| Hybrid multimodal (original study) | Same architecture, original authors | 95.65%¹ | N/R | N/R |

</div>

*N/R = not separately reported for this model. ¹ As reported by the original architecture's authors; not independently reproduced in this repository.*

A Random Forest trained on **20 engineered clinical features alone** performs on par with a substantially more expensive image-based multimodal pipeline, while remaining faster to train, cheaper to deploy, and directly interpretable. Statistical significance testing (Friedman, Nemenyi, Wilcoxon, McNemar), bootstrap confidence intervals, probability calibration analysis, and cross-validated feature importance (SHAP, permutation importance) were used throughout to validate these findings rather than relying on single point estimates.

Recall for the rarer, more dangerous classes (squamous cell carcinoma, melanoma) is markedly lower than for the common classes, and is addressed by a dedicated class-imbalance-handling experiment (SMOTE, cost-sensitive weighting, Balanced Random Forest) within the metadata-only notebook.

---

## Repository Structure

```
.
├── Data/
│   └── metadata.csv                              # PAD-UFES-20 clinical metadata (2,298 records)
├── Notebooks/
│   ├── 01_Clinical_Metadata_Machine_Learning.ipynb  # Metadata-only pipeline: 20 sections, 12 models,
│   │                                                 # statistical testing, SHAP/permutation importance,
│   │                                                 # calibration, error analysis, class-imbalance handling
│   └── 02_Multimodal_Deep_Learning.ipynb            # Hybrid multimodal pipeline: MobileNetV2 image encoder
│                                                     # + Random Forest metadata encoder, late-fusion classifier
├── requirements.txt                               # Python dependencies
├── LICENSE                                        # MIT license (code only — see Dataset section for data license)
└── README.md                                      # This file
```

> **Note:** both notebooks were developed and run on Google Colab and mount Google Drive for data access (`google.colab.drive.mount`). See [Running Locally](#running-locally) for the changes needed to run them outside Colab.

---

## Dataset

This study uses **[PAD-UFES-20](https://doi.org/10.17632/zr7vgbcyr2.1)**, a publicly available skin lesion dataset compiled by the Dermatological and Surgical Assistance Program (PAD) at the Federal University of Espírito Santo (UFES), Brazil.

<div align="center">

| Property | Value |
|:---:|:---:|
| Total cases | 2,298 |
| Diagnostic classes | 6 — BCC, ACK, NEV, SEK, SCC, MEL |
| Clinical metadata fields | 21 (raw), 20 (after feature engineering) |
| Paired image + metadata cases | 2,232 |
| Class balance | Imbalanced (BCC 36.8% → MEL 2.3%) |
| Missing data | ~35% of values across a subset of features |
| License | Openly available for research use via Mendeley Data |

</div>

**Data included in this repository:**

- `Data/metadata.csv` — the clinical metadata file — is included and ready to use with the metadata-only notebook.
- The dermoscopic image archives are **not included** (they are large binary files) and are required only for the hybrid multimodal notebook.

**Download the images (required only for the hybrid multimodal notebook):**

- Official source (Mendeley Data): **https://doi.org/10.17632/zr7vgbcyr2.1**
- The images are distributed as three zipped archives (`imgs_part_1.zip`, `imgs_part_2.zip`, `imgs_part_3.zip`).

**Please cite the dataset separately from this repository** when using it (see [Citation](#citation) below).

---

## Getting Started

### Requirements

- Python 3.9 or later
- ~4 GB free disk space if downloading the full image archive (the metadata-only notebook does not require the images)
- A Google account, if running on Colab (recommended — see below)

### Installation

```bash
git clone https://github.com/TalhaK-20/Skin-Lesion-Metadata-VS-Multimodal-DL.git
cd Skin-Lesion-Metadata-VS-Multimodal-DL

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Running on Google Colab (recommended)

Both notebooks were built for Colab and are easiest to run there unmodified:

1. Upload `Notebooks/01_Clinical_Metadata_Machine_Learning.ipynb` and/or `Notebooks/02_Multimodal_Deep_Learning.ipynb` to [Google Colab](https://colab.research.google.com/), or open them directly from this GitHub repository via **File → Open notebook → GitHub**.
2. Upload `metadata.csv` (and the image archives, for the hybrid notebook) to your Google Drive.
3. Update the hardcoded data paths near the top of each notebook (search for `metadata_path =` and `drive.mount`) to point to your own Drive location.
4. Run all cells in order — each notebook installs its own additional dependencies (e.g. `shap`, `scikit-posthocs`, `imbalanced-learn`) in its first few cells.

### Running Locally

To run outside Colab:

1. Remove or comment out the `from google.colab import drive` / `drive.mount(...)` cell at the top of each notebook.
2. Update `metadata_path` to point to the local `Data/metadata.csv` file already included in this repository.
3. For the hybrid multimodal notebook, also download and unzip the image archives from the [dataset link above](#dataset), and update the corresponding image-directory paths.
4. Launch Jupyter and run the notebooks:

```bash
jupyter notebook
```

---

## Notebooks

### `01_Clinical_Metadata_Machine_Learning.ipynb` — Metadata-Only Pipeline

A 20-section, self-contained pipeline covering:

1. Data loading, exploration, and preprocessing (missing-value handling, ternary encoding, label encoding, scaling)
2. Training and hyperparameter tuning of 12 models: Random Forest, XGBoost, Gradient Boosting, SVM, Logistic Regression, KNN, AdaBoost, Extra Trees, LightGBM, MLP, a soft-voting ensemble, and a Keras/TensorFlow deep neural network
3. Evaluation: accuracy, confusion matrices, classification reports, ROC/AUC, 5-fold cross-validation, feature importance
4. **Advanced evaluation (Sections 13–20):** statistical significance testing (Friedman, Nemenyi, Wilcoxon, McNemar), SHAP and permutation importance, probability calibration, bootstrap confidence intervals, feature-importance bias validation, computational performance benchmarking, misclassification error analysis, and class-imbalance-handling experiments (SMOTE, cost-sensitive weighting, Balanced Random Forest)

### `02_Multimodal_Deep_Learning.ipynb` — Hybrid Multimodal Pipeline

Reproduces a late-fusion multimodal architecture:

- Downloads and preprocesses PAD-UFES-20 dermoscopic images
- Extracts image features via a MobileNetV2 encoder
- Trains a Random Forest on the paired clinical metadata
- Fuses both modalities' class-probability outputs through a dense fusion network for final six-class classification

---

## Methodology at a Glance

- **Data split:** stratified 70/15/15 train/validation/test (metadata-only); a paired 2,232-case subset for the hybrid model
- **Cross-validation:** 5-fold stratified `KFold`, fixed random seed, shared across all models for valid paired statistical comparisons
- **Hyperparameter tuning:** grid search via `GridSearchCV` on the pooled training/validation partitions only, never touching the held-out test set
- **Evaluation metrics:** accuracy, precision, recall, F1 (macro and weighted), ROC-AUC, Brier score
- **Statistical validation:** Friedman/Nemenyi omnibus and post-hoc tests, Wilcoxon signed-rank, McNemar's test, bootstrap and cross-validation confidence intervals
- **Interpretability:** mean-decrease-in-impurity, SHAP, and permutation importance, cross-validated against one another to check for cardinality bias

Full methodological detail, equations, and result tables are provided in the associated manuscript (see [Citation](#citation)).

---

## Reproducibility

- Global random seed: `101` (set for NumPy, Python's `random` module, and `PYTHONHASHSEED`)
- All models within a comparison use the same cross-validation fold assignments
- Core dependencies are listed in [`requirements.txt`](requirements.txt); pin exact versions locally (e.g. via `pip freeze`) if bit-for-bit replication is required

---

## Citation

If you use this code, please cite the associated paper and the dataset separately.

**This repository / paper:**

```bibtex
@misc{khalid2026skinmetadata,
  author       = {Khalid, Talha and Ahmad, Ashfaq and Javed, Iqra},
  title        = {Clinical Metadata versus Multimodal Deep Learning for Skin Lesion
                  Classification: A Comparative and Interpretable Machine Learning Study},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/TalhaK-20/Skin-Lesion-Metadata-VS-Multimodal-DL}
}
```

**Dataset (please cite alongside this repository when using PAD-UFES-20):**

```bibtex
@misc{pacheco2020padufes20,
  author       = {Pacheco, Andr\'e G. C. and others},
  title        = {{PAD-UFES-20}: A skin lesion dataset composed of patient data
                  and clinical images collected from smartphones},
  howpublished = {Mendeley Data},
  year         = {2020},
  doi          = {10.17632/zr7vgbcyr2.1}
}
```

---

## License

The **code** in this repository is released under the [MIT License](LICENSE).

The **PAD-UFES-20 dataset** is distributed separately by its original authors under its own terms — see the [official dataset page](https://doi.org/10.17632/zr7vgbcyr2.1) for its license before use.

---

## Authors

<div align="center">

| Name | Affiliation | Email |
|:---:|:---:|:---:|
| **Talha Khalid** (corresponding author) | Department of Artificial Intelligence, University of Management and Technology, Lahore, Pakistan | tk839587@gmail.com |
| Dr. Ashfaq Ahmad | Department of Artificial Intelligence, University of Management and Technology, Lahore, Pakistan | ashfaqahmad@umt.edu.pk |
| Dr. Iqra Javed | Department of Computer Science, University of Management and Technology, Lahore, Pakistan | iqra.javed@umt.edu.pk |

</div>

## Acknowledgements

The authors thank their colleagues in the Department of Artificial Intelligence and the Department of Computer Science, University of Management and Technology, Lahore, for their guidance and feedback during the preparation of this work.