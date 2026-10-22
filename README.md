# 🫁 Federated Learning for Pneumonia Detection

> **Capstone Project** — Privacy-preserving pneumonia classification from chest X-rays using the **SCAFFOLD federated learning algorithm** with a **DenseNet121** backbone.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Reproduce](#-how-to-reproduce)
- [Streamlit Demo App](#-streamlit-demo-app)
- [Federated Learning Design](#-federated-learning-design)
- [File Descriptions](#-file-descriptions)
- [Publications & Documents](#-publications--documents)
- [Limitations & Future Work](#-limitations--future-work)

---

## 🔬 Project Overview

This project implements a **federated learning** system for detecting pneumonia from chest X-rays — without ever centralizing patient data. Multiple simulated hospital "clients" each train locally on their private dataset, sharing only model weight updates with a central aggregation server.

**Why federated?**  
Hospitals cannot legally share patient X-rays due to HIPAA / GDPR regulations. Federated learning allows collaborative model training while keeping images local.

**Key innovations:**
- **SCAFFOLD algorithm** (Stochastic Controlled Averaging for Federated Learning) reduces *client drift* — a common failure mode in federated training where clients diverge due to heterogeneous data distributions.
- **Multi-cohort support:** adult (RSNA) + pediatric (Kermany) clients trained together.
- **Threshold tuning:** decision threshold swept on a held-out validation set to optimize Fβ (precision-weighted) instead of using a fixed 0.5.
- **Test-Time Augmentation (TTA):** 5 augmented views averaged per prediction in the Streamlit demo.
- **Best-round selection:** the round with the highest validation score is saved — not always the last round.

---

## 📊 Key Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 80.79% |
| Precision | 80.77% |
| Recall    | 80.83% |
| F1-score  | 80.80% |

**Training configuration:**
- Federated rounds: 20
- Local epochs per round: 2
- Batch size: 16
- Decision threshold: 0.57
- Backbone: DenseNet121 (ImageNet pre-trained, last 2 dense blocks fine-tuned)

---

## 🧠 Architecture

```
Input X-ray (224×224 grayscale)
        │
        ▼
  DenseNet121 Backbone
  (ImageNet pre-trained)
  ├── denseblock1, transition1   ← FROZEN
  ├── denseblock2, transition2   ← FROZEN
  ├── denseblock3, transition3   ← Fine-tuned
  ├── denseblock4                ← Fine-tuned
  └── norm5                      ← Fine-tuned
        │
        ▼
  AdaptiveAvgPool2d (1×1)
        │
        ▼
  Linear(1024 → 128) + ReLU + Dropout(0.5)
        │
        ▼
  Linear(128 → 1)  →  Sigmoid  →  P(Pneumonia)
        │
        ▼
  Threshold @ 0.57  →  NORMAL / PNEUMONIA
```

**Federated setup:**
```
┌──────────────────────────────────────────────┐
│                 Flower Server                │
│         (SCAFFOLD + FedAvg strategy)         │
│  • Aggregates weight updates                 │
│  • Tracks best validation round              │
│  • Maintains global control variates (c_g)   │
└────────────┬─────────────────┬───────────────┘
             │                 │
    ┌────────▼──────┐ ┌────────▼──────┐
    │   Client 0    │ │   Client 1    │  ... Client N
    │  (Adult data) │ │ (Pediatric)   │
    │  local c_l    │ │  local c_l    │
    └───────────────┘ └───────────────┘
```

---

## 📦 Datasets

This project uses **two publicly available datasets**. You must download them separately — they are **not included** in this repository due to size (several GB each).

### 1. RSNA Pneumonia Detection Challenge (Adult X-rays)

| Field        | Details |
|--------------|---------|
| **Source**   | Kaggle — RSNA Pneumonia Detection Challenge 2018 |
| **Link**     | 🔗 https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data |
| **License**  | Kaggle competition data (non-commercial research) |
| **Format**   | DICOM (`.dcm`) |
| **Size**     | ~9 GB |
| **Labels**   | `stage_2_train_labels.csv` — columns: `patientId`, `Target` (0=Normal, 1=Lung Opacity/Pneumonia) |

**Files needed:**
- `stage_2_train_images/` — folder of `.dcm` files
- `stage_2_train_labels.csv` — label CSV

**Preprocessing (converts DICOM → 224×224 PNG, split into NORMAL/PNEUMONIA):**
```bash
python preprocess_rsna.py \
  --images-dir stage_2_train_images \
  --labels-csv stage_2_train_labels.csv/stage_2_train_labels.csv \
  --output-root processed_dataset \
  --size 224
```

---

### 2. Kermany Chest X-Ray (Pediatric)

| Field        | Details |
|--------------|---------|
| **Source**   | Kaggle — Chest X-Ray Images (Pneumonia) by Paul Mooney |
| **Link**     | 🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| **Original** | Mendeley Data — DOI: https://doi.org/10.17632/rscbjbr9sj.3 |
| **License**  | CC BY 4.0 |
| **Format**   | JPEG (`.jpeg`) |
| **Size**     | ~1.2 GB |
| **Labels**   | Folder-based: `chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}/` |
| **Subjects** | Pediatric patients (ages 1–5) from Guangzhou Women and Children's Medical Center |

**Note:** Kermany's original train/val/test split has **patient-level data leakage** (same patient appears in multiple splits). Our preprocessing re-splits by patient ID.

**Preprocessing (re-splits by patient, outputs to `pediatric_pool/`):**
```bash
python preprocess_pediatric_chestxray.py \
  --source-root chest_xray \
  --output-root pediatric_pool \
  --size 224 \
  --test-fraction 0.2 \
  --val-fraction 0.1 \
  --seed 42
```

---

### Dataset Layout After Preprocessing

```
dataset/
├── train/
│   ├── client_0/          ← Adult RSNA split 1
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── client_1/          ← Adult RSNA split 2
│   └── client_2/          ← Pediatric (merged from pediatric_pool/train/)
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Use `merge_pediatric_client.py` to merge the pediatric pool into an existing adult dataset layout for multi-cohort training.

---

## 📁 Project Structure

```
.
├── train_scaffold_updated.py        # Main FL training script (SCAFFOLD + Flower)
├── streamlit_app.py                 # Interactive Streamlit demo app
├── preprocess_rsna.py               # DICOM → PNG converter for RSNA dataset
├── preprocess_pediatric_chestxray.py# Kermany dataset preprocessing & patient re-split
├── merge_pediatric_client.py        # Merges pediatric pool into dataset layout
├── balanced_threshold_allages.py    # Post-hoc threshold tuning (all-ages model)
├── retune_threshold_allages.py      # Re-tunes threshold on updated validation set
├── eval_three_thresholds.py         # Evaluates model at 3 candidate thresholds
├── generate_capstone_ppt.py         # Auto-generates a PowerPoint presentation of results
├── scaffold_final_metrics.json      # Final evaluation metrics (auto-generated by training)
├── .streamlit/
│   └── config.toml                  # Streamlit theme config
├── assets/
│   └── project_banner.png           # Banner image for app / presentations
├── paper_figures/
│   ├── class_distribution.png       # Dataset class distribution chart
│   └── roc_curves.png               # ROC curve comparison
├── Gantt_Chart_Full_Year.pdf        # Project timeline (full year)
├── Gantt_Chart_Phase3.pdf           # Phase 3 timeline
├── Progress_Review_1.pdf            # Mid-project progress review
├── SCAFFOLD_Algorithm_Explainer.pdf # Technical explainer for SCAFFOLD
└── IEEE_Paper_Final_Submission.tex  # LaTeX source for the IEEE paper
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.10+**
- CUDA-capable GPU recommended (falls back to CPU automatically)
- ~10 GB free disk space (for datasets)

### 1. Clone the Repository

```bash
git clone https://github.com/satwik5544/Capstone-Project.git
cd Capstone-Project
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **No `requirements.txt` yet?** Install these core packages manually:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install flwr==1.8.0
pip install scikit-learn numpy pillow pydicom pandas
pip install streamlit
pip install python-pptx matplotlib  # for generate_capstone_ppt.py
```

> For CPU-only (no GPU):
> ```bash
> pip install torch torchvision
> ```

---

## 🚀 How to Reproduce

### Step 1 — Download datasets

Download both Kaggle datasets (links above) and place them in the project root:

```
Capstone-Project/
├── chest_xray/              ← Kermany dataset
│   ├── train/
│   ├── test/
│   └── val/
└── stage_2_train_images/    ← RSNA DICOM files
    stage_2_train_labels.csv/
```

### Step 2 — Preprocess datasets

```bash
# Convert RSNA DICOMs to PNGs
python preprocess_rsna.py

# Preprocess Kermany pediatric dataset
python preprocess_pediatric_chestxray.py

# Merge pediatric into federated dataset layout (optional, for multi-cohort)
python merge_pediatric_client.py
```

### Step 3 — Train the federated model

```bash
python train_scaffold_updated.py \
  --num-clients 3 \
  --rounds 20 \
  --local-epochs 2 \
  --batch-size 16 \
  --data-root dataset/train \
  --test-root dataset/test \
  --seed 42
```

**All training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-clients` | `3` | Number of federated clients |
| `--rounds` | `20` | Federated communication rounds |
| `--local-epochs` | `2` | Local training epochs per round |
| `--batch-size` | `16` | Batch size |
| `--data-root` | `dataset/train` | Training data root |
| `--test-root` | `dataset/test` | Test set root |
| `--test-root-2` | `None` | Optional second test set (e.g., pediatric) |
| `--seed` | `42` | Random seed for reproducibility |
| `--backbone` | `torchvision` | `torchvision` (ImageNet) or `xrv` (chest X-ray pre-trained) |
| `--class1-weight` | `1.0` | PNEUMONIA class sampling weight (`-1` = auto-balance) |

Outputs:
- `federated_model_scaffold_v2.pth` — saved model checkpoint
- `scaffold_final_metrics.json` — evaluation metrics & threshold
- `scaffold_round_metrics.json` — per-round accuracy/F1 log

### Step 4 — Run Streamlit demo

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`. Upload any chest X-ray JPEG/PNG to get a prediction + a downloadable radiology-style PDF report.

---

## 🖥️ Streamlit Demo App

The demo app (`streamlit_app.py`) provides:

- **Upload interface** — drag & drop any chest X-ray image
- **AI prediction** — NORMAL / PNEUMONIA with confidence bar
- **5-view Test-Time Augmentation** — averages predictions across 5 augmented views for robustness
- **Low-confidence warning** — shown if confidence < 65%
- **Patient form** — collects demographics, symptoms, referring physician
- **Radiology-style PDF report** — downloadable report with AI addendum, electronic signature block, and clinical recommendations

> ⚠️ **Important:** The app requires the trained model checkpoint at `allages_run/federated_model_scaffold_v4_allages.pth`. If you train from scratch, update `MODEL_PATH` in `streamlit_app.py` to point to your checkpoint.

---

## 🔁 Federated Learning Design

### SCAFFOLD Algorithm

Standard FedAvg suffers from **client drift**: clients with heterogeneous data pull the global model in different directions each round. SCAFFOLD corrects this using **control variates**:

- Each client `i` maintains a local control variate `c_i`
- The server maintains a global control variate `c` (average of all clients)
- During local training, gradients are corrected by `(c - c_i)` before each optimizer step
- After local training, `c_i` is updated using the weight displacement (Option II)

This dramatically stabilizes training when data distributions differ across clients.

### Learning Rate Schedule

| Rounds | Learning Rate |
|--------|---------------|
| 1–5    | 3×10⁻⁴        |
| 6–10   | 2×10⁻⁴        |
| 11+    | 1×10⁻⁴        |

### Best Round Selection

The global model from the **best validation round** is saved — not necessarily the last round — since later rounds can overfit. The selection metric is configurable (`accuracy`, `precision`, `recall`, `f1`; default: `accuracy`).

---

## 📄 File Descriptions

| File | Purpose |
|------|---------|
| `train_scaffold_updated.py` | Core FL training — `ScaffoldClient`, `CustomStrategy`, threshold tuning, final evaluation |
| `streamlit_app.py` | Full Streamlit web demo with TTA inference, PDF report generation, patient form |
| `preprocess_rsna.py` | Reads RSNA DICOM files, normalizes pixels, saves 224×224 PNGs by label |
| `preprocess_pediatric_chestxray.py` | Pools Kermany splits, re-splits at patient level to avoid leakage |
| `merge_pediatric_client.py` | Merges pediatric training pool into the federated `dataset/train/client_N/` layout |
| `balanced_threshold_allages.py` | Sweeps decision thresholds on a combined adult+pediatric validation set |
| `retune_threshold_allages.py` | Re-runs threshold sweep after model update |
| `eval_three_thresholds.py` | Evaluates 3 specific threshold values side-by-side |
| `generate_capstone_ppt.py` | Generates a PowerPoint slide deck from results JSON and figures |

---

## 📚 Publications & Documents

| Document | Description |
|----------|-------------|
| `IEEE_Paper_Final_Submission.tex` | LaTeX source for the IEEE-format research paper |
| `SCAFFOLD_Algorithm_Explainer.pdf` | Technical explainer for the SCAFFOLD algorithm |
| `Progress_Review_1.pdf` | Mid-project progress review presentation |
| `Gantt_Chart_Full_Year.pdf` | Full-year project Gantt chart |
| `Gantt_Chart_Phase3.pdf` | Phase 3 detailed Gantt chart |

---

## ⚠️ Limitations & Future Work

### Current Limitations
- **Model accuracy ~80.8%** — falls short of the 90% accuracy goal, likely due to the heterogeneous RSNA dataset (lung opacities ≠ pneumonia exactly)
- **No GPU scaling** — simulation runs all clients in a single process (Flower in-process simulation); true distributed training requires a network setup
- **Binary classification only** — does not differentiate bacterial vs. viral pneumonia

### Future Work
- [ ] Add a `requirements.txt` with pinned dependencies
- [ ] Experiment with CheXpert or NIH ChestX-ray14 for better adult labels
- [ ] Try DenseNet121 pre-trained on chest X-rays (TorchXRayVision `xrv` backbone — already implemented, pass `--backbone xrv`)
- [ ] Differential privacy (DP-SGD) for stronger privacy guarantees
- [ ] Deploy Streamlit app to Hugging Face Spaces or Streamlit Cloud
- [ ] Grad-CAM visualization to highlight regions driving the prediction

---

## 👤 Author

**Satwik Kunachigi**  
Capstone Project — 2026  
GitHub: [@satwik5544](https://github.com/satwik5544)

---

## 📜 License

This project is released for academic and research purposes.  
The datasets are subject to their own licenses — see the Kaggle competition and dataset pages for terms.

---

*Built with [Flower](https://flower.ai/) · [PyTorch](https://pytorch.org/) · [Streamlit](https://streamlit.io/) · [scikit-learn](https://scikit-learn.org/)*
