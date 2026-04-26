---
title: Predictive Maintenance Dashboard
emoji: 🔧
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
---

# 🔧 Predictive Maintenance Dashboard

A real-time **machine failure prediction system** powered by Decision Tree models, built with Gradio and deployable to Hugging Face Spaces. Feed in industrial sensor readings and instantly know whether a failure is imminent — and what kind.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Dual Prediction Modes** | Manual single-input prediction *and* batch CSV upload with downloadable results |
| **Smart Model Routing** | Choose between *"Minimize missed failures"* (cost-sensitive) or *"Minimize unnecessary maintenance"* (balanced) to match your business priority |
| **Failure Diagnostics** | Get the *primary root cause* (multi-class) or *all contributing causes* (multi-label) for every predicted failure |
| **5 Failure Types Detected** | Tool Wear (TWF), Heat Dissipation (HDF), Power (PWF), Overstrain (OSF), Random (RNF) |
| **Production-Ready Architecture** | Clean 3-layer separation: `pipeline.py` → `router.py` → `app.py` |
| **Zero Data Leakage** | Scaler fitted on training data only, SMOTE applied after scaling, test set never touched during preprocessing |

---

## 🚀 Quick Start

### Run Locally

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/predictive-maintenance
cd predictive-maintenance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
python app.py
```

The dashboard opens at **http://127.0.0.1:7860**

### Deploy to Hugging Face Spaces

1. Create a new Space → choose **Gradio** SDK
2. Push this repo to the Space
3. Done — HF auto-installs `requirements.txt` and runs `app.py`

---

## 📊 Model Performance

### Binary Classification (Failure vs No-Failure)

| Metric | Baseline Model | Cost-Sensitive Model |
|--------|:-:|:-:|
| Accuracy | 95.50% | 88.50% |
| Precision | 40.98% | 20.44% |
| Recall | **73.53%** | **82.35%** |
| F1-Score | 52.63% | 32.75% |
| ROC-AUC | 89.37% | 89.06% |

> **Baseline** = fewer false alarms · **Cost-Sensitive** = catches more real failures

### Multi-Class (Failure Type Identification)

| Metric | Score |
|--------|:-:|
| Overall Accuracy | **98.50%** |
| F1 (weighted) | 98.18% |
| F1 (macro) | 58.99% |

### Multi-Label (Concurrent Failure Detection)

| Metric | Score |
|--------|:-:|
| Hamming Loss | **0.0048** |
| Subset Accuracy | **97.70%** |

---

## 🏗️ Architecture

```
User Input / CSV
       │
       ▼
  ┌─────────────┐
  │  pipeline.py │  ← Preprocessing: encode → align → scale
  └──────┬──────┘
         │  scaled numpy array
         ▼
  ┌─────────────┐
  │  router.py   │  ← Model routing: binary → diagnostic
  └──────┬──────┘
         │  prediction results
         ▼
  ┌─────────────┐
  │   app.py     │  ← Gradio UI: manual + batch tabs
  └─────────────┘
```

---

## 📁 Project Structure

```
├── app.py                    # Gradio UI (entry point)
├── pipeline.py               # Preprocessing logic
├── router.py                 # Model routing & prediction
├── requirements.txt          # Python dependencies
├── Predictive_M.csv          # Training dataset (10,000 samples)
├── save_artifacts.py         # Re-train & export all artifacts
├── DETAILS.md                # In-depth technical documentation
└── Trained_models/
    ├── standard_scaler.joblib
    ├── feature_columns.json
    ├── binary_decision_tree_baseline_smote_*.joblib
    ├── binary_decision_tree_cost_sensitive_smote_*.joblib
    ├── multiclass_decision_tree_*.joblib
    └── multilabel_decision_tree_*.joblib
```

---

## 🔑 Top Predictive Features

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | Rotational Speed (RPM) | 35.23% |
| 2 | Torque (Nm) | 29.97% |
| 3 | Tool Wear (min) | 23.46% |
| 4 | Air Temperature (K) | 6.28% |
| 5 | Process Temperature (K) | 2.98% |

---

## 📋 Dataset

- **Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Samples:** 10,000 industrial machine readings
- **Failure Rate:** 3.39% (339 failures) — heavily imbalanced
- **Features:** 5 sensor readings + 1 categorical (machine type)
- **Targets:** Binary failure flag + 5 individual failure-type flags

---

## ⚙️ Technical Notes

- **Class Imbalance:** Handled with SMOTE on the training set (8,000 → 15,458 samples)
- **Preprocessing Order:** Split → Scale → SMOTE (no data leakage)
- **Algorithms:** Scikit-learn Decision Trees (interpretable, fast, no GPU needed)
- **Scaler:** StandardScaler fitted on X_train only, saved for inference
- See **[DETAILS.md](DETAILS.md)** for the full technical deep-dive

---

## 📄 License

MIT License — free to use, modify, and distribute.
