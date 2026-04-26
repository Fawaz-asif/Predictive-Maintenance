# Predictive Maintenance — Technical Details

This document provides an exhaustive technical reference for the Predictive Maintenance project. For a quick overview, see the [README](README.md).

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Deep-Dive](#2-dataset-deep-dive)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Model Descriptions](#4-model-descriptions)
5. [Evaluation Results](#5-evaluation-results)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Application Architecture](#7-application-architecture)
8. [Deployment Guide](#8-deployment-guide)
9. [Edge Cases & Error Handling](#9-edge-cases--error-handling)
10. [Reproducing Results](#10-reproducing-results)
11. [Future Work](#11-future-work)

---

## 1. Problem Statement

Industrial machines fail unpredictably, causing costly downtime. This project builds a predictive maintenance system that:

1. **Detects** whether a machine is about to fail (binary classification)
2. **Diagnoses** the root cause of failure using two complementary approaches:
   - **Multi-class classification** — identifies the single most likely failure type
   - **Multi-label classification** — identifies all concurrent failure causes

The system exposes these models through a Gradio web app with two user-selectable axes:
- **Business Priority**: Choose between catching more failures (cost-sensitive) vs. reducing false alarms (balanced)
- **Diagnostic Detail**: Choose between single root cause vs. all contributing causes

---

## 2. Dataset Deep-Dive

### Source

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — a synthetic dataset reflecting real-world predictive maintenance scenarios.

### Schema

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `UDI` | int | Unique identifier (1–10000) | Dropped at inference |
| `Product ID` | str | Product serial number | Dropped at inference |
| `Type` | cat | Machine quality variant | L (60%), M (30%), H (10%) |
| `Air temperature [K]` | float | Ambient air temperature | 295.3 – 304.5 K |
| `Process temperature [K]` | float | Process/machine temperature | 305.7 – 313.8 K |
| `Rotational speed [rpm]` | int | Spindle rotational speed | 1,168 – 2,886 RPM |
| `Torque [Nm]` | float | Applied torque | 3.8 – 76.6 Nm |
| `Tool wear [min]` | int | Cumulative tool usage time | 0 – 253 min |
| `Machine failure` | binary | Overall failure flag (target) | 0 or 1 |
| `TWF` | binary | Tool Wear Failure | 0 or 1 |
| `HDF` | binary | Heat Dissipation Failure | 0 or 1 |
| `PWF` | binary | Power Failure | 0 or 1 |
| `OSF` | binary | Overstrain Failure | 0 or 1 |
| `RNF` | binary | Random Failure | 0 or 1 |

### Class Distribution

```
Machine Failure:
  No Failure (0):  9,661  (96.61%)
  Failure    (1):    339  ( 3.39%)  ← heavily imbalanced

Failure Types:
  HDF:  115  (1.15%)  — most common
  OSF:   98  (0.98%)
  PWF:   95  (0.95%)
  TWF:   46  (0.46%)
  RNF:   19  (0.19%)  — rarest
```

### Product Type Distribution

```
  L (Low quality):    6,000  (60.0%)
  M (Medium quality): 2,997  (30.0%)
  H (High quality):   1,003  (10.0%)
```

### Key Observations

- **Extreme class imbalance** (96.6% vs 3.4%) — requires SMOTE or cost-sensitive learning
- A single machine can have **multiple concurrent failures** (e.g., PWF + OSF together)
- **Rotational speed and torque are inversely correlated** — a known physics relationship
- Tool wear is a cumulative counter that resets periodically, indicating tool replacement cycles
- No missing values in the dataset

---

## 3. Preprocessing Pipeline

### Pipeline Order (Critical)

```
Raw Data → Train/Test Split → StandardScaler → SMOTE → Model Training
                  ↑                  ↑            ↑
              stratified         fit on        applied to
             (preserves         X_train        scaled
              class ratio)       ONLY          training
                                               data ONLY
```

This order prevents three common data-leakage scenarios:
1. **Scaling before splitting** — test distribution would leak into scaler statistics
2. **SMOTE before scaling** — synthetic samples would corrupt the scaler
3. **SMOTE on test data** — would invalidate evaluation metrics

### Step-by-Step

#### Step 1 — Feature Engineering
```python
# One-hot encode the 'Type' column (L, M, H)
df = pd.get_dummies(df, columns=["Type"], prefix="Type", drop_first=False)
```

This produces 3 binary columns: `Type_H`, `Type_L`, `Type_M`.

#### Step 2 — Feature Selection

8 features are selected for the models:

| # | Feature | Type |
|---|---------|------|
| 1 | Air temperature [K] | Continuous |
| 2 | Process temperature [K] | Continuous |
| 3 | Rotational speed [rpm] | Continuous |
| 4 | Torque [Nm] | Continuous |
| 5 | Tool wear [min] | Continuous |
| 6 | Type_H | Binary |
| 7 | Type_L | Binary |
| 8 | Type_M | Binary |

Excluded columns: `UDI`, `Product ID` (identifiers), `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (targets/leakage).

#### Step 3 — Stratified Train/Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

- Training set: **8,000** samples (3.39% failure rate)
- Test set: **2,000** samples (3.40% failure rate)

#### Step 4 — StandardScaler

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train ONLY
X_test_scaled  = scaler.transform(X_test)        # transform with fitted scaler
```

The fitted scaler is saved as `Trained_models/standard_scaler.joblib` for inference.

#### Step 5 — SMOTE (Binary Models Only)

```python
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
```

- Before SMOTE: 8,000 samples (271 failures / 7,729 no-failures)
- After SMOTE: **15,458** samples (7,729 failures / 7,729 no-failures)

SMOTE is only used for binary classification models. Multi-class and multi-label models are trained on the original (imbalanced) scaled training data.

---

## 4. Model Descriptions

### 4A — Binary Baseline Decision Tree

**File:** `binary_decision_tree_baseline_smote_8features_threshold_0p50.joblib`

```python
DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
    criterion="gini",
    min_samples_split=5,
    min_samples_leaf=2
)
```

- **Trained on:** SMOTE-balanced, scaled training data (15,458 samples)
- **Use case:** General-purpose failure detection with balanced precision/recall
- **Decision threshold:** 0.50 (standard)

### 4B — Binary Cost-Sensitive Decision Tree

**File:** `binary_decision_tree_cost_sensitive_smote_8features_threshold_0p50.joblib`

```python
DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
    criterion="gini",
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight={0: 1, 1: 10}  # ← penalizes missed failures 10x more
)
```

- **Trained on:** SMOTE-balanced, scaled training data (15,458 samples)
- **Use case:** Safety-critical environments where missing a failure is much worse than a false alarm
- **Trade-off:** Higher recall (82.35%) at the expense of lower precision (20.44%)

### 4C — Multi-Class Decision Tree

**File:** `multiclass_decision_tree_priority_encoded_scaled_original_features.joblib`

```python
DecisionTreeClassifier(max_depth=10, random_state=42)
```

- **Trained on:** Original (imbalanced) scaled training data — no SMOTE
- **Target encoding:** Priority-based single-label: `TWF > HDF > PWF > OSF > RNF`

| Class | Label | Training Count |
|:-----:|-------|:-:|
| 0 | No Failure | 7,729 |
| 1 | TWF | 37 |
| 2 | HDF | 87 |
| 3 | PWF | 82 |
| 4 | OSF | 82 |
| 5 | RNF | 15 |

- **Use case:** When you need to identify the single most likely root cause of failure

### 4D — Multi-Label Decision Tree

**File:** `multilabel_decision_tree_multioutput_scaled_original_features.joblib`

```python
MultiOutputClassifier(
    DecisionTreeClassifier(max_depth=10, random_state=42)
)
```

- **Trained on:** Original (imbalanced) scaled training data — no SMOTE
- **Output:** 5-element binary vector `[TWF, HDF, PWF, OSF, RNF]`
- **Use case:** When a machine might have multiple simultaneous failure modes and you want to identify all of them

---

## 5. Evaluation Results

All metrics are evaluated on the **held-out test set** (2,000 samples, never seen during training).

### 5A — Binary Classification

| Metric | Baseline | Cost-Sensitive |
|--------|:--------:|:--------------:|
| **Accuracy** | 95.50% | 88.50% |
| **Precision** | 40.98% | 20.44% |
| **Recall** | 73.53% | 82.35% |
| **F1-Score** | 52.63% | 32.75% |
| **ROC-AUC** | 89.37% | 89.06% |

**Interpretation:**
- The **baseline model** correctly flags 73.5% of actual failures, but ~60% of its failure alerts are false alarms
- The **cost-sensitive model** catches 82.4% of failures, but ~80% of its alerts are false — use this when missing a failure has catastrophic consequences
- Both models have nearly identical ROC-AUC (~89%), meaning their ranking ability is similar; the difference is where they place the decision boundary

### 5B — Multi-Class Classification (6 classes)

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| No Failure | 0.99 | 1.00 | 0.99 | 1,930 |
| TWF | 0.00 | 0.00 | 0.00 | 9 |
| HDF | 0.83 | 0.89 | 0.86 | 28 |
| PWF | 0.85 | 0.85 | 0.85 | 13 |
| OSF | 0.87 | 0.81 | 0.84 | 16 |
| RNF | 0.00 | 0.00 | 0.00 | 4 |

| Aggregate Metric | Score |
|-----------------|:-----:|
| **Overall Accuracy** | 98.50% |
| **F1 (weighted)** | 98.18% |
| **F1 (macro)** | 58.99% |

**Interpretation:**
- The model excels at detecting HDF, PWF, and OSF failures (F1 > 0.84)
- TWF and RNF are effectively undetectable — they have too few training samples (37 and 15 respectively)
- The high weighted F1 is driven by the dominant "No Failure" class; the low macro F1 reveals the per-class struggle

### 5C — Multi-Label Classification

| Metric | Score |
|--------|:-----:|
| **Hamming Loss** | 0.0048 |
| **Subset Accuracy** | 97.70% |

**Interpretation:**
- Hamming Loss of 0.0048 means on average only 0.48% of the 5 labels are wrong per sample
- 97.7% of predictions match the true label vector exactly
- Performance is high because "all zeros" (no failure) is correct for ~96.6% of samples

### 5D — Feature Importance (Baseline Decision Tree)

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | Rotational speed [rpm] | 35.23% |
| 2 | Torque [Nm] | 29.97% |
| 3 | Tool wear [min] | 23.46% |
| 4 | Air temperature [K] | 6.28% |
| 5 | Process temperature [K] | 2.98% |
| 6 | Type_M | 1.36% |
| 7 | Type_H | 0.50% |
| 8 | Type_L | 0.23% |

**Insights:**
- The **mechanical trio** (RPM, Torque, Tool Wear) accounts for 88.66% of all splits — these are the dominant failure drivers
- Temperature features contribute modestly (~9.26%)
- Product type has minimal direct predictive power (<2%), though it may interact with other features

---

## 6. Inference Pipeline

### How a Single Prediction Flows

```
User Input: {Air Temp: 300.5, Process Temp: 312.0, RPM: 1282,
             Torque: 60.7, Tool Wear: 216, Type: "L"}
     │
     ▼  pipeline.build_single_input()
┌─────────────────────────────────────────────────┐
│ DataFrame:                                       │
│   Type  Air_temp  Proc_temp  RPM   Torque  Wear  │
│   L     300.5     312.0      1282  60.7    216   │
└──────────────┬──────────────────────────────────┘
               │  pipeline.preprocess()
               ▼
┌─────────────────────────────────────────────────┐
│ 1. Drop leakage cols (UDI, Product ID, targets) │
│ 2. One-hot encode Type → Type_H=0, Type_L=1,   │
│    Type_M=0                                      │
│ 3. Compute Temperature_Difference, Power        │
│    (future-proof, dropped by reindex)            │
│ 4. Reindex to 8-column training order            │
│ 5. StandardScaler.transform()                    │
└──────────────┬──────────────────────────────────┘
               │  scaled numpy array (1, 8)
               ▼
┌─────────────────────────────────────────────────┐
│ router.predict()                                 │
│                                                  │
│ Step 1: Binary model → failure_predicted?        │
│   Priority="Minimize missed failures"            │
│     → use cost-sensitive model                   │
│   Priority="Minimize unnecessary maintenance"    │
│     → use baseline model                         │
│                                                  │
│ Step 2: If failure predicted → diagnose          │
│   Detail="Primary cause only"                    │
│     → multi-class model → single label           │
│   Detail="All contributing causes"               │
│     → multi-label model → binary vector          │
└──────────────┬──────────────────────────────────┘
               │
               ▼
   Result: {failure_predicted: True,
            failure_reason: "Overstrain Failure (OSF)",
            model_used: "DecisionTreeClassifier"}
```

### How Batch Prediction Flows

1. User uploads a CSV with raw sensor columns
2. `pipeline.preprocess()` handles the entire DataFrame at once
3. `router.predict()` returns a list of result dicts
4. Two new columns (`Predicted_Failure`, `Failure_Reason`) are appended
5. The augmented DataFrame is shown as a preview and offered as a downloadable CSV

---

## 7. Application Architecture

### File Responsibilities

| File | Responsibility | Dependencies |
|------|---------------|:------------:|
| `pipeline.py` | Raw data → scaled numpy array | joblib, sklearn scaler |
| `router.py` | Scaled array → prediction results | joblib, 4 model files |
| `app.py` | Gradio UI + wiring | pipeline, router, gradio |

### Design Principles

1. **Separation of concerns** — Each file has a single responsibility. If predictions are wrong, check `pipeline.py`; if routing is wrong, check `router.py`; if the UI breaks, check `app.py`.

2. **Module-level loading** — All models and the scaler are loaded once at import time (`joblib.load()` at module scope), not inside functions. This avoids reloading on every prediction call.

3. **Defensive preprocessing** — `pipeline.preprocess()` handles:
   - Missing dummy columns (single-type batches)
   - Extra columns in uploaded CSVs (dropped by `reindex`)
   - Target/leakage columns in CSVs (explicitly dropped)

4. **Gradio 6 compatibility** — Theme and CSS are passed to `launch()` (not the `Blocks()` constructor). SSR mode is disabled for environments without Node.js.

---

## 8. Deployment Guide

### Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) → choose **Gradio** SDK
2. Clone the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
   ```
3. Copy project files into the cloned repo
4. Push:
   ```bash
   git add .
   git commit -m "initial deploy"
   git push
   ```
5. HF Spaces will auto-install `requirements.txt` and run `app.py`

### Local Development

```bash
pip install -r requirements.txt
python app.py
# → http://127.0.0.1:7860
```

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

### Requirements

```
gradio>=4.0.0
scikit-learn
pandas
numpy
joblib
imbalanced-learn
```

> `imbalanced-learn` is required even at inference time — scikit-learn needs it present to deserialize models that were trained in a SMOTE pipeline.

---

## 9. Edge Cases & Error Handling

| Scenario | What Happens |
|----------|-------------|
| `rpm=0` and `torque=0` | Power feature = 0; scaler handles gracefully (just a low z-score) |
| All-H or all-L machine types | Missing dummy columns auto-filled with 0 by `preprocess()` |
| CSV missing `Type` column | `gr.Warning()` shown to user; prediction aborted |
| CSV has `Machine failure` column | Automatically dropped by `DROP_COLS` in `pipeline.py` |
| CSV has extra columns | Dropped by `reindex()` — only the 8 training features are kept |
| Empty CSV upload | `gr.Warning()` shown; returns `None` |
| Very large CSV | Processed in-memory; limited by available RAM |
| Binary model says "no failure" | Diagnostic models are **skipped** — no unnecessary computation |
| Multi-label returns all zeros | Displayed as "Unspecified Failure" (edge case where binary says fail but no type is identified) |

---

## 10. Reproducing Results

### Re-train All Models from Scratch

```bash
python save_artifacts.py
```

This script:
1. Loads `Predictive_M.csv`
2. One-hot encodes `Type`
3. Splits 80/20 with `random_state=42`
4. Fits and saves `StandardScaler`
5. Applies SMOTE (`random_state=42`)
6. Trains and saves all 4 Decision Tree models
7. Prints evaluation metrics for verification

All random seeds are fixed — results are fully deterministic.

### Expected Output

```
Binary Baseline  — Accuracy: 0.9550, F1: 0.5263
Binary Cost-Sens — Accuracy: 0.8850, F1: 0.3275
Multiclass       — Accuracy: 0.9850
Multilabel       — Subset Accuracy: 0.9770
```

---

## 11. Future Work

1. **Ensemble Methods** — Random Forest or XGBoost would likely outperform single Decision Trees, especially on the rare failure types (TWF, RNF)

2. **Threshold Optimization** — Instead of fixed 0.50 threshold, use precision-recall curve analysis to find the optimal operating point for each business priority

3. **Time-Series Features** — If sequential readings are available, adding rolling averages, rate-of-change, and trend features would significantly improve prediction lead time

4. **SHAP Explanations** — Add per-prediction SHAP waterfall plots to the Gradio app so operators can understand *why* a failure was predicted

5. **Alerting Integration** — Connect to email/Slack/PagerDuty for real-time failure alerts in production environments

6. **Model Monitoring** — Track prediction distribution drift over time to detect when the model needs retraining

---

*Last updated: April 2026*
