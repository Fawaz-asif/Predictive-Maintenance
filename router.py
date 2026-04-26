"""
router.py — Prediction Routing Engine
=======================================
Loads all 4 Decision Tree models at module level.
Exposes a single `predict()` function that:
  1. Runs binary classification (baseline *or* cost-sensitive).
  2. If a failure is predicted, runs the chosen diagnostic model
     (multi-class *or* multi-label) to identify failure reasons.

This file never touches raw data — it receives the preprocessed
numpy array produced by `pipeline.preprocess()`.
"""

import os
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths relative to this file
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE, "Trained_models")

# ---------------------------------------------------------------------------
# Load all models once at module level
# ---------------------------------------------------------------------------
binary_baseline = joblib.load(
    os.path.join(_MODEL_DIR,
                 "binary_decision_tree_baseline_smote_8features_threshold_0p50.joblib")
)
binary_cost = joblib.load(
    os.path.join(_MODEL_DIR,
                 "binary_decision_tree_cost_sensitive_smote_8features_threshold_0p50.joblib")
)
multiclass_model = joblib.load(
    os.path.join(_MODEL_DIR,
                 "multiclass_decision_tree_priority_encoded_scaled_original_features.joblib")
)
multilabel_model = joblib.load(
    os.path.join(_MODEL_DIR,
                 "multilabel_decision_tree_multioutput_scaled_original_features.joblib")
)

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------
MULTICLASS_LABELS = {
    0: "No Failure",
    1: "Tool Wear Failure (TWF)",
    2: "Heat Dissipation Failure (HDF)",
    3: "Power Failure (PWF)",
    4: "Overstrain Failure (OSF)",
    5: "Random Failure (RNF)",
}

MULTILABEL_LABELS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict(
    X: np.ndarray,
    business_priority: str,
    diagnostic_detail: str,
) -> list[dict]:
    """
    Parameters
    ----------
    X : np.ndarray
        Preprocessed feature matrix (n_samples × n_features).
    business_priority : str
        "Minimize missed failures"          → use cost-sensitive binary model
        "Minimize unnecessary maintenance"  → use baseline binary model
    diagnostic_detail : str
        "Primary cause only"       → multi-class model  (single reason)
        "All contributing causes"  → multi-label model   (all reasons)

    Returns
    -------
    list[dict]
        One dict per row with keys:
          failure_predicted  (bool)
          failure_reason     (str)
          model_used         (str)
    """

    # Step 1 — Choose binary model based on business priority
    if business_priority == "Minimize missed failures":
        binary_model = binary_cost
    else:
        binary_model = binary_baseline

    binary_preds = binary_model.predict(X)  # shape (n_samples,)

    results: list[dict] = []
    for i, pred in enumerate(binary_preds):
        row = X[i : i + 1]

        if pred == 0:
            results.append(
                {
                    "failure_predicted": False,
                    "failure_reason": "System Normal",
                    "model_used": type(binary_model).__name__,
                }
            )
            continue  # no failure → skip diagnostic models

        # Step 2 — Identify failure reason(s)
        if diagnostic_detail == "Primary cause only":
            mc_pred = multiclass_model.predict(row)[0]
            reason = MULTICLASS_LABELS.get(int(mc_pred), "Unknown")
        else:
            ml_pred = multilabel_model.predict(row)[0]
            active = [
                MULTILABEL_LABELS[j]
                for j, v in enumerate(ml_pred)
                if v == 1
            ]
            reason = ", ".join(active) if active else "Unspecified Failure"

        results.append(
            {
                "failure_predicted": True,
                "failure_reason": reason,
                "model_used": type(binary_model).__name__,
            }
        )

    return results
