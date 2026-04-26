"""
pipeline.py — Preprocessing Logic
===================================
The *only* file that touches raw input data.
Loads the saved StandardScaler and feature-column order at module level,
then exposes two public helpers:
  • preprocess(df)          → scaled numpy array ready for model.predict()
  • build_single_input(…)   → one-row DataFrame from manual UI fields
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Resolve paths relative to this file (works on HF Spaces and locally)
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE, "Trained_models")

# ---------------------------------------------------------------------------
# Load artifacts once at module level (avoids reloading on every prediction)
# ---------------------------------------------------------------------------
scaler = joblib.load(os.path.join(_MODEL_DIR, "standard_scaler.joblib"))

with open(os.path.join(_MODEL_DIR, "feature_columns.json")) as _f:
    FEATURE_COLUMNS: list[str] = json.load(_f)

# Columns that leak the target or are just IDs — drop if present
DROP_COLS = [
    "UDI", "Product ID",
    "TWF", "HDF", "PWF", "OSF", "RNF",
    "Machine failure",
]

TYPE_CATEGORIES = ["L", "M", "H"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> np.ndarray:
    """
    Accept a *raw* DataFrame (CSV upload or manual-input dict).
    Return a scaled numpy array shaped (n_samples, n_features),
    ready for any of the loaded models.
    """
    df = df.copy()

    # 1. Drop leakage / ID columns if present
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # 2. One-hot encode 'Type' to match training get_dummies behaviour
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"])

    # Ensure all three dummy columns exist (handles single-type batches)
    for cat in TYPE_CATEGORIES:
        col = f"Type_{cat}"
        if col not in df.columns:
            df[col] = 0

    # 3. (Future-proof) Feature engineering — computed then dropped by reindex
    #    if FEATURE_COLUMNS doesn't include them (current 8-feature models)
    if "Process temperature [K]" in df.columns and "Air temperature [K]" in df.columns:
        df["Temperature_Difference"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
        )
    if "Torque [Nm]" in df.columns and "Rotational speed [rpm]" in df.columns:
        df["Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]

    # 4. Align to exact training column order; fill missing with 0, drop extras
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # 5. Scale using the saved StandardScaler
    return scaler.transform(df.values)


def build_single_input(
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
    machine_type: str,
) -> pd.DataFrame:
    """
    Convert manual UI inputs into a one-row DataFrame
    that `preprocess()` can handle.
    """
    return pd.DataFrame(
        [
            {
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ]
    )
