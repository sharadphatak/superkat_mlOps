"""One contract shared by training, validation and inference."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Product_Store_Sales_Total"
NUMERIC = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
CATEGORICAL = ["Product_Sugar_Content", "Product_Type", "Store_Id", "Store_Size",
               "Store_Location_City_Type", "Store_Type"]
FEATURES = NUMERIC + CATEGORICAL
EXPECTED = ["Product_Id"] + FEATURES + [TARGET]
SEED = 42


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {p: importlib.metadata.version(p) for p in
            ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "streamlit"]}


def canonical_features(frame):
    """Stateless normalization only: learned imputation remains in fitted pipeline."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing inference features: {missing}")
    result = frame[FEATURES].copy()
    for col in NUMERIC:
        original = result[col]
        result[col] = pd.to_numeric(original, errors="coerce")
        malformed = original.notna() & result[col].isna()
        if malformed.any():
            raise ValueError(f"Non-numeric values in {col}: {original[malformed].head().tolist()}")
        if np.isinf(result[col].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values in {col}")
    for col in CATEGORICAL:
        result[col] = result[col].map(lambda v: str(v).strip().lower() if pd.notna(v) else np.nan)
        result[col] = result[col].replace("", np.nan)
    # Common low-fat sugar aliases are folded only when actually present.
    result["Product_Sugar_Content"] = result["Product_Sugar_Content"].replace(
        {"low fat": "low sugar", "lf": "low sugar", "low_fat": "low sugar",
         "reg": "regular", "no_sugar": "no sugar"})
    for col in ["Product_Weight", "Product_MRP"]:
        if (result[col].dropna() < 0).any():
            raise ValueError(f"Negative {col} is invalid")
    area = result["Product_Allocated_Area"].dropna()
    if ((area < 0) | (area > 1)).any():
        raise ValueError("Product_Allocated_Area must be a ratio in [0,1]")
    years = result["Store_Establishment_Year"].dropna()
    if ((years < 1800) | (years > pd.Timestamp.today().year) | (years % 1 != 0)).any():
        raise ValueError("Store_Establishment_Year must be a valid integer year")
    return result
