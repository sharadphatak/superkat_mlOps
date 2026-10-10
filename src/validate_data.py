"""Register the CSV: required schema, semantic validity and transparent summary."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import EXPECTED, TARGET, canonical_features, fingerprint, write_json


def validate(path, output="reports/data_registration.json"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Assignment CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(EXPECTED) - set(frame.columns))
    if missing:
        raise ValueError(f"Expected columns absent: {missing}; found: {frame.columns.tolist()}")
    if len(frame) < 20:
        raise ValueError("At least 20 rows are required for the train/test and CV protocol")
    canonical_features(frame)
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    if (frame[TARGET].notna() & target.isna()).any():
        raise ValueError("Non-numeric target values")
    if np.isinf(target.to_numpy(dtype=float)).any() or (target.dropna() < 0).any():
        raise ValueError("Target must be finite and nonnegative when provided")
    if target.notna().sum() < 20:
        raise ValueError("At least 20 labeled rows required")
    result = {
        "source_file": str(path), "sha256": fingerprint(path), "rows": len(frame),
        "columns": frame.columns.tolist(), "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_values": frame.isna().sum().astype(int).to_dict(),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "extra_columns": sorted(set(frame.columns) - set(EXPECTED)),
        "product_id_format_mismatches": int((~frame.Product_Id.fillna("").astype(str)
            .str.fullmatch(r"[A-Za-z]{2}\d+")).sum()),
        "target_summary": target.describe().to_dict(),
        "temporal_forecast_supported": False,
    }
    write_json(output, result)
    print(pd.DataFrame({"dtype": frame.dtypes, "missing": frame.isna().sum()}))
    print(f"Registered {len(frame)} rows; SHA256={result['sha256']}")
    print(f"Extra columns: {result['extra_columns']}; duplicate rows: {result['exact_duplicate_rows']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/SupeKart.csv")
    parser.add_argument("--output", default="reports/data_registration.json")
    args = parser.parse_args()
    validate(args.data, args.output)
