"""Deterministic cleaning and train/test workflow artifacts."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.common import FEATURES, SEED, TARGET, canonical_features, fingerprint, write_json
from src.validate_data import validate


def prepare(data="data/SupeKart.csv", output="data/processed"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate(data, output / "data_registration.json")
    raw = pd.read_csv(data)
    labeled = raw.dropna(subset=[TARGET]).drop_duplicates().copy()
    # Exact duplicated product/store feature rows should not straddle the holdout.
    features = canonical_features(labeled)
    clean = features.copy()
    clean[TARGET] = pd.to_numeric(labeled[TARGET], errors="raise")
    before = len(clean)
    clean = clean.drop_duplicates()
    # Same features with conflicting targets need a business rule, not arbitrary dropping.
    if clean.duplicated(subset=FEATURES, keep=False).any():
        raise ValueError("Identical retained features have conflicting targets. Inspect duplicate observations before splitting.")
    if len(clean) < 20:
        raise ValueError("Too few distinct labeled observations after cleaning")
    train, test = train_test_split(clean, test_size=0.2, random_state=SEED)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    info = {
        "seed": SEED, "test_fraction": 0.2, "source_sha256": fingerprint(data),
        "raw_rows": len(raw), "missing_target_rows_removed": int(raw[TARGET].isna().sum()),
        "duplicates_removed": len(raw.dropna(subset=[TARGET])) - len(labeled) + before - len(clean),
        "unnecessary_columns_removed": sorted(set(raw.columns) - set(FEATURES + [TARGET])),
        "train_rows": len(train), "test_rows": len(test),
        "train_sha256": fingerprint(output / "train.csv"), "test_sha256": fingerprint(output / "test.csv"),
        "scope": "Random holdout: existing product/store distribution; unseen-store performance unproven",
        "normalization": "Strip/lowercase categorical values; explicit common sugar aliases; no learned preprocessing before split",
    }
    write_json(output / "split_metadata.json", info)
    print(pd.Series(info).to_string())
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/SupeKart.csv")
    p.add_argument("--output", default="data/processed")
    a = p.parse_args()
    prepare(a.data, a.output)
