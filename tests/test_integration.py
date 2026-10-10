"""Synthetic fixtures test integration only; they are never assignment results."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.common import FEATURES, TARGET
from src.prepare_data import prepare
from src.train import train
from src.predict import load_bundle, predict_frame
from src.validate_data import validate


def fixture_data(n=90):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "Product_Id": [f"FD{i:04d}" for i in range(n)],
        "Product_Weight": rng.uniform(3, 20, n),
        "Product_Allocated_Area": rng.uniform(0.01, 0.2, n),
        "Product_MRP": rng.uniform(30, 300, n),
        "Store_Establishment_Year": rng.choice([1998, 2005, 2012], n),
        "Product_Sugar_Content": rng.choice(["low sugar", "regular", "no sugar"], n),
        "Product_Type": rng.choice(["dairy", "bread", "seafood"], n),
        "Store_Id": rng.choice(["OUT01", "OUT02", "OUT03"], n),
        "Store_Size": rng.choice(["low", "medium", "high"], n),
        "Store_Location_City_Type": rng.choice(["Tier 1", "Tier 2"], n),
        "Store_Type": rng.choice(["Food Mart", "Supermarket Type 1"], n),
    })
    frame[TARGET] = 15 * frame.Product_MRP + rng.normal(0, 30, n)
    frame.loc[0, "Product_Weight"] = np.nan
    frame.loc[1, "Store_Size"] = np.nan
    return frame


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_integration")
    source = root / "fixture.csv"
    fixture_data().to_csv(source, index=False)
    prepare(source, root / "splits")
    train(root / "splits", root / "models", root / "reports", quick=True)
    return root


def test_pipeline_reload_unknown_categories_and_stateless_normalization(trained):
    model, meta = load_bundle(trained / "models")
    frame = pd.DataFrame([meta["sample_input"]])
    expected = predict_frame(model, frame)
    frame["Store_Id"] = "never_seen_store"
    frame["Product_Weight"] = np.nan
    assert np.isfinite(predict_frame(model, frame)).all()
    frame = pd.DataFrame([meta["sample_input"]])
    frame["Store_Type"] = " " + frame["Store_Type"].str.upper() + " "
    np.testing.assert_allclose(predict_frame(model, frame), expected)
    assert set(meta["features"]) == set(FEATURES)


def test_model_hash_and_feature_validation(trained, tmp_path):
    shutil.copytree(trained / "models", tmp_path / "models")
    model = tmp_path / "models/model.joblib"
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_bundle(tmp_path / "models")
    good, meta = load_bundle(trained / "models")
    with pytest.raises(ValueError, match="Missing inference"):
        predict_frame(good, pd.DataFrame([meta["sample_input"]]).drop(columns=["Store_Id"]))


def test_app_form_submission(trained, tmp_path):
    # Isolated app fixture; does not put synthetic model into project models/.
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "app.py", tmp_path / "app.py")
    shutil.copytree(trained / "models", tmp_path / "models")
    at = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=30).run()
    assert not at.exception
    at.button[1].click().run()
    assert not at.exception
    assert len(at.metric) == 1
    assert not at.error
    at.button[0].click().run()
    assert not at.exception
    at.button[1].click().run()
    assert len(at.metric) == 1


def test_bad_data_is_rejected(tmp_path):
    frame = fixture_data()
    path = tmp_path / "invalid.csv"
    frame.drop(columns=["Store_Id"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Expected columns absent"):
        validate(path)
    frame.loc[0, "Product_Allocated_Area"] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="ratio"):
        validate(path)


def test_artifact_integrity(trained, tmp_path):
    shutil.copytree(trained / "splits", tmp_path / "splits")
    data = tmp_path / "splits/train.csv"
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        train(tmp_path / "splits", tmp_path / "models", tmp_path / "reports", quick=True)


def test_all_missing_columns_survive_pipeline(tmp_path):
    frame = fixture_data()
    frame["Product_Weight"] = np.nan
    frame["Store_Size"] = np.nan
    source = tmp_path / "fixture.csv"
    frame.to_csv(source, index=False)
    prepare(source, tmp_path / "splits")
    train(tmp_path / "splits", tmp_path / "models", tmp_path / "reports", quick=True)
    model, meta = load_bundle(tmp_path / "models")
    assert np.isfinite(predict_frame(model, pd.DataFrame([meta["sample_input"]]))).all()
