"""Reload the committed-format bundle and smoke-test actual training sample."""
import pandas as pd
from src.predict import load_bundle, predict_frame

model, meta = load_bundle()
print("Trusted model reloaded successfully; sample prediction:",
      predict_frame(model, pd.DataFrame([meta["sample_input"]])).tolist())
