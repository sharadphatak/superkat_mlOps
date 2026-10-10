"""Actual-data integration checks after full training; no deployment claims."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.common import FEATURES, canonical_features
from src.predict import load_bundle, predict_frame
from src.train import metrics

root=Path.cwd()
model, meta=load_bundle()
source=pd.read_csv('data/SupeKart.csv')
pred=predict_frame(model,source)
assert len(pred)==len(source) and np.isfinite(pred).all()
heldout=pd.read_csv('data/processed/test.csv')
actual_metrics=metrics(heldout.Product_Store_Sales_Total,predict_frame(model,heldout))
for name,value in actual_metrics.items():
 assert np.isclose(value,meta['evaluation']['model'][name])
unknown=pd.DataFrame([meta['sample_input']])
unknown['Store_Id']='new_store_not_in_training'
unknown['Product_Weight']=np.nan
assert np.isfinite(predict_frame(model,unknown)).all()
at=AppTest.from_file('app.py',default_timeout=30).run()
assert not at.exception and not at.error
at.button[1].click().run()
assert not at.exception and not at.error and len(at.metric)==1
at.button[0].click().run()
assert not at.exception
at.button[1].click().run()
assert not at.exception and len(at.metric)==1
sample_prediction=float(predict_frame(model,pd.DataFrame([meta['sample_input']]))[0])
result={
 'rows_predicted':len(pred), 'all_predictions_finite':True,
 'prediction_range':[float(pred.min()),float(pred.max())],
 'negative_prediction_count':int((pred<0).sum()),
 'holdout_metrics_reproduced':actual_metrics,
 'unknown_store_and_missing_weight':'passed',
 'actual_model_streamlit_form_and_sample_button':'passed',
 'sample_prediction':sample_prediction,
 'model_sha256':meta['model_sha256'],
 'no_live_deployment_tested':True,
}
Path('reports/actual_data_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
