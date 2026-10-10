"""Repository-backed Streamlit product/store revenue estimator."""
from pathlib import Path
import pandas as pd
import streamlit as st
from src.common import FEATURES, NUMERIC
from src.predict import load_bundle, predict_frame

st.set_page_config(page_title="SuperKart | Revenue estimator", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart")
st.subheader("Product and store revenue estimator")
st.caption("Estimate a product/store row's revenue from retail attributes. Currency and period are unspecified in the source schema.")

@st.cache_resource
def bundle():
    return load_bundle(Path(__file__).parent / "models")

try:
    model, metadata = bundle()
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    st.error(f"Model unavailable: {exc}. Run and publish the training pipeline first.")
    st.stop()

if st.button("Use training sample"):
    for col in FEATURES:
        st.session_state[col] = metadata["sample_input"][col]

with st.form("revenue_inputs"):
    values = {}
    left, right = st.columns(2)
    for i, col in enumerate(FEATURES):
        container = left if i % 2 == 0 else right
        sample = metadata["sample_input"].get(col)
        with container:
            label = col.replace("_", " ")
            if col in NUMERIC:
                if col == "Store_Establishment_Year":
                    values[col] = st.number_input(label, min_value=1800, max_value=pd.Timestamp.today().year,
                        value=int(sample) if sample is not None else 2000, step=1, key=col)
                elif col == "Product_Allocated_Area":
                    values[col] = st.number_input(label, min_value=0.0, max_value=1.0,
                        value=float(sample) if sample is not None else 0.05, step=0.01, format="%.4f", key=col)
                else:
                    values[col] = st.number_input(label, min_value=0.0,
                        value=float(sample) if sample is not None else 0.0, key=col)
            else:
                choices = metadata["categories"].get(col) or ["unknown"]
                selected = choices.index(sample) if sample in choices else 0
                values[col] = st.selectbox(label, choices, index=selected, key=col)
    submitted = st.form_submit_button("Estimate revenue")
if submitted:
    try:
        frame = pd.DataFrame([values], columns=FEATURES)
        prediction = float(predict_frame(model, frame)[0])
        st.metric("Estimated revenue (source units)", f"{prediction:,.2f}")
        if prediction < 0:
            st.warning("The model produced a negative estimate; this case needs review.")
        st.dataframe(frame, hide_index=True)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
with st.expander("Model and validation"):
    st.write("Algorithm:", metadata["algorithm"])
    st.json(metadata["evaluation"])
    st.caption(f"Data SHA256: {metadata['split']['source_sha256'][:16]} · Model: {metadata['model_sha256'][:16]}")
    st.info("Random holdout evaluates the existing product/store distribution. Future periods and unseen stores are not validated.")
