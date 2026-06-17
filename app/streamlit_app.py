"""Streamlit web interface for plant disease detection."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import Predictor
from src.preprocess import ImageValidationError, load_image, validate_upload
from src.utils import DEFAULT_WEIGHTS_PATH, MAX_UPLOAD_SIZE_MB

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

CONFIDENCE_BADGE = {"high": "🟢", "medium": "🟡", "low": "🔴"}


@st.cache_resource
def get_predictor():
    return Predictor(weights_path=DEFAULT_WEIGHTS_PATH)


def render_result(result: dict):
    top = result["top_prediction"]
    info = top["info"]
    level = top["confidence_level"]

    st.subheader(f"{CONFIDENCE_BADGE[level]} {top['label']}")

    confidence_text = f"Confidence: {top['confidence'] * 100:.1f}%"
    if level == "high":
        st.success(confidence_text)
    elif level == "medium":
        st.warning(confidence_text)
    else:
        st.error(confidence_text)

    chart_df = pd.DataFrame(
        {"Confidence (%)": [p["confidence"] * 100 for p in result["predictions"]]},
        index=[p["label"] for p in result["predictions"]],
    )
    st.bar_chart(chart_df)

    if info:
        if not info.get("healthy", False):
            with st.expander("ℹ️ More info: description, symptoms, treatment, prevention", expanded=True):
                st.markdown(f"**Description:** {info.get('description', 'N/A')}")
                st.markdown(f"**Symptoms:** {info.get('symptoms', 'N/A')}")
                st.markdown(f"**Treatment:** {info.get('treatment', 'N/A')}")
                st.markdown(f"**Prevention:** {info.get('prevention', 'N/A')}")
        else:
            st.success("This leaf appears healthy!")

    st.caption(f"Inference time: {result['inference_seconds'] * 1000:.0f} ms")


def single_image_tab():
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info(f"Upload an image to get started. Supported: JPG, PNG, JPEG | Max size: {MAX_UPLOAD_SIZE_MB}MB")
        return

    try:
        validate_upload(uploaded_file.name, uploaded_file.size)
        image = load_image(uploaded_file)
    except ImageValidationError as e:
        st.error(str(e))
        return

    st.image(image, caption="Uploaded image", use_container_width=True)

    if not st.button("🔍 Analyze", type="primary"):
        return

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    with st.spinner("Analyzing leaf..."):
        result = predictor.predict(image, top_k=3)

    render_result(result)


def batch_tab():
    uploaded_files = st.file_uploader(
        "Choose leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info(f"Upload multiple images to process them as a batch. Max size per file: {MAX_UPLOAD_SIZE_MB}MB")
        return

    if not st.button("🔍 Analyze All", type="primary"):
        return

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    rows = []
    progress = st.progress(0.0)
    status = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status.write(f"Processing {uploaded_file.name} ({i + 1}/{len(uploaded_files)})...")
        try:
            validate_upload(uploaded_file.name, uploaded_file.size)
            image = load_image(uploaded_file)
            result = predictor.predict(image, top_k=3)
            top = result["top_prediction"]
            rows.append({
                "filename": uploaded_file.name,
                "prediction": top["label"],
                "confidence_pct": round(top["confidence"] * 100, 1),
                "confidence_level": top["confidence_level"],
                "status": "ok",
            })
        except ImageValidationError as e:
            rows.append({
                "filename": uploaded_file.name,
                "prediction": None,
                "confidence_pct": None,
                "confidence_level": None,
                "status": str(e),
            })
        progress.progress((i + 1) / len(uploaded_files))

    status.empty()
    progress.empty()

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv_bytes,
        file_name="plant_disease_results.csv",
        mime="text/csv",
    )


def main():
    st.title("🌿 Plant Disease Detector")
    st.write("Upload a photo of a plant leaf to identify possible diseases.")

    tab_single, tab_batch = st.tabs(["Single Image", "Batch Processing"])
    with tab_single:
        single_image_tab()
    with tab_batch:
        batch_tab()


if __name__ == "__main__":
    main()
