"""Streamlit web interface for plant disease detection."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import Predictor
from src.preprocess import ImageValidationError, load_image, validate_upload
from src.utils import DEFAULT_WEIGHTS_PATH, MAX_UPLOAD_SIZE_MB

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* page padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

    /* hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* header */
    .app-header { text-align: center; padding: 1.5rem 0 0.5rem 0; }
    .app-header h1 { font-size: 2.4rem; font-weight: 700; color: #1E4D2B; margin-bottom: 0.2rem; }
    .app-header p  { font-size: 1.05rem; color: #555; margin-top: 0; }

    /* result card */
    .result-card {
        background: #F0F7F4;
        border-left: 5px solid #2D6A4F;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .result-card h2 { margin: 0 0 0.3rem 0; font-size: 1.4rem; color: #1E4D2B; }
    .result-card .conf { font-size: 1rem; color: #444; }

    /* prediction mini-cards */
    .pred-card {
        background: #fff;
        border: 1px solid #d4e8dc;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .pred-card .rank  { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .pred-card .label { font-size: 0.95rem; font-weight: 600; color: #1E4D2B; margin: 0.25rem 0; }
    .pred-card .pct   { font-size: 1.5rem; font-weight: 700; color: #2D6A4F; }

    /* info section */
    .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-top: 0.5rem; }
    .info-block {
        background: #fff;
        border: 1px solid #d4e8dc;
        border-radius: 6px;
        padding: 0.8rem 1rem;
    }
    .info-block .info-label { font-size: 0.75rem; text-transform: uppercase;
                              letter-spacing: 0.06em; color: #888; margin-bottom: 0.3rem; }
    .info-block .info-text  { font-size: 0.9rem; color: #222; line-height: 1.5; }

    /* healthy banner */
    .healthy-banner {
        background: #d4edda; border: 1px solid #a3d9a5;
        border-radius: 8px; padding: 1rem 1.2rem;
        font-size: 1rem; color: #1a4d2e; font-weight: 500;
    }

    /* divider */
    hr { border: none; border-top: 1px solid #e0ede6; margin: 1.2rem 0; }

    /* tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

CONFIDENCE_BADGE = {"high": "🟢", "medium": "🟡", "low": "🔴"}
CONFIDENCE_COLOR = {"high": "#1a6b3c", "medium": "#b5770a", "low": "#c0392b"}


@st.cache_resource
def get_predictor():
    return Predictor(weights_path=DEFAULT_WEIGHTS_PATH)


def render_result(result: dict):
    top = result["top_prediction"]
    info = top["info"]
    level = top["confidence_level"]
    conf_pct = top["confidence"] * 100

    # ── top prediction card ──
    badge = CONFIDENCE_BADGE[level]
    color = CONFIDENCE_COLOR[level]
    st.markdown(f"""
    <div class="result-card">
        <h2>{badge} {top['label']}</h2>
        <div class="conf" style="color:{color}; font-weight:600;">
            {conf_pct:.1f}% confidence &nbsp;·&nbsp; {level.capitalize()} certainty
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── top-3 mini cards ──
    cols = st.columns(3)
    ranks = ["1st", "2nd", "3rd"]
    for col, pred, rank in zip(cols, result["predictions"], ranks):
        with col:
            st.markdown(f"""
            <div class="pred-card">
                <div class="rank">{rank} prediction</div>
                <div class="label">{pred['label']}</div>
                <div class="pct">{pred['confidence']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── disease info ──
    if info:
        if not info.get("healthy", False):
            st.markdown(f"""
            <div class="info-grid">
                <div class="info-block">
                    <div class="info-label">📋 Description</div>
                    <div class="info-text">{info.get('description', 'N/A')}</div>
                </div>
                <div class="info-block">
                    <div class="info-label">🔍 Symptoms</div>
                    <div class="info-text">{info.get('symptoms', 'N/A')}</div>
                </div>
                <div class="info-block">
                    <div class="info-label">💊 Treatment</div>
                    <div class="info-text">{info.get('treatment', 'N/A')}</div>
                </div>
                <div class="info-block">
                    <div class="info-label">🛡️ Prevention</div>
                    <div class="info-text">{info.get('prevention', 'N/A')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="healthy-banner">✅ This leaf appears healthy — no disease detected.</div>',
                        unsafe_allow_html=True)

    st.caption(f"Inference time: {result['inference_seconds'] * 1000:.0f} ms")


def single_image_tab():
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### Upload a leaf image")
        uploaded_file = st.file_uploader(
            "Drag and drop or browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.caption(f"Supported formats: JPG, PNG, JPEG · Max size: {MAX_UPLOAD_SIZE_MB} MB")
            return

        try:
            validate_upload(uploaded_file.name, uploaded_file.size)
            image = load_image(uploaded_file)
        except ImageValidationError as e:
            st.error(str(e))
            return

        st.image(image, use_container_width=True)
        analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

    with col_result:
        if not uploaded_file:
            return

        if not analyze:
            st.markdown("""
            <div style="height:100%; display:flex; align-items:center; justify-content:center;
                        color:#aaa; font-size:1rem; padding-top:4rem; text-align:center;">
                Upload an image and click <strong>Analyze</strong><br>to see the results here.
            </div>
            """, unsafe_allow_html=True)
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
        "Upload multiple leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown("#### Upload multiple leaf images")
        st.caption(f"All files are processed in sequence. Max size per file: {MAX_UPLOAD_SIZE_MB} MB")
        return

    st.markdown(f"**{len(uploaded_files)} image(s) ready.**")
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
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv_bytes,
        file_name="plant_disease_results.csv",
        mime="text/csv",
    )


def main():
    st.markdown("""
    <div class="app-header">
        <h1>🌿 Plant Disease Detector</h1>
        <p>Upload a leaf photograph to identify diseases across 38 categories and 14 plant species.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_single, tab_batch = st.tabs(["🔎 Single Image", "📦 Batch Processing"])
    with tab_single:
        single_image_tab()
    with tab_batch:
        batch_tab()


if __name__ == "__main__":
    main()
