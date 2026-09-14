"""
Animal Classification — Streamlit UI entrypoint.

Run from the project root with:
    streamlit run app/app.py
"""

from __future__ import annotations

import logging
import sys
from io import BytesIO
from typing import List, Tuple

import streamlit as st
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Bootstrap logging before any other app imports
# ---------------------------------------------------------------------------
from config import ALLOWED_EXTENSIONS, CLASS_NAMES, DEVICE, LOG_FORMAT, LOG_LEVEL, MAX_UPLOAD_MB, MODEL_PATH

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, stream=sys.stdout)
logger = logging.getLogger(__name__)

from inference import get_top_k, predict  # noqa: E402
from model import load_model  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Animal Classifier — MobileNetV3",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(160deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* ── Sidebar ────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141432 0%, #1c1c44 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0f0 !important;
    }

    /* ── Hero header ────────────────────────────────────────── */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ── Glass card ─────────────────────────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(167, 139, 250, 0.12);
    }

    /* ── Prediction result ──────────────────────────────────── */
    .prediction-label {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #34d399, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: capitalize;
        margin-bottom: 0.25rem;
    }
    .prediction-confidence {
        font-size: 1.2rem;
        color: #a78bfa;
        font-weight: 600;
    }

    /* ── Bar chart bars ─────────────────────────────────────── */
    .bar-container {
        margin-bottom: 0.65rem;
    }
    .bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-bottom: 3px;
        font-weight: 500;
        text-transform: capitalize;
    }
    .bar-track {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        overflow: hidden;
        height: 22px;
    }
    .bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
    }

    /* ── Device badge ───────────────────────────────────────── */
    .device-badge {
        display: inline-block;
        background: rgba(167, 139, 250, 0.15);
        color: #a78bfa;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* ── Class chips ────────────────────────────────────────── */
    .class-chip {
        display: inline-block;
        background: rgba(96, 165, 250, 0.12);
        color: #93c5fd;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 2px 3px;
        font-weight: 500;
        text-transform: capitalize;
        border: 1px solid rgba(96, 165, 250, 0.2);
    }

    /* ── Upload area ────────────────────────────────────────── */
    .stFileUploader > div {
        border: 2px dashed rgba(167, 139, 250, 0.3) !important;
        border-radius: 12px !important;
        background: rgba(167, 139, 250, 0.03) !important;
    }
    .stFileUploader > div:hover {
        border-color: rgba(167, 139, 250, 0.5) !important;
        background: rgba(167, 139, 250, 0.06) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🐾 About")
    st.markdown(
        "This app uses a **MobileNetV3-Large** deep learning model "
        "fine-tuned to classify images into **23 animal species**."
    )
    st.markdown(f'<span class="device-badge">Device: {DEVICE}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Supported Animals")
    chips_html = "".join(f'<span class="class-chip">{name.replace("_", " ")}</span>' for name in CLASS_NAMES)
    st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "**How to use:**\n"
        "1. Upload a `.jpg`, `.jpeg`, or `.png` image\n"
        "2. The model will classify the animal\n"
        "3. View top-5 predictions with confidence scores"
    )
    st.markdown(
        f"<p style='color:#64748b;font-size:0.75rem;margin-top:1rem;'>"
        f"Max upload size: {MAX_UPLOAD_MB} MB</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="hero-title">Animal Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    "Upload an image and let AI identify the animal species with MobileNetV3"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
try:
    model = load_model()
except Exception as exc:
    logger.exception("Failed to load model.")
    st.error(
        f"⚠️ **Model failed to load.**\n\n"
        f"`{type(exc).__name__}: {exc}`\n\n"
        f"Please check that `model/{MODEL_PATH.name}` "
        "exists and is a valid checkpoint."
    )
    st.stop()


# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=list(ALLOWED_EXTENSIONS),
    help=f"Accepted formats: {', '.join(ALLOWED_EXTENSIONS)} — Max {MAX_UPLOAD_MB} MB",
)

if uploaded_file is not None:
    # ── File-size validation ──────────────────────────────────────────────
    file_bytes = uploaded_file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_MB:
        st.error(
            f"🚫 File is too large ({file_size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_UPLOAD_MB} MB."
        )
        st.stop()

    # ── Image decoding ────────────────────────────────────────────────────
    try:
        image = Image.open(BytesIO(file_bytes))
        image.verify()  # catch truncated / corrupted files
        image = Image.open(BytesIO(file_bytes))  # reopen after verify
    except (UnidentifiedImageError, Exception) as exc:
        logger.warning("Corrupted or unreadable image: %s", exc)
        st.error(
            "🖼️ **Could not read this image.** "
            "The file may be corrupted or not a valid image format."
        )
        st.stop()

    # ── Display image ─────────────────────────────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Inference ─────────────────────────────────────────────────────────
    with col_result:
        with st.spinner("🔍 Analyzing image…"):
            try:
                probabilities = predict(model, image)
                top5: List[Tuple[str, float]] = get_top_k(probabilities, k=5)
            except Exception as exc:
                logger.exception("Inference failed.")
                st.error(
                    f"⚠️ **Inference error.**\n\n`{type(exc).__name__}: {exc}`"
                )
                st.stop()

        best_name, best_conf = top5[0]

        # ── Primary prediction ────────────────────────────────────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            f'<p class="prediction-label">{best_name.replace("_", " ")}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="prediction-confidence">{best_conf * 100:.1f}% confidence</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Top-5 bar chart ───────────────────────────────────────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            "<h4 style='color:#e2e8f0;margin-bottom:1rem;font-weight:600;'>"
            "Top 5 Predictions</h4>",
            unsafe_allow_html=True,
        )

        max_prob = top5[0][1] if top5[0][1] > 0 else 1.0
        for name, prob in top5:
            bar_width = (prob / max_prob) * 100
            st.markdown(
                f"""
                <div class="bar-container">
                    <div class="bar-label">
                        <span>{name.replace("_", " ")}</span>
                        <span>{prob * 100:.1f}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {bar_width:.1f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

else:
    # ── Empty state ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="glass-card" style="text-align:center; padding:3rem 1rem;">'
        '<p style="font-size:3rem; margin-bottom:0.5rem;">📷</p>'
        '<p style="color:#94a3b8; font-size:1rem;">'
        "Upload an animal image to get started</p>"
        "</div>",
        unsafe_allow_html=True,
    )
