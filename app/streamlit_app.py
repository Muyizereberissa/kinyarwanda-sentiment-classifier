"""Streamlit UI for Kinyarwanda tweet sentiment classification."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ISO_CODE, LANGUAGE
from src.predict import Predictor

# Streamlit Cloud stores secrets in st.secrets, not in the process environment.
try:
    secret_model = st.secrets.get("MODEL_ID", "")
except Exception:
    secret_model = ""
if secret_model and not os.environ.get("MODEL_ID"):
    os.environ["MODEL_ID"] = str(secret_model)

st.set_page_config(
    page_title="Kinyarwanda Sentiment",
    page_icon="🇷🇼",
    layout="centered",
)

st.title("Kinyarwanda Sentiment Classifier")
st.caption(
    f"{LANGUAGE} ({ISO_CODE}) · AfriSenti · negative / neutral / positive"
)

EXAMPLES = [
    "Urakoze cyane, serivisi yanyu irakomeye!",
    "Ntacyo mpagarariye kuri iki gitekerezo.",
    "Ntabwo nishimira uko byagenze uyu munsi.",
]


@st.cache_resource
def load_model():
    return Predictor()


try:
    predictor = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Train a model locally, then refresh this app.")
    st.stop()

text = st.text_area(
    "Enter Kinyarwanda text (e.g. a tweet)",
    height=120,
    placeholder="Andika umubare w'amanota...",
)

col1, col2 = st.columns(2)
with col1:
    run = st.button("Classify", type="primary", use_container_width=True)
with col2:
    example = st.selectbox("Try an example", ["—"] + EXAMPLES)

if example != "—" and not text:
    text = example

if run or (text and example != "—"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        label, scores = predictor.predict_one(text)
        emoji = {"positive": "🙂", "neutral": "😐", "negative": "🙁"}.get(label, "")
        st.subheader(f"Prediction: {label} {emoji}")
        st.progress(scores.get(label, 0.0))
        st.write("Class probabilities")
        for lab in ["negative", "neutral", "positive"]:
            st.write(f"**{lab}**: {scores.get(lab, 0.0):.1%}")

st.divider()
st.markdown(
    """
**About:** Fine-tuned on the [AfriSenti](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) 
Kinyarwanda subset (SemEval 2023 Task 12). For research and education—not for high-stakes decisions.
"""
)
