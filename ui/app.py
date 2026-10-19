"""Streamlit UI: Cloud #1 (Streamlit Community Cloud).

A thin client: no torch, no model code. Every prediction is an HTTPS call to the
FastAPI service; Grad-CAM PNGs arrive already rendered and are only decoded here.

Secrets (st.secrets / .streamlit/secrets.toml): API_URL, SUPABASE_URL, SUPABASE_ANON_KEY
"""
from __future__ import annotations

import base64

import requests
import streamlit as st

st.set_page_config(page_title="See-Sense", page_icon="🔍", layout="wide")

try:
    API_URL = st.secrets["API_URL"].rstrip("/")
except Exception:
    st.error("Missing secret `API_URL`. Copy `.streamlit/secrets.toml.example` to "
             "`.streamlit/secrets.toml` and fill it in.")
    st.stop()

MODEL_LABELS = {"improved": "Improved model", "baseline": "Baseline model"}

st.title("🔍 See-Sense: how a CNN \"sees\"")
st.caption("Streamlit (this UI) → FastAPI (CNN + Grad-CAM) → Supabase (history). "
           "A Fashion-MNIST classifier with a visible explanation.")


def classify(raw: bytes, filename: str, mime: str, model: str) -> dict:
    r = requests.post(
        f"{API_URL}/predict",
        files={"file": (filename, raw, mime)},
        data={"model": model},
        timeout=120,
    )
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"API returned {r.status_code}: {detail}")
    return r.json()


def show_result(res: dict) -> None:
    st.subheader(MODEL_LABELS.get(res["model_name"], res["model_name"]))
    if res["low_confidence"]:
        st.warning("Low confidence: the model is unsure, and its answer should not be trusted.")

    c_in, c_cam, c_prob = st.columns(3)
    c_in.image(base64.b64decode(res["input_png"]), caption="What you uploaded",
               use_column_width=True)
    c_cam.image(base64.b64decode(res["cam_png"]),
                caption="Grad-CAM: warm colors pushed the model toward its answer",
                use_column_width=True)
    with c_prob:
        st.markdown(f"### {res['class_name']}")
        st.caption(f"{res['confidence']:.1%} confident (uncalibrated softmax)")
        for item in res["top5"]:
            p = min(max(item["prob"], 0.0), 1.0)
            st.progress(p, text=f"{item['class_name']}: {p:.2%}")

    if res["logged"]:
        st.caption(f"Logged as prediction #{res['prediction_id']} on run #{res['run_id']} · "
                   f"SHA-256 {res['image_sha256'][:12]}… · your pixels are never stored")
    else:
        st.caption("⚠ This prediction could not be logged to the database.")


concepts_tab, try_tab, compare_tab, recent_tab, card_tab = st.tabs(
    ["Concepts", "Try It", "Compare Models", "Recent Classifications", "Model Card"]
)

with concepts_tab:
    st.info("Built in a later step: convolution, receptive field, and Grad-CAM derivations.")

with try_tab:
    st.header("Try It")
    st.markdown("Upload an image to see what the CNN decides, and **where it looked** to decide.")
    st.info(
        "This model only knows 10 clothing and accessory classes. It must pick one of "
        "them for *any* image, even a photo of a dog. Try something that is not "
        "clothing and watch how confidently it answers."
    )

    up = st.file_uploader("PNG or JPG, up to 5 MB", type=["png", "jpg", "jpeg"])
    choice = st.radio("Model", ["Improved", "Baseline", "Both"], horizontal=True)
    if up is not None:
        st.image(up, caption=up.name, width=160)

    if up is not None and st.button("Classify", type="primary"):
        st.session_state["try_results"] = []
        models = ["improved", "baseline"] if choice == "Both" else [choice.lower()]
        try:
            with st.spinner("Classifying... the free API server can take up to a minute to wake up."):
                st.session_state["try_results"] = [
                    classify(up.getvalue(), up.name, up.type or "image/png", m) for m in models
                ]
        except requests.RequestException:
            st.error("Could not reach the API. If it was idle, wait about 30 seconds and try again.")
        except RuntimeError as exc:
            st.error(str(exc))

    for res in st.session_state.get("try_results", []):
        show_result(res)
        st.divider()

    with st.expander("How to read the heatmap"):
        st.markdown(
            "- The network never sees \"a shirt\". It sees a grid of 784 numbers.\n"
            "- **Grad-CAM** highlights the regions whose activations most increased the "
            "score of the predicted class. It shows what *influenced* the answer, not "
            "what the model \"understands\".\n"
            "- A confident answer is not necessarily a correct one. Softmax always sums "
            "to 100%, even for inputs the model has never seen anything like."
        )

with compare_tab:
    st.info("Built in a later step: baseline vs. improved on a held-out gallery.")

with recent_tab:
    st.info("Built in a later step: recent predictions from Supabase, joined to runs.")

with card_tab:
    st.info("Built in a later step: the Model Card.")