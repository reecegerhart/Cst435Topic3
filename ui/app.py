"""Streamlit UI: Cloud #1 (Streamlit Community Cloud).

A thin client: no torch, no model code. Every prediction is an HTTPS call to the
FastAPI service; Grad-CAM PNGs arrive already rendered and are only decoded here.

Secrets (st.secrets / .streamlit/secrets.toml): API_URL, SUPABASE_URL, SUPABASE_ANON_KEY
"""
from __future__ import annotations

import base64

import pandas as pd
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


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Supabase helpers (read-only, via the REST API with the anon key)
# ---------------------------------------------------------------------------
def _sb_get(table: str, params: dict) -> list[dict]:
    base = st.secrets["SUPABASE_URL"].rstrip("/")
    key = st.secrets["SUPABASE_ANON_KEY"]
    r = requests.get(
        f"{base}/rest/v1/{table}",
        params=params,
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_recent(limit: int) -> list[dict]:
    return _sb_get("predictions", {
        "select": "id,created_at,predicted_class,top5,image_hash,served_by_run_id,"
                  "runs!served_by_run_id(id,name,model_type)",
        "order": "created_at.desc",
        "limit": limit,
    })


@st.cache_data(ttl=300, show_spinner=False)
def fetch_runs() -> list[dict]:
    return _sb_get("runs", {
        "select": "id,name,model_type,dataset,epochs,batch_size,lr,weight_decay,"
                  "seed,hyperparams,train_acc,val_acc,test_acc,created_at",
        "order": "created_at.desc",
    })


def pct(v) -> str:
    """Format an accuracy stored either as 0.91 or as 91.0."""
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%" if v <= 1 else f"{v:.2f}%"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
concepts_tab, try_tab, compare_tab, recent_tab, card_tab = st.tabs(
    ["Concepts", "Try It", "Compare Models", "Recent Classifications", "Model Card"]
)

# ----------------------------- Concepts ------------------------------------
with concepts_tab:
    st.header("Concepts")
    st.markdown(
        "A Fashion-MNIST image is a 28×28 grid of numbers between 0 and 1. A convolutional "
        "network turns that grid into a class label in three ideas, each of which you can "
        "see at work in the Try It tab."
    )

    st.subheader("1. Convolution")
    st.markdown(
        "A convolutional layer slides a small grid of learned weights, called a kernel "
        "(usually 3×3), across the image. At each position it multiplies the kernel by the "
        "pixels underneath and adds the results into one output number. The same kernel is "
        "reused everywhere, so a detector for, say, a vertical edge works anywhere in the "
        "image, and the layer needs far fewer parameters than a fully connected one."
    )
    st.latex(r"(I * K)[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m,\, j+n]\; K[m,n]")
    st.markdown(
        "Many kernels run in parallel, each producing its own feature map. Early layers "
        "tend to learn edges and textures. Later layers combine those into shapes like "
        "sleeves, straps, or heels."
    )

    st.subheader("2. Receptive field")
    st.markdown(
        "Each output value depends on only a small patch of the input, called its "
        "receptive field. One 3×3 convolution sees a 3×3 patch. Stack a second 3×3 layer "
        "on top and each of its outputs depends on a 5×5 patch of the original image, "
        "because it combines nine neighbors that each looked at their own 3×3 patch. "
        "Pooling grows the field faster. Deep in the network a single value can depend on "
        "most of the image, which is how the model can respond to the overall shape of a "
        "coat and not just local texture."
    )
    st.latex(r"r_{\ell} = r_{\ell-1} + (k_\ell - 1)\prod_{i<\ell} s_i")
    st.caption("r is the receptive field size, k the kernel size, s the stride of each earlier layer.")

    st.subheader("3. Grad-CAM")
    st.markdown(
        "Grad-CAM answers the question \"which regions pushed the model toward this "
        "class?\" Take the last convolutional layer's feature maps A^k. For the predicted "
        "class c, compute how much the class score y^c changes when each feature map "
        "changes (the gradient), and average that over space to get one weight per map. "
        "The heatmap is the weighted sum of the maps, keeping only the positive part."
    )
    st.latex(r"\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}"
             r"\qquad L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\!\Big(\sum_k \alpha_k^c A^k\Big)")
    st.markdown(
        "The result is upsampled to the input size and laid over the image as the warm "
        "colors you see. It shows what influenced the score, not what the model "
        "understands. It is also coarse, because the last conv layer has a much lower "
        "resolution than the input."
    )

# ------------------------------ Try It -------------------------------------
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

# --------------------------- Compare Models --------------------------------
with compare_tab:
    st.header("Compare Models")
    st.markdown(
        "Upload one image and run it through both models at once. Where they disagree, "
        "or where their heatmaps point at different regions, you learn something about "
        "what the improved model changed."
    )
    cmp_up = st.file_uploader("PNG or JPG, up to 5 MB", type=["png", "jpg", "jpeg"], key="cmp_up")
    if cmp_up is not None and st.button("Compare both models", type="primary"):
        try:
            with st.spinner("Running both models... the free API server can take up to a minute to wake up."):
                st.session_state["cmp_results"] = [
                    classify(cmp_up.getvalue(), cmp_up.name, cmp_up.type or "image/png", m)
                    for m in ("baseline", "improved")
                ]
        except requests.RequestException:
            st.error("Could not reach the API. If it was idle, wait about 30 seconds and try again.")
        except RuntimeError as exc:
            st.error(str(exc))

    cmp_results = st.session_state.get("cmp_results")
    if cmp_results:
        base_res, imp_res = cmp_results
        if base_res["class_name"] == imp_res["class_name"]:
            st.success(f"Both models predict **{base_res['class_name']}**. "
                       f"Confidence: baseline {base_res['confidence']:.1%}, "
                       f"improved {imp_res['confidence']:.1%}.")
        else:
            st.warning(f"The models disagree: baseline says **{base_res['class_name']}** "
                       f"({base_res['confidence']:.1%}), improved says "
                       f"**{imp_res['class_name']}** ({imp_res['confidence']:.1%}).")
        for res in cmp_results:
            show_result(res)
            st.divider()

# ----------------------- Recent Classifications ----------------------------
with recent_tab:
    st.header("Recent Classifications")
    st.markdown("The latest predictions logged to Supabase. Only a SHA-256 hash of each "
                "image is stored, never the pixels.")
    n = st.slider("How many to show", 5, 100, 25, step=5)
    if st.button("Refresh"):
        fetch_recent.clear()
    try:
        rows = fetch_recent(n)
    except Exception as exc:
        st.error(f"Could not load history from Supabase: {exc}")
        rows = None

    if rows:
        records = []
        for r in rows:
            run = r.get("runs") or {}
            top5 = r.get("top5") or []
            prob = top5[0].get("prob") if top5 else None
            records.append({
                "When": r.get("created_at"),
                "Prediction": r.get("predicted_class"),
                "Confidence": prob * 100 if prob is not None else None,
                "Model": MODEL_LABELS.get(run.get("model_type"), run.get("model_type")),
                "Run": run.get("name") or r.get("served_by_run_id"),
                "Image hash": (r.get("image_hash") or "")[:12],
            })
        df = pd.DataFrame(records)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={"Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=100.0, format="%.1f%%")},
        )
        top = df["Prediction"].value_counts()
        st.caption(f"Most common prediction in this view: {top.index[0]} ({top.iloc[0]} of {len(df)}).")
    elif rows is not None:
        st.info("No predictions logged yet. Classify something in the Try It tab.")

# ------------------------------ Model Card ---------------------------------
with card_tab:
    st.header("Model Card")
    st.markdown(
        "**Intended use.** An educational demo of how a small CNN classifies images and "
        "how Grad-CAM explains the result. It is not meant for any real decision-making."
    )
    st.markdown(
        "**Training data.** Fashion-MNIST: 60,000 training and 10,000 test grayscale "
        "28×28 images across 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, "
        "Sandal, Shirt, Sneaker, Bag, Ankle boot). Uploaded photos are converted to this "
        "format before classification."
    )

    st.subheader("Models")
    try:
        runs = fetch_runs()
    except Exception as exc:
        st.error(f"Could not load runs from Supabase: {exc}")
        runs = []

    latest: dict[str, dict] = {}
    for r in runs:  # newest first, so the first one seen per type wins
        latest.setdefault(r["model_type"], r)

    if latest:
        def col(model_type: str) -> list:
            r = latest.get(model_type)
            if not r:
                return ["not trained"] * 9
            return [r["name"], r["dataset"], r["epochs"], r["batch_size"], r["lr"],
                    r["weight_decay"], pct(r["train_acc"]), pct(r["val_acc"]), pct(r["test_acc"])]

        st.table(pd.DataFrame({
            "": ["Run", "Dataset", "Epochs", "Batch size", "Learning rate",
                 "Weight decay", "Train accuracy", "Validation accuracy", "Test accuracy"],
            "Baseline": col("baseline"),
            "Improved": col("improved"),
        }).set_index("").astype(str))

        for mt in ("baseline", "improved"):
            hp = (latest.get(mt) or {}).get("hyperparams")
            if hp:
                with st.expander(f"{MODEL_LABELS[mt]}: full hyperparameters"):
                    st.json(hp)
    else:
        st.info("No training runs found in Supabase yet.")

    st.markdown("**Architecture and what changed.** _Add two or three sentences here "
                "describing the baseline layers and what the improved model adds._")

    st.subheader("Limitations")
    st.markdown(
        "The model must choose one of ten classes for any input, so a photo of a dog "
        "still gets a clothing label. The softmax confidence is uncalibrated and can be "
        "high on wrong answers. Real photos differ from Fashion-MNIST (backgrounds, "
        "color, lighting, framing), so accuracy on uploads will be well below the test "
        "set number. Shirt, T-shirt, Pullover, and Coat are the classes most often "
        "confused with each other. Grad-CAM is a coarse, last-layer view of influence, "
        "not proof of reasoning."
    )

    st.subheader("Privacy")
    st.markdown("Uploaded images are processed in memory and never stored. Each "
                "prediction is logged with a SHA-256 hash of the image, its dimensions, "
                "the predicted class, the top five probabilities, and which training run "
                "served it.")