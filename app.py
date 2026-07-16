from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Digit Recognition", page_icon="✍️", layout="wide")
ROOT = Path(__file__).parent

@st.cache_resource
def load_assets():
    model = joblib.load(ROOT / "digit_recognition_pipeline.joblib")
    metadata = json.loads((ROOT / "model_metadata.json").read_text())
    return model, metadata

def prepare_image(uploaded):
    image = Image.open(uploaded).convert("L")
    if np.asarray(image).mean() > 127:
        image = ImageOps.invert(image)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    side = max(image.size)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(image, ((side-image.width)//2, (side-image.height)//2))
    image8 = canvas.resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(image8, dtype=float) / 255.0 * 16.0
    return image8, pixels

model, metadata = load_assets()
st.title("Handwritten Digit Recognition")
st.caption("COM763 Portfolio Task 1 • PCA + tuned RBF Support Vector Machine")

with st.sidebar:
    st.header("Model information")
    st.metric("Held-out accuracy", f"{metadata['test_accuracy']:.1%}")
    st.metric("CV macro F1", f"{metadata['cv_macro_f1']:.1%}")
    st.info("Educational demonstration only. Confidence is not a guarantee of correctness.")

tab1, tab2 = st.tabs(["Upload an image", "Edit an 8×8 pixel grid"])
with tab1:
    uploaded = st.file_uploader("Upload one handwritten digit (PNG/JPG)", type=["png", "jpg", "jpeg"])
    st.caption("Best results: one dark digit centred on a plain light background.")
    if uploaded:
        image8, pixels = prepare_image(uploaded)
        prediction = int(model.predict(pixels.reshape(1, -1))[0])
        probs = model.predict_proba(pixels.reshape(1, -1))[0]
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.image(uploaded, caption="Uploaded image", width=220)
        c2.image(image8.resize((240, 240), Image.Resampling.NEAREST), caption="Model input (8×8)", width=240)
        c3.metric("Predicted digit", prediction)
        c3.metric("Confidence", f"{probs[prediction]:.1%}")
        c3.bar_chart(pd.DataFrame({"Digit": range(10), "Probability": probs}).set_index("Digit"))

with tab2:
    st.write("Set each intensity from 0 (black) to 16 (bright ink), then classify.")
    edited = st.data_editor(pd.DataFrame(np.zeros((8, 8), dtype=int)), hide_index=True,
                            use_container_width=True, key="pixel_editor")
    if st.button("Classify pixel grid", type="primary"):
        pixels = edited.to_numpy(dtype=float).clip(0, 16)
        prediction = int(model.predict(pixels.reshape(1, -1))[0])
        probs = model.predict_proba(pixels.reshape(1, -1))[0]
        st.success(f"Predicted digit: {prediction} • confidence: {probs[prediction]:.1%}")
        preview = Image.fromarray(np.uint8(pixels / 16 * 255)).resize((320, 320), Image.Resampling.NEAREST)
        st.image(preview, width=320)

st.divider()
st.subheader("How the system works")
st.write("Input image → 64 pixel features → standardisation → PCA → RBF SVM → class probabilities")
