import pandas as pd
import streamlit as st
from PIL import Image

import oct_model

st.set_page_config(page_title="Retinal OCT Classifier", page_icon="👁️", layout="wide")

CLASS_INFO = {
    "AMD": "Age-related Macular Degeneration",
    "CNV": "Choroidal Neovascularization",
    "CSR": "Central Serous Retinopathy",
    "DME": "Diabetic Macular Edema",
    "DR": "Diabetic Retinopathy",
    "DRUSEN": "Drusen deposits",
    "MH": "Macular Hole",
    "NORMAL": "Normal retina",
}


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return oct_model.load_model()


# ---------------- header ----------------
st.title("👁️ Retinal OCT Image Classifier")
st.caption("Upload a retinal OCT scan. The model shows how strongly the image matches each class.")

try:
    model, cfg = get_model()
except FileNotFoundError as e:
    st.error(f"Model file not found: `{e}`")
    st.info("Put `best_oct_model.weights.h5` and `model_config.json` (exported from Colab) "
            "inside the `model/` folder, then refresh this page.")
    st.stop()

class_names = cfg["class_names"]
img_size = cfg["img_size"]

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("About the model")
    st.write(f"**Architecture:** {cfg['model_name']}")
    st.write(f"**Input size:** {img_size} x {img_size}")
    st.write(f"**Classes ({len(class_names)}):**")
    for c in class_names:
        st.write(f"- **{c}**: {CLASS_INFO.get(c, '')}")
    st.divider()
    use_tta = st.toggle("Test-time augmentation (average with mirrored image)", value=True)
    show_cam = st.toggle("Show Grad-CAM heatmap", value=True)
    st.divider()
    st.caption("⚠️ Research demo only. Not a medical diagnosis tool.")

# ---------------- upload ----------------
uploaded = st.file_uploader("Upload an OCT image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("👆 Upload an image to start.")
    st.stop()

image = Image.open(uploaded)
batch = oct_model.preprocess(image, img_size)

with st.spinner("Analysing image..."):
    probs = oct_model.predict_probs(model, batch, use_tta=use_tta)

top = int(probs.argmax())
top_name, top_pct = class_names[top], float(probs[top]) * 100

# ---------------- results ----------------
left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Uploaded image")
    st.image(image, width="stretch")
    if show_cam:
        with st.spinner("Computing Grad-CAM..."):
            cam = oct_model.gradcam_overlay(model, batch, class_idx=top)
        if cam is not None:
            st.subheader("Where the model looked")
            st.image(cam, caption=f"Grad-CAM for: {top_name}", width="stretch")
            st.caption("Red/yellow areas influenced the prediction the most.")

with right:
    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Best match", top_name)
    c2.metric("Match percentage", f"{top_pct:.2f}%")
    if CLASS_INFO.get(top_name):
        st.write(CLASS_INFO[top_name])

    if top_pct < 60:
        st.warning("Low confidence. The image may be unclear, or it may not be an OCT scan of this type.")

    st.subheader("Match percentage for every class")
    df = (pd.DataFrame({"Class": class_names, "Match (%)": probs * 100})
          .sort_values("Match (%)", ascending=False).reset_index(drop=True))
    for _, row in df.iterrows():
        a, b = st.columns([1, 4])
        a.write(f"**{row['Class']}**")
        b.progress(min(max(row["Match (%)"] / 100, 0.0), 1.0), text=f"{row['Match (%)']:.2f}%")

    with st.expander("Show as chart / table"):
        st.bar_chart(df.set_index("Class"))
        st.dataframe(df.style.format({"Match (%)": "{:.2f}"}), width="stretch")

st.divider()
st.caption("The model always picks the closest of its known classes, even for images that are not OCT scans. "
           "Percentages show model confidence, not a clinical diagnosis.")
