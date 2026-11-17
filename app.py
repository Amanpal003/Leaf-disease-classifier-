import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

# ---------------- SIMPLE DESIGN ----------------
st.markdown("""
<style>
    .stApp {
        background: #f5f8f5;
    }
    .main-title {
        color: #174c38;
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 0;
    }
    .sub-title {
        color: #607568;
        font-size: 16px;
        margin-top: 5px;
    }
    .result-card {
        background: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e1ebe3;
        margin-top: 15px;
    }
    div.stButton > button {
        background: #176b48;
        color: white;
        border: 0;
        border-radius: 9px;
        padding: 10px 22px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background: #105337;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn_model():
    model_path = Path("potato_cnn_model.keras")
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(model_path)

def load_class_names():
    names_path = Path("classification_report.txt")
    if names_path.exists():
        return [
            line.strip()
            for line in names_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return []

model = load_cnn_model()
class_names = load_class_names()

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">🌿 Potato Disease Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Potato Leaf Disease Detection using Deep Learning</p>',
    unsafe_allow_html=True
)
st.divider()

st.write(
    "Upload a clear picture of a potato leaf. "
    "Your CNN model will analyse the image and show the predicted class."
)

# ---------------- CHECK MODEL FILES ----------------
if model is None:
    st.error(
        "Model file not found. Please place "
        "`potato_cnn_model.keras` in the same folder as `app.py`."
    )
    st.stop()

if not class_names:
    st.error(
        "Class names not found. Please place `classification_report.txt` "
        "in the same folder as `app.py`."
    )
    st.stop()

if len(class_names) != model.output_shape[-1]:
    st.error(
        "The number of class names does not match the model output. "
        "Please check classification_report.txt."
    )
    st.stop()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Choose a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded leaf image", width=200)

    if st.button("Detect Disease", use_container_width=True):
        with st.spinner("Analysing leaf image..."):
            # Model input size used in the notebook: 128 x 128
            resized_image = image.resize((128, 128))
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            # The model already contains a Rescaling layer.
            predictions = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        st.success(f"Predicted class: {predicted_class.replace('Potato___', '').replace('potato__', '').replace('_', ' ')}")
        st.metric("Model Predicted Score", f"{confidence:.2f}%")
        st.progress(float(predictions[predicted_index]))
        st.markdown("</div>", unsafe_allow_html=True)

        
else:
    st.info("Please upload a leaf image to start prediction.")

# ---------------- FOOTER ----------------
st.divider()

