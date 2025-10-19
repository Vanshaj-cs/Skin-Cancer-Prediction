import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import io

st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Skin Cancer Classifier")
st.write(
    "Upload a dermatoscopic image and the app will predict the type of skin lesion."
)

MODEL_FILENAME = "skin_model.h5"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]

@st.cache_resource(show_spinner=True)
def load_model():
    model = tf.keras.models.load_model(MODEL_FILENAME)
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Choose a skin image (jpg/png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)

    img = load_img(io.BytesIO(image_bytes), target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    top_idx = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]
    confidence = float(preds[top_idx])

    st.markdown(f"### ✅ Predicted Class: **{top_class}**")
    st.progress(confidence)  
    st.markdown(f"**Confidence:** {confidence:.2f}")

    prob_dict = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    st.write("### Class Probabilities:")
    st.table(prob_dict)

st.markdown("---")
st.markdown("Developed by **Vanshaj Bhardwaj** | Skin Disease Classifier Project")
