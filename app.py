import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import os
import io

st.set_page_config(page_title="Skin Disease Classifier", layout="centered")

st.title("Skin Disease Classifier")
st.write("Upload a dermatoscopic image and the app will predict the skin lesion type.")

MODEL_FILENAME = "skin_model.h5"   
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model(MODEL_FILENAME)
    return model

model = None
try:
    model = load_model()
except Exception as e:
    st.error(
        "Model file not found or failed to load. Make sure `skin_model.h5` is present in the app folder.\n\n"
        "Detailed error: " + str(e)
    )

uploaded_file = st.file_uploader("Choose an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:

    image_bytes = uploaded_file.read()

    st.image(image_bytes, caption="Uploaded image", use_column_width=True)
    st.write("")


    try:
        img = load_img(io.BytesIO(image_bytes), target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]
        top_idx = int(np.argmax(preds))
        top_class = CLASS_NAMES[top_idx]
        confidence = float(preds[top_idx])

        st.markdown(f"### Prediction: **{top_class}**")
        st.markdown(f"**Confidence:** {confidence:.3f}")

        st.write("All class probabilities:")
        probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        st.table(probs)

    except Exception as e:
        st.error("Failed to preprocess or predict the uploaded image. Error: " + str(e))

st.markdown("---")
st.caption("Model: HAM10000-based skin lesion classifier. This demo uses a CNN trained to predict 7 lesion types.")
