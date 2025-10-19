# Skin Cancer Prediction App 🩺

**Tagline:** A deep learning-powered Streamlit app that detects skin lesion types from dermatoscopic images.  

[Live Demo on Streamlit](https://share.streamlit.io/your-username/Skin-Cancer-Prediction/main/app.py)  

---

## Overview

This project is a **Skin Cancer Classification** web application built using **TensorFlow** and **Streamlit**. Users can upload a dermatoscopic image (JPG/PNG), and the app predicts which type of skin lesion it belongs to. The model was trained on the **HAM10000 dataset**, which contains 7 classes of skin lesions:

- **akiec**: Actinic keratoses and intraepithelial carcinoma  
- **bcc**: Basal cell carcinoma  
- **bkl**: Benign keratosis-like lesions  
- **df**: Dermatofibroma  
- **mel**: Melanoma  
- **nv**: Melanocytic nevi  
- **vasc**: Vascular lesions  

> ⚠️ **Disclaimer:** This tool is for research and educational purposes only. It is not a medical diagnosis.

---

## Features

- Upload a skin image and preview it before analysis  
- Predicts lesion type with **confidence scores**  
- Displays **all class probabilities** in a table  
- Friendly and responsive **Streamlit interface**  
- Quick deployment without backend setup  

---

## Tech Stack

- **Python 3.x**  
- **TensorFlow / Keras** for CNN model  
- **Streamlit** for web app UI  
- **NumPy / Pillow** for image preprocessing  
- **Git LFS** for storing large model files  

---

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/Vanshaj-cs/Skin-Cancer-Prediction.git
cd Skin-Cancer-Prediction
