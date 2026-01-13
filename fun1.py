import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model_path = "diabetic_retinopathy1_model.h5"  # Update with your model path
model = load_model(model_path)

# Define categories
categories = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Function to preprocess the image
def preprocess_image(image, img_size=128):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict_image_category(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_label = categories[predicted_class]
    return predicted_label

# Streamlit Page Configuration
st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="centered")

# Custom CSS for Styling
st.markdown(
    """
    <style>
        body {
            background-color: #f8f9fa;
        }
        .title {
            font-size: 50px;
            font-weight: bold;
            color: black;
            text-align: center;
        }
        .uploaded-img {
            display: block;
            margin: auto;
            border-radius: 15px;
            border: 4px solid #FF4B4B;
            padding: 10px;
            background-color: #fff;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            width: 400px;
        }
        .result-box {
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            background-color: #1e1e2f;
            color: #FFD700;
            width: 60%;
            margin: auto;
        }
        .predict-btn {
            display: block;
            width: 50%;
            margin: auto;
            padding: 15px;
            background-color: #FF4B4B;
            color: white;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: 0.3s;
        }
        .predict-btn:hover {
            background-color: #e63e3e;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="title">🩺 Diabetic Retinopathy Classification</h1>', unsafe_allow_html=True)

# Upload an image
uploaded_file = st.file_uploader("📤 Upload a Retinal Image...", type=["jpg", "jpeg", "png"])

# Display uploaded image in center
if uploaded_file is not None:
    st.markdown("<br>", unsafe_allow_html=True)  # Adds space
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", width=400, use_column_width=False)

# Centering the Predict Button
if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Predict Category", key="predict"):
            predicted_category = predict_image_category(image)
            st.markdown(f'<p class="result-box">🩺 Prediction: {predicted_category}</p>', unsafe_allow_html=True)
