import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Load model
MODEL_PATH = r"C:\Users\kotha\Downloads\CNN_plantdiseases_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to add a background
def add_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to process and predict the disease
def model_predict(image):
    H, W = 224, 224
    img = cv2.resize(image, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0  # Rescaling
    img = np.expand_dims(img, axis=0)  # Reshaping for model input
    
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Sidebar
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display an image
img = Image.open(r"C:\Users\kotha\Downloads\p.png")
st.image(img)

# Add background
add_background("https://plus.unsplash.com/premium_photo-1663962158789-0ab624c4f17d?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8cGxhbnRzfGVufDB8fDB8fHww")

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Upload an Image for Disease Recognition")
    test_image = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        # Convert file to OpenCV format
        file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if st.button("Show Image"):
            st.image(img, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            result_index = model_predict(img)

            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']

            st.success(f"Model Prediction: **{class_name[result_index]}**")
