import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# ✅ Update MODEL_PATH: Use relative path for Streamlit Cloud
MODEL_PATH = "CNN_plantdiseases_model.keras"  # Place model in the same directory

# ✅ Function to load the model safely
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except ValueError as e:
        st.error(f"Model loading error: {str(e)}")
        return None

model = load_model()  # Load model once

# ✅ Function to preprocess and predict
def model_predict(image_path):
    H, W, C = 224, 224, 3
    img = cv2.imread(image_path)

    if img is None:
        st.error("Error: Unable to read the image. Please upload a valid image file.")
        return None

    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0  # Normalize
    img = img.reshape(1, H, W, C)  # Reshape

    preds = model.predict(img)
    if preds is None or len(preds) == 0:
        st.error("Error: Model prediction failed.")
        return None

    return np.argmax(preds, axis=-1)[0]

# ✅ Sidebar for Navigation
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# ✅ Display Home Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# ✅ Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("📷 Upload an Image for Disease Detection")

    test_image = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, use_column_width=True, caption="Uploaded Image")

        # Save uploaded file temporarily
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

        if st.button("🔍 Predict"):
            st.snow()  # Animation
            result_index = model_predict(save_path)

            if result_index is not None:
                class_names = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
                    'Tomato___healthy'
                ]

                st.success(f"🍃 Model Prediction: **{class_names[result_index]}**")
