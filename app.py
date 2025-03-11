import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Class label mapping
CLASS_LABELS = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Function to set background
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

# Load and predict function
def model_predict(uploaded_file):
    model = tf.keras.models.load_model("CNN_plantdiseases_model.keras")

    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    result_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return result_index, confidence

# UI
add_background("https://images.unsplash.com/photo-1596631133404-2f3948ed7ccc?fm=jpg&q=60&w=3000")

st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("📌 Select Page", ["HOME", "DISEASE RECOGNITION"])

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("🔍 Upload an Image for Disease Recognition")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("🔍 Analyzing... Please wait.")

            result_index, confidence = model_predict(test_image)
            predicted_class = CLASS_LABELS[result_index]

            st.success(f"🌿 **Prediction:** {predicted_class}")
            st.info(f"🎯 **Confidence:** {confidence:.2f}%")

            if confidence < 50:
                st.warning("⚠️ Low confidence in prediction. Try using a clearer image.")

