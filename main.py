import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Modeli yükleyin
model_path = 'x-ray-model.h5'
if not os.path.exists(model_path):
    st.error(f"Model File '{model_path}' not found.")
    st.stop()

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# Görüntüyü hazırlayın
def prepare_image(img_path, target_size=(512, 512)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizasyon
    return img_array


# Görüntüyü tahmin edin
def predict_cancer(model, img_path):
    img_array = prepare_image(img_path)
    prediction = model.predict(img_array)
    return "Cancer" if prediction[0][0] > 0.5 else "Not Cancer"


# Geçici dosya yolunu tanımlayın
temp_file_path = "temp_image.png"

# Streamlit uygulaması
st.title("Breast Cancer Detection")
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50;'>Please upload an X-ray image:</h2>",
    unsafe_allow_html=True)

# Kişisel veri işleme onayı
consent_given = st.checkbox("I agree to the processing of my personal data")

if consent_given:
    # Dosya yükleme alanı
    uploaded_file = st.file_uploader("Upload image file",
                                     type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # Daha önce yüklenmiş bir geçici dosya varsa silin
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            # Görüntüyü geçici bir dosyaya kaydedin
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Tahmin yapın
            result = predict_cancer(model, temp_file_path)

            # Görüntüyü gösterin ve sonucu belirtin
            img = Image.open(temp_file_path)
            st.image(img,
                     caption='Uploaded image',
                     use_column_width=True,
                     width=512)
            st.markdown(
                f"<h3 style='text-align: center; color: #FF5733;'>Forecast Result: {result}</h3>",
                unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing the image: {e}")
else:
    st.warning("Please agree to the processing of your personal data")

# Sayfa stili
st.markdown("""
    <style>
        .reportview-container {
            background: #f5f5dc; /* Kum rengi arka plan */
            color: #333;
        }
        .sidebar .sidebar-content {
            background: #f5f5dc; /* Kum rengi arka plan */
        }
        .stTitle {
            color: #4CAF50;
        }
        .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stCheckbox>div>div>label {
            color: #4CAF50;
        }
        .stFileUploader>label {
            color: #4CAF50;
        }
        .stAlert {
            background-color: #ffcccc;
            color: #990000;
            border-radius: 5px;
            padding: 10px;
        }
    </style>
    """,
            unsafe_allow_html=True)
