import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Download model from Google Drive if not present
@st.cache_resource
def load_model():
    model_path = 'brain_tumor_resnet50_model.keras'
    if not os.path.exists(model_path):
        st.write("Downloading model...")
        gdown.download('https://drive.google.com/file/d/1erswbAb97hei7KUihetMFTw6R3CXr1rH/view?usp=drive_link', model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()
IMG_SIZE = (224, 224)
class_names = {0: 'yes (Tumor Detected)', 1: 'no (No Tumor)'}

st.title("Brain Tumor Detection Demo")
st.write("Upload an MRI image for prediction.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    img_array = np.array(image.resize(IMG_SIZE)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    predicted_class = 1 if prob > 0.5 else 0
    label = class_names[predicted_class]
    confidence = max(prob, 1 - prob) * 100

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")
    if predicted_class == 0:
        st.warning("Potential tumor—consult a doctor!")
    else:
        st.success("No tumor detected.")