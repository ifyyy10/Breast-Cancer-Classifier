import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# Function to download model from Google Drive
# def download_model():
#     file_id = '1xJBec2aPibWKWYMfoOz2K5hiq16QeI0x'  # Extracted file ID from your Google Drive shareable link
#     url = f'https://drive.google.com/uc?id={file_id}'
#     output = 'breast_cancer_classifier_corrected.h5'
#     gdown.download(url, output, quiet=False)

# # Check if model file exists, if not, download it
# model_file = 'breast_cancer_classifier_corrected.h5'
# if not os.path.isfile(model_file):
#     download_model()

# # Load the pre-trained model
# model = load_model(model_file)

# Define the image preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array

# Define the classification function
def classify_image(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    classes = ['benign', 'malignant', 'normal', 'unknown']
    return classes[np.argmax(prediction)]

# Streamlit app layout
st.title('Breast Cancer Image Classifier')
st.write("Upload an image to classify it.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Classify the image
    result = classify_image(image)
    st.write(f"Prediction: {result}")
