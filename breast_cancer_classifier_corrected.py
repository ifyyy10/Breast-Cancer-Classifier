import streamlit as st
import numpy as np
import gdown  # For downloading files from Google Drive
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# Download the model from Google Drive
@st.cache(allow_output_mutation=True)
def download_model():
    url = 'https://drive.google.com/uc?id=1-YNdnqs-YMmIv6U2lmL9uT3-Lor85cuB'  # Direct link to model
    output = 'breast_cancer_classifier.h5'  # File name to save locally
    gdown.download(url, output, quiet=False)
    model = load_model(output)  # Load the model after downloading
    return model

# Load the model
model = download_model()

# Set up the Streamlit app layout
st.title("Breast Cancer Image Classification")
st.write("Upload an image to classify whether it's benign, malignant, normal, or unknown.")

# Upload image section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image to fit model input
    img = img.convert('RGB')  # Ensure the image is in RGB mode
    img = img.resize((224, 224))  # Resize image to match input size of model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess input for DenseNet

    # Predict using the loaded model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Class names corresponding to predictions
    class_names = ['Benign', 'Malignant', 'Normal', 'Unknown']
    confidence = predictions[0][predicted_class] * 100

    # Display the result
    st.write(f"Prediction: **{class_names[predicted_class]}** with confidence {confidence:.2f}%")
