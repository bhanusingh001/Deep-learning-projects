import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import time
import os
# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Visual Appeal ---
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background-color: #F63366;
        color: white;
        height: 3em;
        font-weight: 600;
        font-size: 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        border-color: #F63366;
        background-color: #ff4b7a;
        box-shadow: 0 4px 15px rgba(246, 51, 102, 0.4);
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 2rem;
    }
    h1 {
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.title("🧬 Deep Learning Breast Cancer Predictor")
st.markdown("""
    Welcome to the Breast Cancer classification tool. This application uses a **Neural Network**
    trained on cell nuclei characteristics to predict whether a breast mass is **Malignant** or **Benign**.
    Please enter the 30 features below.
""")

# Load artifacts
@st.cache_resource
def load_scaler():
    try:
        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'breast_cancer_model.h5')
        return keras.models.load_model(model_path)
    except Exception as e:
        return None

scaler = load_scaler()
model = load_model()

if model is None or scaler is None:
    st.error("Error: Model or Scaler not found. Please ensure `breast_cancer_model.h5` and `scaler.pkl` exist by running `train_model.py` first.")
    st.stop()


# Sidebar Information
with st.sidebar:
    st.header("🔬 About")
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865589.png", width=150)
    st.info("""
    **Dataset**: Computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
    **Features**: 30 numerical variables.
    """)
    
    st.markdown("### How to use:")
    st.markdown("1. Enter the values for Mean, SE, and Worst features.")
    st.markdown("2. Click **Predict Diagnosis**.")
    st.markdown("3. View the result and confidence score.")

# Group the 30 inputs into 3 tabs
tab1, tab2, tab3 = st.tabs(["📊 Mean Features", "📉 Standard Error (SE) Features", "⚠️ Worst Features"])

# Dictionaries to store inputs
features = {}

# 1. Mean Features
with tab1:
    st.subheader("Mean Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        features['radius_mean'] = st.number_input("Radius Mean", value=17.99)
        features['texture_mean'] = st.number_input("Texture Mean", value=10.38)
        features['perimeter_mean'] = st.number_input("Perimeter Mean", value=122.8)
        features['area_mean'] = st.number_input("Area Mean", value=1001.0)
    with col2:
        features['smoothness_mean'] = st.number_input("Smoothness Mean", value=0.1184, format="%.4f")
        features['compactness_mean'] = st.number_input("Compactness Mean", value=0.2776, format="%.4f")
        features['concavity_mean'] = st.number_input("Concavity Mean", value=0.3001, format="%.4f")
    with col3:
        features['concave points_mean'] = st.number_input("Concave Points Mean", value=0.1471, format="%.4f")
        features['symmetry_mean'] = st.number_input("Symmetry Mean", value=0.2419, format="%.4f")
        features['fractal_dimension_mean'] = st.number_input("Fractal Dimension Mean", value=0.07871, format="%.5f")

# 2. SE Features
with tab2:
    st.subheader("Standard Error Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        features['radius_se'] = st.number_input("Radius SE", value=1.095, format="%.4f")
        features['texture_se'] = st.number_input("Texture SE", value=0.9053, format="%.4f")
        features['perimeter_se'] = st.number_input("Perimeter SE", value=8.589, format="%.4f")
        features['area_se'] = st.number_input("Area SE", value=153.4)
    with col2:
        features['smoothness_se'] = st.number_input("Smoothness SE", value=0.006399, format="%.6f")
        features['compactness_se'] = st.number_input("Compactness SE", value=0.04904, format="%.5f")
        features['concavity_se'] = st.number_input("Concavity SE", value=0.05373, format="%.5f")
    with col3:
        features['concave points_se'] = st.number_input("Concave Points SE", value=0.01587, format="%.5f")
        features['symmetry_se'] = st.number_input("Symmetry SE", value=0.03003, format="%.5f")
        features['fractal_dimension_se'] = st.number_input("Fractal Dimension SE", value=0.006193, format="%.6f")

# 3. Worst Features
with tab3:
    st.subheader("Worst (Largest) Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        features['radius_worst'] = st.number_input("Radius Worst", value=25.38)
        features['texture_worst'] = st.number_input("Texture Worst", value=17.33)
        features['perimeter_worst'] = st.number_input("Perimeter Worst", value=184.6)
        features['area_worst'] = st.number_input("Area Worst", value=2019.0)
    with col2:
        features['smoothness_worst'] = st.number_input("Smoothness Worst", value=0.1622, format="%.4f")
        features['compactness_worst'] = st.number_input("Compactness Worst", value=0.6656, format="%.4f")
        features['concavity_worst'] = st.number_input("Concavity Worst", value=0.7119, format="%.4f")
    with col3:
        features['concave points_worst'] = st.number_input("Concave Points Worst", value=0.2654, format="%.4f")
        features['symmetry_worst'] = st.number_input("Symmetry Worst", value=0.4601, format="%.4f")
        features['fractal_dimension_worst'] = st.number_input("Fractal Dim Worst", value=0.1189, format="%.4f")


st.markdown("---")

# Prediction logic
if st.button("🔮 Predict Diagnosis"):
    with st.spinner('Analyzing cell nuclei data through Neural Network...'):
        time.sleep(1.5) # Micro-animation feel
        
        # Format the input
        input_data = [
            features['radius_mean'], features['texture_mean'], features['perimeter_mean'], features['area_mean'], features['smoothness_mean'], features['compactness_mean'], features['concavity_mean'], features['concave points_mean'], features['symmetry_mean'], features['fractal_dimension_mean'],
            features['radius_se'], features['texture_se'], features['perimeter_se'], features['area_se'], features['smoothness_se'], features['compactness_se'], features['concavity_se'], features['concave points_se'], features['symmetry_se'], features['fractal_dimension_se'],
            features['radius_worst'], features['texture_worst'], features['perimeter_worst'], features['area_worst'], features['smoothness_worst'], features['compactness_worst'], features['concavity_worst'], features['concave points_worst'], features['symmetry_worst'], features['fractal_dimension_worst']
        ]
        
        # Convert to numpy array and shape properly
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Scale the data
        std_data = scaler.transform(input_data_reshaped)
        
        # Predict
        prediction_prob = model.predict(std_data)[0][0]
        prediction = 1 if prediction_prob >= 0.5 else 0

        st.markdown("---")
        if prediction == 1:
            st.error("### ⚠️ Diagnosis: Malignant")
            st.markdown(f"<h4 style='text-align: center; color: #F63366;'>Confidence: {prediction_prob*100:.2f}%</h4>", unsafe_allow_html=True)
            st.info("The neural network predicts that the breast mass is **Malignant**. Please consult with a medical professional immediately.")
        else:
            st.success("### ✅ Diagnosis: Benign")
            st.markdown(f"<h4 style='text-align: center; color: #00CC96;'>Confidence: {(1-prediction_prob)*100:.2f}%</h4>", unsafe_allow_html=True)
            st.info("The neural network predicts that the breast mass is **Benign**. Continue regular check-ups.")
            st.balloons()
