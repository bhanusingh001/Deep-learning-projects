# 🧬 Breast Cancer Prediction System

> **Predict the onset of Malignant Breast Cancer based on cell nuclei measurements using Deep Learning**

[**🚀 View Live App**](https://your-breast-cancer-app-link.streamlit.app/)

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📋 Overview

This project uses an advanced **Deep Neural Network** to predict whether a breast mass is Malignant or Benign based on 30 biological features computed from a digitized image of a fine needle aspirate (FNA). 

The neural network is built with a Sequential Keras architecture (64 -> 32 -> 1) capturing non-linear relationships across **Mean, Standard Error, and Worst** biological metrics to ensure top-tier diagnostic accuracy.

The project is presented through a **premium, sleek Streamlit web app** featuring a modern neon-dark theme configuration (`config.toml`), designed to elegantly separate complex 30-feature inputs into intuitive tabs for medical professionals.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Deep Learning Model** | High-accuracy Sequential Neural Network built with TensorFlow/Keras |
| 🌐 **Web App** | Premium neon-dark theme with micro-animations & clear visual hierarchy |
| 🗂️ **Organized Inputs** | 30 features cleanly segmented into Mean, SE, and Worst metric tabs |
| 🛡️ **Risk Confidence** | Outputs the statistical probability of malignancy alongside the diagnosis |
| ⚖️ **Data Scaling** | Built-in custom `StandardScaler` to handle extreme biological variances |
|📱 **Responsive** | Works flawlessly on desktop and mobile browsers |
| 🚀 **Deploy Ready** | One-click deployment to Streamlit Cloud |

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Deep Learning:** TensorFlow & Keras (Sequential API, Dense Layers, Adam Optimizer)
- **Web Framework:** Streamlit
- **Data Processing:** Scikit-learn (StandardScaler), NumPy, Pandas
- **Deployment:** Streamlit Cloud

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |
| **Samples** | 569 instances (357 Benign + 212 Malignant) |
| **Features** | 30 numerical variables (Radius, Texture, Perimeter, Area, Smoothness, etc.) |
| **Target** | 1 (Malignant) or 0 (Benign) |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Architecture** | 3-Layer Dense Neural Network |
| **Optimizer** | Adam (binary_crossentropy loss) |
| **Test Accuracy** | ~97.37% |
| **Test Split** | 20% |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/bhanusingh001/Machine-Learning-Projects.git
cd "dl projects"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Generate .h5 and .pkl artifacts)

```bash
python train_model.py
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo, branch (`main`), and file (`app.py`)
5. Click **"Deploy"** — your app will be live in minutes! 🚀

---

## 🔍 Input Format

The model expects **30 numerical values** formatted across three distinct biological categories. 

### Important Parameters:
1. **Mean Features** (10 inputs): e.g., Radius Mean, Texture Mean, Perimeter Mean...
2. **Standard Error (SE)** (10 inputs): e.g., Radius SE, Texture SE, Perimeter SE...
3. **Worst (Largest)** (10 inputs): e.g., Radius Worst, Texture Worst, Perimeter Worst...

### Example Malignant Patient Input (Comma-separated array values):
```
17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
```

---

## 📁 Project Structure

```text
dl projects/
├── .streamlit/             
│   └── config.toml         # Custom neon-dark theme Streamlit settings
├── app.py                  # Streamlit web application
├── train_model.py          # Script to train the Deep Learning model
├── breast_cancer_model.h5  # Trained Neural Network model artifact
├── scaler.pkl              # Feature scaler (StandardScaler) artifact
├── breast cancer dataset.csv # Dataset containing mass records
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🤝 Credits

- **Dataset:** Dr. William H. Wolberg, General Surgery Dept., University of Wisconsin
- **Built with:** [TensorFlow](https://tensorflow.org), [Streamlit](https://streamlit.io), [Scikit-learn](https://scikit-learn.org)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ 
  <br>
  ⭐ Star this repo if you found it helpful!
</p>
