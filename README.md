# ✍️ Signature Recognition System

A Machine Learning based Signature Recognition System that classifies handwritten signatures as **Genuine** or **Forged** using behavioral and statistical features extracted from online signature data.

The project compares different approaches and uses a **Random Forest classifier** as the final model.

---

## 🚀 Live Demo

The trained model is deployed using Streamlit.

🔗 **Streamlit App:**  
PASTE_YOUR_STREAMLIT_LINK_HERE

---

## 📌 Project Overview

Signature verification is an important problem in areas such as banking, document verification, authentication, and fraud detection.

This project analyzes online signature data containing features such as:

- X and Y coordinates
- Timestamp
- Pen status
- Azimuth
- Altitude
- Pressure

Statistical features are extracted from each signature and used to train Machine Learning models for classification.

---

## 🎯 Objective

The main objective is to build a system that can distinguish between:

- ✅ Genuine signatures
- ⚠️ Forged signatures

The final model is designed to provide both a predicted class and prediction confidence.

---

## 📊 Dataset

The dataset contains online signature sequences from **40 users**.

### Dataset distribution

| Class | Samples |
|---|---:|
| Genuine | 800 |
| Forged | 800 |
| **Total** | **1600** |

Each user contains:

- 20 Genuine signatures
- 20 Forged signatures

The dataset contains **7 raw features** per timestep:

| Feature | Description |
|---|---|
| `x` | X-coordinate |
| `y` | Y-coordinate |
| `timestamp` | Time information |
| `pen_status` | Pen movement/status |
| `azimuth` | Pen orientation |
| `altitude` | Pen altitude |
| `pressure` | Pen pressure |

---

## ⚙️ Data Preprocessing

The signature sequences are processed before model training.

Main preprocessing steps:

1. Load signature files
2. Convert timestamp into elapsed time
3. Handle variable-length sequences
4. Limit sequences to 300 timesteps
5. Extract statistical features
6. Prepare training and testing data

---

## 🧮 Feature Engineering

For every signature, statistical features are extracted from the seven original features.

The final feature vector contains:

- Mean of each feature
- Standard deviation of each feature

This produces **14 features per signature**.

### Extracted Features

```text
mean_x
mean_y
mean_timestamp
mean_pen_status
mean_azimuth
mean_altitude
mean_pressure

std_x
std_y
std_timestamp
std_pen_status
std_azimuth
std_altitude
std_pressure

🤖 Models

Different approaches were explored during development, including sequence-based and classical Machine Learning approaches.

The final selected model is:

🌲 Random Forest Classifier

The Random Forest model performed better on the engineered statistical features than the initial LSTM approach.

📈 Final Model Performance
Accuracy: 84.69%
Confusion Matrix
[[124  36]
 [ 13 147]]
Classification Report
Class	Precision	Recall	F1-score
Genuine	0.91	0.78	0.84
Forged	0.80	0.92	0.86
Overall	0.85	0.85	0.85

The model achieved a good balance between Genuine and Forged signature detection.

🔍 Feature Importance

The most important features identified by the Random Forest model include:

Feature	Importance
mean_pressure	0.1276
std_timestamp	0.1184
mean_timestamp	0.1008
mean_azimuth	0.0775
std_altitude	0.0736
std_pen_status	0.0722
mean_pen_status	0.0669

This shows that pressure and timing-related characteristics contribute significantly to distinguishing genuine and forged signatures.

🧪 Prediction Example

The application accepts a .TXT signature file and returns a prediction.

Example:

Testing file: U10S1.TXT
Prediction: Genuine
Confidence: 90.33%

Another example:

Testing file: U10S21.TXT
Prediction: Forged
Confidence: 94.33%
🖥️ Streamlit Application

The project includes an interactive Streamlit interface where users can:

Upload a signature .TXT file
Process the signature
Extract statistical features
Run the trained Random Forest model
View Genuine/Forged prediction
View prediction confidence
📁 Project Structure
signature-recognition/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── signature_random_forest.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   └── 02_Preprocessing.ipynb
│
└── src/
    ├── preprocessing.py
    └── feature_engineering.py
🛠️ Technologies Used
Python
NumPy
Pandas
Scikit-learn
PyTorch
Streamlit
Matplotlib
Jupyter Notebook
Joblib
📦 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/signature-recognition.git

Move into the project directory:

cd signature-recognition

Install the required dependencies:

pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

📓 Notebooks

The project contains two main notebooks:

01_Data_Exploration.ipynb

Contains:

Dataset exploration
Class distribution
User distribution
Sequence length analysis
Feature analysis
02_Preprocessing.ipynb

Contains:

Signature preprocessing
Sequence handling
Feature engineering
Model training
Model evaluation
Feature importance
Prediction testing
🔐 Data Privacy

The original signature dataset is not included in this public repository.

Only the project code, notebooks, trained model, and supporting files are included.  


## 🚀 Live Demo: https://signature-recognition-bwvyn4xtbb22wwfuzsdni4.streamlit.app/

🔮 Future Improvements

Potential improvements include:

CNN-based signature representation
Improved LSTM/GRU architectures
Siamese Neural Networks
Signature image-based verification
Data augmentation
Hyperparameter optimization
Larger and more diverse datasets
Improved user-specific verification
Explainable AI for signature verification
👩‍💻 Author

Shashi Agrawal

B.Tech CSE | Machine Learning & Deep Learning Enthusiast

⭐ If you find this project useful, consider giving the repository a star!
