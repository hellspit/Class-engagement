import streamlit as st
import numpy as np
import pickle

# Load model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

W = model['weights']
b = model['bias']

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X):
    z = np.dot(X, W) + b
    return 1 if sigmoid(z) > 0.5 else 0

st.title("Student Engagement Predictor")

# --- Input for numerical features ---
delta_psd = st.number_input("Delta PSD", min_value=0.0)
theta_psd = st.number_input("Theta PSD", min_value=0.0)
alpha_psd = st.number_input("Alpha PSD", min_value=0.0)
beta_psd = st.number_input("Beta PSD", min_value=0.0)
gamma_psd = st.number_input("Gamma PSD", min_value=0.0)
Pupil_Dilation = st.number_input("Pupil_Dilation", min_value=0.0)
Blink_Rate = st.number_input("Blink_Rate", min_value=0)	
Fixation_Duration = st.number_input("Fixation_Duration", min_value=0)
Saccade_Velocity = st.number_input("Saccade_Velocity", min_value=0.0)

# --- Input for categorical features ---
learning_type = st.selectbox("Learning Content Type", ['Video', 'Text', 'Interactive'])
difficulty = st.selectbox("Difficulty Level", ['Easy', 'Medium', 'Hard'])

# --- One-hot encoding manually (must match training order!) ---
# Replace this with the exact order from your training encoder
# Example: One-hot encode manually
learning_encoded = [0,0,0]
difficulty_encoded = [0,0,0]
if learning_type == 'Video':
    learning_encoded[2] = 1
elif learning_type == 'Text':   
    learning_encoded[1] = 1
elif learning_type == 'Interactive':    
    learning_encoded[0] = 1
if difficulty == 'Easy':
    difficulty_encoded[0] = 1
elif difficulty == 'Medium':
    difficulty_encoded[2] = 1
elif difficulty == 'Hard':
    difficulty_encoded[1] = 1       

# Combine all features
X_input = np.array([delta_psd, theta_psd, alpha_psd, beta_psd, gamma_psd,Pupil_Dilation,Blink_Rate,Fixation_Duration,Saccade_Velocity] + learning_encoded + difficulty_encoded)

if st.button("Predict Engagement"):
    result = predict(X_input)
    st.success(f"Predicted Engagement: {'Engaged' if result == 1 else 'Not Engaged'}")
