import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import numpy as np

print(f"Pickle Module Version: {pickle.format_version}") 

# Load all models
with open('decision_tree_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('gradient_boosting_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

# Function to predict diabetes
def predict_diabetes(model, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return int(prediction[0])

# Streamlit app
st.title("Diabetes Prediction App")
st.write("Enter your health indicators and select a model to predict diabetes.")

# Input fields for user data
st.sidebar.header("User Input Features")
high_bp = st.sidebar.selectbox("High Blood Pressure", [0, 1])
high_chol = st.sidebar.selectbox("High Cholesterol", [0, 1])
chol_check = st.sidebar.selectbox("Cholesterol Check in Last 5 Years", [0, 1])
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
smoker = st.sidebar.selectbox("Smoker", [0, 1])
stroke = st.sidebar.selectbox("History of Stroke", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease or Attack", [0, 1])
phys_activity = st.sidebar.selectbox("Physical Activity", [0, 1])
fruits = 1
veggies = 1
hvy_alcohol_consump = st.sidebar.selectbox("Heavy Alcohol Consumption", [0, 1])
any_healthcare = 1
no_doc_bc_cost = st.sidebar.selectbox("No Doctor Due to Cost", [0, 1])
gen_hlth = st.sidebar.selectbox("General Health (1=Excellent, 5=Poor)", [1, 2, 3, 4, 5])
ment_hlth = st.sidebar.number_input("Mental Health (Days in Last 30 Days)", min_value=0, max_value=30, value=0)
phys_hlth = st.sidebar.number_input("Physical Health (Days in Last 30 Days)", min_value=0, max_value=30, value=0)
sex = st.sidebar.selectbox("Sex (0=Female, 1=Male)", [0, 1])
age = st.sidebar.selectbox("Age Group (1=18-24, 13=80+)", list(range(1, 14)))

# Create a dictionary of user inputs
user_input = {
    'HighBP': high_bp,
    'HighChol': high_chol,
    'CholCheck': chol_check,
    'BMI': bmi,
    'Smoker': smoker,
    'Stroke': stroke,
    'HeartDiseaseorAttack': heart_disease,
    'PhysActivity': phys_activity,
    'Fruits': fruits,
    'Veggies': veggies,
    'HvyAlcoholConsump': hvy_alcohol_consump,
    'AnyHealthcare': any_healthcare,
    'NoDocbcCost': no_doc_bc_cost,
    'GenHlth': gen_hlth,
    'MentHlth': ment_hlth,
    'PhysHlth': phys_hlth,
    'Sex': sex,
    'Age': age,
}

# Display user input
st.subheader("User Input Features")
st.write(pd.DataFrame([user_input]))

# Buttons for each model
st.subheader("Select a Model for Prediction")
def display_prediction(model, model_name):
    prediction = predict_diabetes(model, user_input)
    prediction_text = ["No Diabetes", "Pre-Diabetes", "Diabetes"][prediction]
    color = ["green", "orange", "red"][prediction]
    message = ["You are good to go", "You are OK but need to eat some veggies that reduce diabetes", "You have Diabetes. You need to consult a doctor and follow medical advice"]
    st.markdown(f"**{model_name} Prediction:** <span style='color:{color}; font-weight:bold;'>{prediction_text}</span>", unsafe_allow_html=True)
    st.write(message[prediction])

if st.button("Decision Tree"):
    display_prediction(dt_model, "Decision Tree")

if st.button("Random Forest"):
    display_prediction(rf_model, "Random Forest")

if st.button("Gradient Boosting"):
    display_prediction(gb_model, "Gradient Boosting")

if st.button("Logistic Regression"):
    display_prediction(lr_model, "Logistic Regression")

if st.button("Naive Bayes"):
    display_prediction(nb_model, "Naive Bayes")
