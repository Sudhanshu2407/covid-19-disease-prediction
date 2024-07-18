# Import the important libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from PIL import Image
import pyttsx3

# Configure the page (must be the first Streamlit command)
st.set_page_config(
    page_title="COVID-19 Disease Prediction App",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ignore the warnings
warnings.filterwarnings("ignore")

# Load the saved model
model=pickle.load(open(r"C:\sudhanshu_projects\project-task-training-course\covid-19_prediction\covid_19_prediction.pkl","rb"))

# Here we add one image.
image = Image.open(r"C:\sudhanshu_projects\project-task-training-course\covid-19_prediction\covid_19_disease_prediction_app_image.jpeg")

st.image(image=image, caption="COVID-19 Disease Prediction")

# Your Streamlit code here
st.title("COVID-19 Disease Prediction App")
st.write("Predicting the likelihood of COVID-19 infection based on symptoms and other features.")

st.subheader("Enter the values of below parameters: ")

cough = st.select_slider("Cough", [0, 1])
fever = st.select_slider("Fever", [0, 1])
sore_throat = st.select_slider("Sore Throat", [0, 1])
shortness_of_breath = st.select_slider("Shortness of Breath", [0, 1])
head_ache = st.select_slider("Headache", [0, 1])
age_60_and_above = st.select_slider("Age 60 and Above", [0, 1])
gender = st.select_slider("Gender", [0, 1])
test_indication = st.select_slider("Test Indication", [0, 1, 2])

input_field = [[cough, fever, sore_throat, shortness_of_breath, head_ache, age_60_and_above, gender, test_indication]]

engine = pyttsx3.init()

if st.button("Predict"):
    prediction = model.predict(input_field)[0]
    if prediction == 0:
        st.write("The person is not diagnosed with COVID-19.")
        engine.say("You are not diagnosed with COVID-19. Enjoy your life and follow all precautions.")
    else:
        st.write("The person is diagnosed with COVID-19.")
        tips = (
            "Please isolate yourself from others to prevent the spread of the virus. "
            "Stay hydrated and get plenty of rest. Monitor your symptoms and seek medical attention if they worsen. "
            "Follow the guidelines provided by health authorities."
        )
        st.write(tips)
        engine.say("The person is diagnosed with COVID-19.")
        engine.say(tips)
    
    engine.runAndWait()

st.write("Thank you for using our app. Hope you get the correct predictions.")

st.write("© Copyrights are licensed to sudhanshu sharma.")