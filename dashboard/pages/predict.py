import os
import requests
import streamlit as st
from modules.nav import Navbar
from dotenv import load_dotenv
from uuid import UUID
st.set_page_config(page_title="Predict")

load_dotenv(override=True)
APP_API_KEY = os.getenv('APP_API_KEY')
APP_API_URL = os.environ['APP_API_URL']

def make_prediction(input_data: dict):
    headers = {'x-api-key': APP_API_KEY}
    response = requests.post(
        f'{APP_API_URL}/predict',
        json=input_data,
        headers=headers
    )
    if response.status_code != 200:
        st.error(f'API Error: {response.status_code}- {response.json()["detail"]}')
        return None
    return response.json()

def log_feedback(feedback_data:dict):
    headers = {'x-api-key': APP_API_KEY}
    response = requests.post(
        f'{APP_API_URL}/log_feedback',
        json=feedback_data,
        headers=headers
    )
    return response

Navbar()

with st.form('predict_form', clear_on_submit=False, border=True):
    st.subheader("Enter Patient Features")
    age = st.number_input("Age", format="%f")
    sex = st.number_input("Sex", format="%.3f")
    bmi = st.number_input("BMI", format="%.3f")
    bp = st.number_input("Blood Pressure", format="%.3f")
    s1 = st.number_input("S1", format="%.3f")
    s2 = st.number_input("S2", format="%.3f")
    s3 = st.number_input("S3", format="%.3f")
    s4 = st.number_input("S4", format="%.3f")
    s5 = st.number_input("S5", format="%.3f")
    s6 = st.number_input("S6", format="%.3f")

    submitted = st.form_submit_button('Predict')

# Initialize session state
if 'prediction_return' not in st.session_state:
    st.session_state.prediction_return = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

if submitted:

    input_data = {
        "age": age, "sex": sex, "bmi": bmi, "bp": bp,
        "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6
    }

    prediction_return = make_prediction(input_data)
    st.session_state.prediction_return = prediction_return
    st.session_state.feedback_submitted = False

if 'prediction_return' in st.session_state and st.session_state.prediction_return:
    prediction_return = st.session_state.get('prediction_return')
    st.success(f'Prediction: {round(prediction_return["prediction"], 2)}')

    if not st.session_state.get('feedback_submitted', False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Correct'):
                response = log_feedback(feedback_data={
                    'correct': True,
                    'id': prediction_return['id']
                })
                st.session_state.feedback_submitted=True
                st.success(f'Feedback logged as correct.')
        with col2:
            if st.button('Incorrect'):
                response = log_feedback(feedback_data={
                    'correct': False,
                    'id': prediction_return['id']
                })
                st.session_state.feedback_submitted=True
                st.success(f'Feedback logged as incorrect.')
    else:
        st.info("Feedback already submitted.")
