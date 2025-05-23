import os
import pandas as pd
import requests
import streamlit as st
from modules.nav import Navbar
from dotenv import load_dotenv

st.set_page_config(page_title='Feedback Dashboard')
Navbar()

st.title('Feedbacks')

load_dotenv()
APP_API_KEY = os.getenv('APP_API_KEY')
APP_API_URL = os.getenv('APP_API_URL')

# @st.cache_data(ttl=300)
def get_feedback():
    headers ={'x-api-key': APP_API_KEY}
    response = requests.get(
        f'{APP_API_URL}/get_feedback_data',
        headers=headers
    )
    if response.status_code != 200:
        st.error(f'API Error: {response.status_code}- {response.json()["detail"]}')
        return None
    return response.json()['result']

feedbacks = pd.DataFrame(get_feedback())
feedbacks['input'] = feedbacks['input_json'].apply(eval)
feedbacks = pd.concat([feedbacks.drop('input_json', axis=1), feedbacks['input'].apply(pd.Series)], axis=1)
feedbacks.drop(columns='input', inplace=True)
if feedbacks.empty:
    st.warning("No predictions with feedback found.")
    st.stop()

# showing tables
st.subheader(f'Showing {len(feedbacks)} records.')
st.dataframe(feedbacks)

# Chart: Feedback count
st.subheader("Feedback Summary")
counts = feedbacks["correct"].value_counts().rename({True: "Correct", False: "Incorrect"})
st.bar_chart(counts)
