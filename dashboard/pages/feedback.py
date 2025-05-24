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
        st.error(f'API Error: {response.status_code}')
        return None
    return response.json()['result']

# @st.cache_data(ttl=300)
def get_feedback_summary():
    headers ={'x-api-key': APP_API_KEY}
    response = requests.get(
        f'{APP_API_URL}/feedback_summary',
        headers=headers
    )
    if response.status_code != 200:
        st.error(f'API Error: {response.status_code}')
        return None
    return response.json()['result']

feedbacks = pd.DataFrame(get_feedback())
feedback_summary = get_feedback_summary()

feedbacks['input'] = feedbacks['input_json'].apply(eval)
feedbacks = pd.concat([feedbacks.drop('input_json', axis=1), feedbacks['input'].apply(pd.Series)], axis=1)
feedbacks.drop(columns='input', inplace=True)
if feedbacks.empty:
    st.warning("No predictions with feedback found.")
    st.stop()

summary = get_feedback_summary()
if summary:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Feedbacks", summary["total"])
    col2.metric("Correct", summary["correct"])
    col3.metric("Incorrect", summary["incorrect"])
    col4.metric("Accuracy (%)", summary["accuracy"])

# showing tables
st.subheader(f'Showing {len(feedbacks)} records.')
model_versions = feedbacks['model_version'].unique().tolist()
selected_version = st.selectbox('Filter by Model Version', ["All"] + model_versions)
if selected_version != 'All':
    feedbacks = feedbacks[feedbacks['model_version'] == selected_version]
st.dataframe(feedbacks)
# export to csv

csv_feedback = feedbacks.drop(columns=['id','timestamp','model_version']).to_csv(index=False).encode('utf-8')
st.download_button(
    label='Download feedbacks as CSV',
    data=csv_feedback,
    file_name='feedback_data.csv',
    mime='text/csv'
)
# Chart: Feedback count
st.subheader("Feedback Summary")
counts = feedbacks["correct"].value_counts().rename({True: "Correct", False: "Incorrect"})
st.bar_chart(counts)
