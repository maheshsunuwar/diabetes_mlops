import requests
import streamlit as st
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

from modules.nav import Navbar

st.set_page_config(page_title="Diabetes Predictor Dashboard")

load_dotenv(override=True)
APP_API_KEY = os.getenv('APP_API_KEY')
APP_API_URL = os.environ['APP_API_URL']


# @st.cache_data(ttl=300)
def get_data():
    headers = {'x-api-key': APP_API_KEY}
    response = requests.get(
        f'{APP_API_URL}/get_data',
        headers=headers
    )
    df = pd.DataFrame(response.json()['result'])
    df['input'] = df['input_json'].apply(eval)
    df = pd.concat([df.drop('input_json', axis=1), df['input'].apply(pd.Series)], axis=1)
    return df

Navbar()

st.title("Diabetes Prediction Dashboard")
df = get_data()

st.subheader("Latest Predictions")
st.dataframe(df, use_container_width=True)

st.markdown("---")

model_version = st.sidebar.selectbox('Model Version', df['model_version'].unique())
filtered_df = df[df['model_version'] == model_version]

st.subheader('Predictions over time')
st.line_chart(filtered_df.set_index('timestamp')['prediction'])

feature = st.selectbox('Select Feature', ['age', 'bmi', 'bp', 's1', 's2'])
st.subheader(f'Distibution of {feature}')
st.bar_chart(filtered_df[feature].value_counts().sort_index())

outliers = filtered_df[filtered_df['prediction'] > 300]
if not outliers.empty:
    st.warning('Outlier predictions detected, High risk.')
    st.dataframe(outliers[['timestamp', 'prediction', 'model_version']])
else:
    st.success('No Outliers found.')

# st.metric("Last Prediction", round(df['prediction'].iloc[0], 2))
# st.metric("Model Version", df['model_version'].iloc[0])
# st.line_chart(df['prediction'])
