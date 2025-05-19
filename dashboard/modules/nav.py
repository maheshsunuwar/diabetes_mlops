import streamlit as st

def Navbar():
    with st.sidebar:
        st.page_link('app.py', label='Home')
        st.page_link('pages/predict.py', label='Make Predictions')
