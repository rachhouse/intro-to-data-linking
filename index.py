import streamlit as st

st.set_page_config(page_title="Intro to Data Linking in Python")

st.title("An Introduction to Data Linking in Python")

with st.echo():
    import datetime
    import dateparser
    import pandas as pd
    import re


with st.echo():
    df_jvc = pd.read_csv("data/jvc_raw.csv")

st.write(df_jvc.head())

with st.echo():
    df_vandal = pd.read_csv("data/vandal_raw.csv")

st.write(df_vandal.head())
