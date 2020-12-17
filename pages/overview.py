import streamlit as st

def main():
    st.title("Overview")

    st.markdown("""Welcome to this introduction to data linking in Python! I developed this streamlit app to provide supporting examples for a related conference talk, but it can also serve as standalone walkthrough of a sample data linking process. I recommend that you check out the talk slides for more background and context on data linking:

[https://slides.com/rachhouse/intro-to-data-linking-in-python](https://slides.com/rachhouse/intro-to-data-linking-in-python)""")

    st.markdown("""The tutorial follows the basic flow of data linking steps:
* Pre-processing (Cleaning)
* Indexing (Blocking)
* Comparing

Using the sidebar to the left (click on the small carat icon in the upper left to expand the sidebar if it is currently collapsed), you can navigate to your page(s) of interest.""")

    st.markdown("Though this code is currently at a good \"stopping point\", there is still more I would like to add, including code for the Classification and Evaluation steps of the data linking process. Please check back in the future for additional material.")

    st.markdown("Thanks, and I hope you find this material useful and enjoyable.")