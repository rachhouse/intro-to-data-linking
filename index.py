import streamlit as st

import pages.blocking
import pages.cleaning
import pages.overview
import pages.problem_statement

VALID_PAGE_PARAMS = ["intro", "cleaning", "blocking", "problem_statement"]

st.set_page_config(
    page_title="Intro to Data Linking in Python", initial_sidebar_state="collapsed"
)

# The starting page can be passed in as a query param on the
# streamlit URL, e.g.
#   http://url-to-streamlit-app?page=blocking
query_params = st.experimental_get_query_params()

page_query_param = None

if ("page" in query_params.keys()) and (
    query_params["page"][0].lower() in VALID_PAGE_PARAMS
):
    page_query_param = query_params["page"][0].title()
    page_query_param = page_query_param.replace("_", " ")

# Sidebar: add a title and radio button toggle for page selection.
st.sidebar.title("An Introduction to Data Linking in Python")
st.sidebar.markdown("Rachel House | December 2020")

page_names = ["Overview", "Problem Statement", "Cleaning", "Blocking"]

pages = {
    "Overview": pages.overview,
    "Problem Statement": pages.problem_statement,
    "Cleaning": pages.cleaning,
    "Blocking": pages.blocking,
}

# Set the radio button to the page supplied by the query param
# (if it exists). Otherwise, just load the Intro page.
if page_query_param is not None:
    page_selection = st.sidebar.radio(
        "", page_names, index=page_names.index(page_query_param)
    )
else:
    page_selection = st.sidebar.radio("", page_names)

page = pages[page_selection]

with st.spinner(f"Loading {page_selection}..."):
    page.main()
