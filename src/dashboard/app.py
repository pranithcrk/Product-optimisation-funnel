"""E-Commerce Conversion Funnel Intelligence Platform — Streamlit Dashboard.

Run with: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# Add the dashboard directory to sys.path so page modules can import data_loader
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(
    page_title="Funnel Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pages import funnel_overview, dropoff_analysis, conversion_prediction, ab_test_calculator, cohort_retention

PAGES = {
    "Funnel Overview": funnel_overview,
    "Drop-off Analysis": dropoff_analysis,
    "Conversion Prediction": conversion_prediction,
    "A/B Test Calculator": ab_test_calculator,
    "Cohort & Retention": cohort_retention,
}

st.sidebar.title("Funnel Intelligence")
st.sidebar.markdown("---")
selection = st.sidebar.radio("Navigate", list(PAGES.keys()))

PAGES[selection].render()
