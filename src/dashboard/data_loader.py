"""Shared data loading utilities for all dashboard pages."""

import json
import pickle
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "funnel.duckdb"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data(ttl=600)
def query_duckdb(sql):
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(sql).fetchdf()
    con.close()
    return df


@st.cache_data(ttl=600)
def load_session_summary():
    return pd.read_parquet(DATA_DIR / "session_funnel_summary.parquet")


@st.cache_data(ttl=600)
def load_ml_features():
    return pd.read_parquet(DATA_DIR / "ml_features.parquet")


@st.cache_data(ttl=600)
def load_funnel_events():
    return pd.read_parquet(DATA_DIR / "funnel_events.parquet")


@st.cache_resource
def load_model():
    with open(MODEL_DIR / "gb_conversion.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=600)
def load_metrics():
    with open(MODEL_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data(ttl=600)
def load_shap_importance():
    return pd.read_csv(MODEL_DIR / "shap_importance.csv")


@st.cache_data(ttl=600)
def load_feature_cols():
    with open(MODEL_DIR / "feature_cols.json") as f:
        return json.load(f)
