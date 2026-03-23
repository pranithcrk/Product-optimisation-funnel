"""Page 1: Interactive funnel chart with conversion rates and filters."""

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from data_loader import load_funnel_events, load_session_summary


def render():
    st.title("Funnel Overview")
    st.markdown("Conversion rates at each stage of the purchase funnel.")

    events = load_funnel_events()
    events["event_date"] = pd.to_datetime(events["event_time"]).dt.date

    # --- Filters ---
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.date_input(
            "Date range",
            value=(events["event_date"].min(), events["event_date"].max()),
            min_value=events["event_date"].min(),
            max_value=events["event_date"].max(),
        )
    with col2:
        brands = ["All"] + sorted(events["brand"].dropna().unique().tolist())
        selected_brand = st.selectbox("Brand", brands)

    # Apply filters
    if len(date_range) == 2:
        mask = (events["event_date"] >= date_range[0]) & (events["event_date"] <= date_range[1])
        events = events[mask]
    if selected_brand != "All":
        events = events[events["brand"] == selected_brand]

    # --- Funnel metrics ---
    stages = ["view", "cart", "purchase"]
    stage_labels = ["Browse / View", "Add to Cart", "Purchase"]
    stage_sessions = []
    for s in stages:
        n = events[events["event_type"] == s]["computed_session_id"].nunique()
        stage_sessions.append(n)

    total = stage_sessions[0] if stage_sessions[0] > 0 else 1

    # KPI row
    cols = st.columns(3)
    for i, (label, count) in enumerate(zip(stage_labels, stage_sessions)):
        with cols[i]:
            pct = count / total * 100
            st.metric(label, f"{count:,}", f"{pct:.1f}% of viewers")

    # --- Funnel chart ---
    fig = go.Figure(go.Funnel(
        y=stage_labels,
        x=stage_sessions,
        textinfo="value+percent initial",
        marker=dict(color=["#636EFA", "#EF553B", "#00CC96"]),
    ))
    fig.update_layout(
        title="Session Funnel",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Stage-to-stage conversion table ---
    st.subheader("Stage-to-Stage Conversion")
    rows = []
    for i in range(len(stages) - 1):
        conv = stage_sessions[i + 1] / stage_sessions[i] * 100 if stage_sessions[i] > 0 else 0
        rows.append({
            "From": stage_labels[i],
            "To": stage_labels[i + 1],
            "From Sessions": stage_sessions[i],
            "To Sessions": stage_sessions[i + 1],
            "Conversion %": f"{conv:.1f}%",
            "Drop-off %": f"{100 - conv:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Daily conversion trend ---
    st.subheader("Daily Conversion Trend")
    daily = events.groupby(["event_date", "event_type"])["computed_session_id"].nunique().reset_index()
    daily.columns = ["date", "event_type", "sessions"]

    import plotly.express as px
    fig2 = px.line(
        daily, x="date", y="sessions", color="event_type",
        color_discrete_map={"view": "#636EFA", "cart": "#EF553B", "purchase": "#00CC96"},
        labels={"sessions": "Unique Sessions", "date": "Date"},
    )
    fig2.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig2, use_container_width=True)
