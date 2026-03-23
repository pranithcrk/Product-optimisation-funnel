"""Page 5: Weekly cohort retention curves and LTV estimates."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from data_loader import query_duckdb, load_funnel_events


def render():
    st.title("Cohort & Retention")
    st.markdown("Weekly cohort retention and lifetime value analysis by acquisition period.")

    events = load_funnel_events()
    events["event_time"] = pd.to_datetime(events["event_time"])

    # --- Build cohorts ---
    user_first = events.groupby("user_id")["event_time"].min().reset_index()
    user_first.columns = ["user_id", "first_seen"]
    user_first["cohort_week"] = user_first["first_seen"].dt.to_period("W").dt.start_time

    events = events.merge(user_first[["user_id", "cohort_week"]], on="user_id")
    events["activity_week"] = events["event_time"].dt.to_period("W").dt.start_time
    events["weeks_since"] = ((events["activity_week"] - events["cohort_week"]).dt.days / 7).astype(int)

    # --- Retention matrix ---
    retention = events.groupby(["cohort_week", "weeks_since"])["user_id"].nunique().reset_index()
    retention.columns = ["cohort_week", "weeks_since", "users"]

    cohort_sizes = retention[retention["weeks_since"] == 0][["cohort_week", "users"]].rename(columns={"users": "cohort_size"})
    retention = retention.merge(cohort_sizes, on="cohort_week")
    retention["retention_pct"] = (retention["users"] / retention["cohort_size"] * 100).round(1)

    # Pivot for heatmap
    max_weeks = min(12, retention["weeks_since"].max())
    ret_pivot = retention[retention["weeks_since"] <= max_weeks].pivot(
        index="cohort_week", columns="weeks_since", values="retention_pct"
    ).fillna(0)
    ret_pivot.index = ret_pivot.index.strftime("%Y-%m-%d")

    st.subheader("Retention Heatmap (% of cohort active)")
    fig = px.imshow(
        ret_pivot, labels=dict(x="Weeks Since First Visit", y="Cohort Week", color="Retention %"),
        color_continuous_scale="YlGnBu", aspect="auto",
        text_auto=".0f",
    )
    fig.update_layout(height=max(300, len(ret_pivot) * 35), margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- Retention curves by cohort ---
    st.subheader("Retention Curves")
    cohorts = sorted(retention["cohort_week"].unique())
    # Show last 6 cohorts for clarity
    show_cohorts = cohorts[-6:] if len(cohorts) > 6 else cohorts

    fig2 = go.Figure()
    for cw in show_cohorts:
        cdata = retention[(retention["cohort_week"] == cw) & (retention["weeks_since"] <= max_weeks)]
        fig2.add_trace(go.Scatter(
            x=cdata["weeks_since"], y=cdata["retention_pct"],
            mode="lines+markers", name=pd.Timestamp(cw).strftime("%Y-%m-%d"),
        ))
    fig2.update_layout(
        xaxis_title="Weeks Since First Visit",
        yaxis_title="Retention %",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- LTV by cohort ---
    st.subheader("Revenue per User by Cohort")
    purchase_events = events[events["event_type"] == "purchase"]
    cohort_revenue = purchase_events.groupby("cohort_week").agg(
        total_revenue=("price", "sum"),
        purchasers=("user_id", "nunique"),
    ).reset_index()
    cohort_revenue = cohort_revenue.merge(cohort_sizes, on="cohort_week")
    cohort_revenue["revenue_per_user"] = (cohort_revenue["total_revenue"] / cohort_revenue["cohort_size"]).round(2)
    cohort_revenue["purchase_rate"] = (cohort_revenue["purchasers"] / cohort_revenue["cohort_size"] * 100).round(1)
    cohort_revenue["cohort_label"] = cohort_revenue["cohort_week"].dt.strftime("%Y-%m-%d")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=cohort_revenue["cohort_label"], y=cohort_revenue["revenue_per_user"],
        name="Rev/User ($)", marker_color="#636EFA",
    ))
    fig3.add_trace(go.Scatter(
        x=cohort_revenue["cohort_label"], y=cohort_revenue["purchase_rate"],
        name="Purchase Rate %", yaxis="y2", mode="lines+markers", marker_color="#EF553B",
    ))
    fig3.update_layout(
        barmode="group", height=400,
        yaxis=dict(title="Revenue per User ($)"),
        yaxis2=dict(title="Purchase Rate %", overlaying="y", side="right"),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --- Summary stats ---
    st.subheader("Cohort Summary")
    summary = cohort_revenue[["cohort_label", "cohort_size", "purchasers", "purchase_rate", "total_revenue", "revenue_per_user"]].copy()
    summary.columns = ["Cohort", "Users", "Purchasers", "Purchase Rate %", "Total Revenue $", "Rev/User $"]
    st.dataframe(summary, use_container_width=True, hide_index=True)
