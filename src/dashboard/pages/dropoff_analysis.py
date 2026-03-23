"""Page 2: Drop-off analysis — where, when, and which segments drop off most."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_funnel_events, load_session_summary


def render():
    st.title("Drop-off Analysis")
    st.markdown("Understand where users drop off and identify patterns.")

    events = load_funnel_events()
    sessions = load_session_summary()

    # --- Drop-off by hour of day ---
    st.subheader("Conversion by Hour of Day")
    events["hour"] = pd.to_datetime(events["event_time"]).dt.hour

    hourly = events.groupby(["hour", "event_type"])["computed_session_id"].nunique().reset_index()
    hourly.columns = ["hour", "event_type", "sessions"]
    hourly_pivot = hourly.pivot(index="hour", columns="event_type", values="sessions").fillna(0)

    if "view" in hourly_pivot.columns and "purchase" in hourly_pivot.columns:
        hourly_pivot["conv_rate"] = hourly_pivot["purchase"] / hourly_pivot["view"] * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(x=hourly_pivot.index, y=hourly_pivot["view"], name="Views", marker_color="#636EFA"))
        fig.add_trace(go.Bar(x=hourly_pivot.index, y=hourly_pivot["purchase"], name="Purchases", marker_color="#00CC96"))
        fig.add_trace(go.Scatter(
            x=hourly_pivot.index, y=hourly_pivot["conv_rate"],
            name="Conv. Rate %", yaxis="y2", mode="lines+markers", marker_color="#EF553B",
        ))
        fig.update_layout(
            barmode="group", height=400,
            yaxis=dict(title="Sessions"),
            yaxis2=dict(title="Conversion Rate %", overlaying="y", side="right"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Drop-off by day of week ---
    st.subheader("Conversion by Day of Week")
    events["dow"] = pd.to_datetime(events["event_time"]).dt.dayofweek
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    dow_data = events.groupby(["dow", "event_type"])["computed_session_id"].nunique().reset_index()
    dow_data.columns = ["dow", "event_type", "sessions"]
    dow_pivot = dow_data.pivot(index="dow", columns="event_type", values="sessions").fillna(0)

    if "view" in dow_pivot.columns and "purchase" in dow_pivot.columns:
        dow_pivot["conv_rate"] = dow_pivot["purchase"] / dow_pivot["view"] * 100
        dow_pivot.index = [dow_labels[i] for i in dow_pivot.index]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=dow_pivot.index, y=dow_pivot["view"], name="Views", marker_color="#636EFA"))
        fig2.add_trace(go.Bar(x=dow_pivot.index, y=dow_pivot["purchase"], name="Purchases", marker_color="#00CC96"))
        fig2.add_trace(go.Scatter(
            x=dow_pivot.index, y=dow_pivot["conv_rate"],
            name="Conv. Rate %", yaxis="y2", mode="lines+markers", marker_color="#EF553B",
        ))
        fig2.update_layout(
            barmode="group", height=350,
            yaxis=dict(title="Sessions"),
            yaxis2=dict(title="Conversion Rate %", overlaying="y", side="right"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Heatmap: hour x day of week conversion rate ---
    st.subheader("Conversion Rate Heatmap (Hour x Day)")
    events["event_type_is_purchase"] = (events["event_type"] == "purchase").astype(int)
    events["event_type_is_view"] = (events["event_type"] == "view").astype(int)

    heat_data = events.groupby(["dow", "hour"]).agg(
        views=("event_type_is_view", "sum"),
        purchases=("event_type_is_purchase", "sum"),
    ).reset_index()
    heat_data["conv_rate"] = np.where(heat_data["views"] > 0, heat_data["purchases"] / heat_data["views"] * 100, 0)
    heat_pivot = heat_data.pivot(index="dow", columns="hour", values="conv_rate").fillna(0)
    heat_pivot.index = [dow_labels[i] for i in heat_pivot.index]

    fig3 = px.imshow(
        heat_pivot, labels=dict(x="Hour of Day", y="Day of Week", color="Conv. Rate %"),
        color_continuous_scale="YlOrRd", aspect="auto",
    )
    fig3.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    # --- Drop-off by price segment ---
    st.subheader("Drop-off by Price Segment")
    sessions["price_segment"] = pd.cut(
        sessions["avg_price"],
        bins=[0, 10, 30, 60, 100, float("inf")],
        labels=["$0-10", "$10-30", "$30-60", "$60-100", "$100+"],
    )
    seg = sessions.groupby("price_segment").agg(
        total=("max_funnel_stage", "count"),
        viewed=("view_count", lambda x: (x > 0).sum()),
        carted=("cart_count", lambda x: (x > 0).sum()),
        purchased=("purchase_count", lambda x: (x > 0).sum()),
    ).reset_index()
    seg["view_to_cart"] = (seg["carted"] / seg["viewed"] * 100).round(1)
    seg["cart_to_purchase"] = (seg["purchased"] / seg["carted"] * 100).round(1)

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=seg["price_segment"], y=seg["view_to_cart"], name="View→Cart %", marker_color="#636EFA"))
    fig4.add_trace(go.Bar(x=seg["price_segment"], y=seg["cart_to_purchase"], name="Cart→Purchase %", marker_color="#00CC96"))
    fig4.update_layout(barmode="group", height=350, yaxis_title="Conversion Rate %",
                       margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig4, use_container_width=True)
