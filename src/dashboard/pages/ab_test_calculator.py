"""Page 4: A/B Test sample size and duration calculator."""

import math

import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats


def render():
    st.title("A/B Test Calculator")
    st.markdown("Calculate required sample size and test duration for conversion rate experiments.")

    col1, col2 = st.columns(2)
    with col1:
        baseline_rate = st.number_input("Baseline conversion rate (%)", 0.1, 99.0, 5.0, 0.1) / 100
        mde = st.number_input("Minimum Detectable Effect (%)", 0.1, 50.0, 10.0, 0.1) / 100
    with col2:
        significance = st.selectbox("Significance level", [0.01, 0.05, 0.10], index=1)
        power = st.selectbox("Statistical power", [0.80, 0.85, 0.90, 0.95], index=0)
        daily_traffic = st.number_input("Daily sessions (both variants)", 100, 1_000_000, 10_000, 100)

    # --- Sample size calculation ---
    alpha = significance
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    delta = abs(p2 - p1)

    # Per-variant sample size (two-proportion z-test)
    pooled_p = (p1 + p2) / 2
    n_per_variant = math.ceil(
        (z_alpha * math.sqrt(2 * pooled_p * (1 - pooled_p))
         + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / delta ** 2
    )
    total_sample = n_per_variant * 2
    days_needed = math.ceil(total_sample / daily_traffic)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample per variant", f"{n_per_variant:,}")
    c2.metric("Total sample needed", f"{total_sample:,}")
    c3.metric("Days to run", f"{days_needed:,}")
    c4.metric("Expected lift", f"{p1:.2%} -> {p2:.2%}")

    # --- Power curve ---
    st.subheader("Power Curve")
    st.markdown("How statistical power changes with sample size for your parameters.")

    sample_range = np.linspace(max(100, n_per_variant * 0.2), n_per_variant * 3, 200)
    powers = []
    for n in sample_range:
        se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
        z = delta / se - z_alpha
        powers.append(stats.norm.cdf(z))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_range, y=powers, mode="lines", name="Power", line=dict(color="#636EFA")))
    fig.add_hline(y=power, line_dash="dash", line_color="#EF553B", annotation_text=f"Target power={power}")
    fig.add_vline(x=n_per_variant, line_dash="dash", line_color="#00CC96", annotation_text=f"n={n_per_variant:,}")
    fig.update_layout(
        xaxis_title="Sample Size (per variant)",
        yaxis_title="Statistical Power",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- MDE sensitivity table ---
    st.subheader("Sensitivity: Days Needed by MDE and Daily Traffic")
    mde_values = [0.05, 0.10, 0.15, 0.20, 0.25]
    traffic_values = [1000, 5000, 10000, 50000, 100000]

    rows = []
    for m in mde_values:
        row = {"MDE": f"{m:.0%}"}
        p2_m = baseline_rate * (1 + m)
        delta_m = abs(p2_m - baseline_rate)
        pooled_m = (baseline_rate + p2_m) / 2
        n_m = math.ceil(
            (z_alpha * math.sqrt(2 * pooled_m * (1 - pooled_m))
             + z_beta * math.sqrt(baseline_rate * (1 - baseline_rate) + p2_m * (1 - p2_m))) ** 2
            / delta_m ** 2
        )
        for t in traffic_values:
            days = math.ceil(n_m * 2 / t)
            row[f"{t:,}/day"] = f"{days} days"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
