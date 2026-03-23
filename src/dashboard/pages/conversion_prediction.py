"""Page 3: Conversion prediction with SHAP explanations."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_model, load_feature_cols, load_shap_importance, load_metrics, load_ml_features


def render():
    st.title("Conversion Prediction")
    st.markdown("Predict conversion probability for a user session and understand what drives the prediction.")

    model = load_model()
    feature_cols = load_feature_cols()
    shap_imp = load_shap_importance()
    metrics = load_metrics()
    ml_data = load_ml_features()

    # --- Model performance summary ---
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC (test)", f"{metrics['test_roc_auc']:.4f}")
    col2.metric("PR-AUC (test)", f"{metrics['test_pr_auc']:.4f}")
    col3.metric("CV ROC-AUC", f"{metrics['cv_roc_auc']:.4f} +/- {metrics['cv_roc_auc_std']:.4f}")

    # --- SHAP feature importance ---
    st.subheader("Feature Importance (SHAP)")
    top_n = st.slider("Top N features", 5, len(shap_imp), 10)
    top_features = shap_imp.head(top_n)

    fig = px.bar(
        top_features, x="mean_abs_shap", y="feature", orientation="h",
        color="mean_abs_shap", color_continuous_scale="Viridis",
        labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""},
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(300, top_n * 30),
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Interactive prediction ---
    st.subheader("Predict for a Session Profile")
    st.markdown("Adjust the sliders to simulate a user session and see the predicted conversion probability.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            total_events = st.slider("Total events", 1, 100, 10)
            view_count = st.slider("View count", 1, 80, 8)
            cart_count = st.slider("Cart count", 0, 20, 1)
            unique_products = st.slider("Unique products", 1, 50, 5)
        with c2:
            avg_price = st.slider("Avg price ($)", 1.0, 150.0, 30.0)
            session_duration = st.slider("Session duration (sec)", 0, 3600, 300)
            unique_brands = st.slider("Unique brands", 1, 15, 3)
            unique_categories = st.slider("Unique categories", 1, 8, 2)
        with c3:
            start_hour = st.slider("Hour of day", 0, 23, 14)
            day_of_week = st.slider("Day of week (0=Mon)", 0, 6, 2)
            prior_sessions = st.slider("Prior sessions", 0, 50, 2)
            prior_purchases = st.slider("Prior purchases", 0, 20, 0)

        submitted = st.form_submit_button("Predict Conversion")

    if submitted:
        # Build feature vector matching model's expected columns
        input_data = {col: 0.0 for col in feature_cols}
        input_data.update({
            "total_events": total_events,
            "view_count": view_count,
            "cart_count": cart_count,
            "unique_products": unique_products,
            "avg_price": avg_price,
            "max_price": avg_price * 1.5,
            "min_price": avg_price * 0.5,
            "price_range": avg_price,
            "session_duration_sec": session_duration,
            "unique_brands": unique_brands,
            "unique_categories": unique_categories,
            "start_hour": start_hour,
            "day_of_week": day_of_week,
            "is_weekend": 1 if day_of_week >= 5 else 0,
            "prior_sessions": prior_sessions,
            "prior_purchases": prior_purchases,
            "days_since_first_seen": prior_sessions * 3,
            "days_since_last_session": 2 if prior_sessions > 0 else 0,
            "cart_to_view_ratio": cart_count / view_count if view_count > 0 else 0,
            "product_diversity_ratio": unique_products / total_events if total_events > 0 else 0,
            "is_returning_user": 1 if prior_sessions > 0 else 0,
        })

        input_df = pd.DataFrame([{col: input_data.get(col, 0) for col in feature_cols}])
        prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            color = "#00CC96" if prob > 0.5 else "#EF553B"
            st.markdown(f"### Conversion Probability: <span style='color:{color}'>{prob:.1%}</span>",
                        unsafe_allow_html=True)
            if prob > 0.7:
                st.success("High likelihood of conversion. Prioritize this user for checkout nudges.")
            elif prob > 0.4:
                st.warning("Moderate likelihood. Consider targeted offers or cart reminders.")
            else:
                st.error("Low likelihood. Investigate session quality and product relevance.")

        with col_b:
            # Show a gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Conversion Score"},
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=color),
                    steps=[
                        dict(range=[0, 40], color="#FFCCCC"),
                        dict(range=[40, 70], color="#FFFFCC"),
                        dict(range=[70, 100], color="#CCFFCC"),
                    ],
                ),
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Key drivers for this prediction
        st.subheader("Key Drivers")
        top3 = shap_imp.head(3)["feature"].tolist()
        for feat in top3:
            val = input_data.get(feat, 0)
            med = ml_data[feat].median() if feat in ml_data.columns else 0
            direction = "above" if val > med else "below"
            st.markdown(f"- **{feat}**: {val:.2f} ({direction} median of {med:.2f})")
