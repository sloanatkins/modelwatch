"""
ModelWatch Monitoring Dashboard
================================

Streamlit + Plotly dashboard that reads from PostgreSQL and surfaces:
1. Prediction volume and fraud rate over time
2. Per-feature drift scores (KS statistic and PSI)
3. Alert status per feature
4. Model version in use

Run with:
    streamlit run dashboard/app.py

Dependencies:
- streamlit
- plotly
- sqlalchemy
- pandas
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

# ── Database connection ──────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://modelwatch:modelwatch@localhost:5432/modelwatch"
)
engine = create_engine(DATABASE_URL)


# ── Data loaders ─────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_predictions():
    query = """
        SELECT
            timestamp,
            model_version,
            prediction,
            fraud_probability,
            latency_ms,
            status
        FROM prediction_log
        ORDER BY timestamp ASC
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


@st.cache_data(ttl=30)
def load_drift_metrics():
    query = """
        SELECT
            computed_at,
            model_version,
            feature_name,
            ks_statistic,
            ks_p_value,
            psi_score,
            alert
        FROM drift_metrics
        ORDER BY computed_at DESC
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="ModelWatch",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 ModelWatch")
st.caption("Production ML Observability & Drift Detection")

# ── Load data ─────────────────────────────────────────────────────────
pred_df = load_predictions()
drift_df = load_drift_metrics()

if pred_df.empty:
    st.warning("No predictions logged yet. Hit the /predict endpoint to get started.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
model_versions = pred_df["model_version"].unique().tolist()
selected_version = st.sidebar.selectbox("Model Version", model_versions)

st.sidebar.markdown("---")
st.sidebar.metric("Total Predictions", len(pred_df))
st.sidebar.metric(
    "Fraud Rate",
    f"{pred_df['prediction'].mean() * 100:.2f}%"
)
st.sidebar.metric(
    "Avg Latency",
    f"{pred_df['latency_ms'].mean():.0f} ms"
)

# ── Section 1: Prediction volume over time ────────────────────────────
st.subheader("Prediction Volume Over Time")

pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
pred_df["minute"] = pred_df["timestamp"].dt.floor("min")
volume_df = pred_df.groupby("minute").size().reset_index(name="count")

fig_volume = px.bar(
    volume_df,
    x="minute",
    y="count",
    labels={"minute": "Time", "count": "Predictions"},
    color_discrete_sequence=["#1F5C8B"]
)
fig_volume.update_layout(showlegend=False, height=300)
st.plotly_chart(fig_volume, use_container_width=True)

# ── Section 2: Fraud probability distribution ─────────────────────────
st.subheader("Fraud Probability Distribution")

fig_prob = px.histogram(
    pred_df.dropna(subset=["fraud_probability"]),
    x="fraud_probability",
    nbins=50,
    labels={"fraud_probability": "Fraud Probability"},
    color_discrete_sequence=["#1F5C8B"]
)
fig_prob.update_layout(showlegend=False, height=300)
st.plotly_chart(fig_prob, use_container_width=True)

# ── Section 3: Drift metrics ──────────────────────────────────────────
st.subheader("Feature Drift — Latest Monitoring Cycle")

if drift_df.empty:
    st.info("No drift metrics yet. Run the monitoring engine to compute drift.")
else:
    # Get most recent cycle per feature
    latest_drift = (
        drift_df
        .sort_values("computed_at", ascending=False)
        .drop_duplicates(subset=["feature_name"])
        .sort_values("ks_statistic", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**KS Statistic by Feature**")
        fig_ks = px.bar(
            latest_drift,
            x="ks_statistic",
            y="feature_name",
            orientation="h",
            color="alert",
            color_discrete_map={True: "#C0392B", False: "#1F5C8B"},
            labels={"ks_statistic": "KS Statistic", "feature_name": "Feature"},
        )
        fig_ks.add_vline(x=0.1, line_dash="dash", line_color="orange",
                         annotation_text="Alert threshold")
        fig_ks.update_layout(showlegend=False, height=600)
        st.plotly_chart(fig_ks, use_container_width=True)

    with col2:
        st.markdown("**PSI Score by Feature**")
        fig_psi = px.bar(
            latest_drift,
            x="psi_score",
            y="feature_name",
            orientation="h",
            color="alert",
            color_discrete_map={True: "#C0392B", False: "#1F5C8B"},
            labels={"psi_score": "PSI Score", "feature_name": "Feature"},
        )
        fig_psi.add_vline(x=0.2, line_dash="dash", line_color="orange",
                          annotation_text="Alert threshold")
        fig_psi.update_layout(showlegend=False, height=600)
        st.plotly_chart(fig_psi, use_container_width=True)

    # Alert summary table
    st.subheader("Alert Summary")
    alert_df = latest_drift[["feature_name", "ks_statistic", "psi_score", "alert"]].copy()
    alert_df.columns = ["Feature", "KS Statistic", "PSI Score", "Alert"]
    alert_df["KS Statistic"] = alert_df["KS Statistic"].round(4)
    alert_df["PSI Score"] = alert_df["PSI Score"].round(4)
    st.dataframe(
        alert_df.style.applymap(
            lambda v: "background-color: #fadbd8" if v is True else "",
            subset=["Alert"]
        ),
        use_container_width=True,
        hide_index=True
    )

# ── Section 4: Latency over time ──────────────────────────────────────
st.subheader("Prediction Latency Over Time")

fig_latency = px.line(
    pred_df.dropna(subset=["latency_ms"]),
    x="timestamp",
    y="latency_ms",
    labels={"timestamp": "Time", "latency_ms": "Latency (ms)"},
    color_discrete_sequence=["#1F5C8B"]
)
fig_latency.update_layout(height=300)
st.plotly_chart(fig_latency, use_container_width=True)
