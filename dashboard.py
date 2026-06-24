import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import requests
from src.database import load_from_db
from src.monitoring import run_drift_report, check_prediction_error
from datetime import datetime

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Volatility Forecasting System",
    page_icon="📈",
    layout="wide"
)

# ── header ────────────────────────────────────────────────────────────────────
st.title("📈 Volatility Forecasting System")
st.caption("Real-time volatility forecasting with drift detection and automated retraining")

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
ticker = st.sidebar.selectbox("Ticker", ["SPY", "AAPL", "TSLA", "MSFT", "GOOGL"])
lookback_days = st.sidebar.slider("Lookback days for chart", 60, 500, 252)
api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Run Pipeline"):
    with st.spinner("Running pipeline..."):
        from src.pipeline import run_pipeline
        run_pipeline(ticker=ticker)
    st.sidebar.success("Pipeline complete!")

# ── live forecast ─────────────────────────────────────────────────────────────
st.header("Live Forecast")

col1, col2, col3 = st.columns(3)

try:
    response = requests.post(
        f"{api_url}/predict",
        json={"ticker": ticker, "lookback_days": 60},
        timeout=5
    )
    forecast_data = response.json()

    with col1:
        st.metric(
            label=f"{ticker} Forecast Volatility",
            value=forecast_data['annualized_percent'],
            help="Predicted next-day annualized realized volatility"
        )
    with col2:
        st.metric(
            label="Model",
            value=forecast_data['model']
        )
    with col3:
        st.metric(
            label="Last Updated",
            value=datetime.fromisoformat(forecast_data['timestamp']).strftime("%H:%M:%S UTC")
        )

except Exception as e:
    st.warning(f"API not reachable — start the FastAPI server to see live forecasts. ({e})")

# ── volatility chart ──────────────────────────────────────────────────────────
st.header("Realized Volatility History")

try:
    df = load_from_db(f"features_{ticker.lower()}")
    df = df.iloc[-lookback_days:]

    # load model and generate predictions
    model = joblib.load("models/xgb_model.pkl")
    features = ['rv_daily', 'rv_weekly', 'rv_monthly']
    df['predicted_vol'] = model.predict(df[features])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['realized_vol'],
        name="Realized Volatility",
        line=dict(color="#1f77b4", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['predicted_vol'],
        name="Predicted Volatility",
        line=dict(color="#ff7f0e", width=2, dash="dash")
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Could not load data: {e}")

# ── model performance ─────────────────────────────────────────────────────────
st.header("Model Performance")

try:
    df_full = load_from_db(f"features_{ticker.lower()}")
    model = joblib.load("models/xgb_model.pkl")
    features = ['rv_daily', 'rv_weekly', 'rv_monthly']

    df_full['predicted_vol'] = model.predict(df_full[features])
    df_full['error'] = np.abs(df_full['predicted_vol'] - df_full['target'])

    split = int(len(df_full) * 0.8)

    col1, col2, col3 = st.columns(3)

    with col1:
        historical_mae = df_full.iloc[:split]['error'].mean()
        st.metric("Historical MAE", f"{historical_mae:.4f}")

    with col2:
        recent_mae = df_full.iloc[-30:]['error'].mean()
        delta = recent_mae - historical_mae
        st.metric(
            "Recent MAE (30d)",
            f"{recent_mae:.4f}",
            delta=f"{delta:+.4f}",
            delta_color="inverse"
        )

    with col3:
        rmse = np.sqrt((df_full.iloc[split:]['error'] ** 2).mean())
        st.metric("Test RMSE", f"{rmse:.4f}")

    # rolling error chart
    df_full['rolling_error'] = df_full['error'].rolling(30).mean()
    fig2 = px.line(
        df_full.iloc[-lookback_days:],
        y='rolling_error',
        title="30-Day Rolling MAE Over Time",
        labels={"rolling_error": "MAE", "date": "Date"}
    )
    fig2.update_traces(line_color="#e74c3c")
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"Could not compute performance metrics: {e}")

# ── drift status ──────────────────────────────────────────────────────────────
st.header("Drift Status")

try:
    drift_result = run_drift_report(ticker, save=False)
    feature_results = drift_result['feature_results']

    overall = drift_result['drift_detected']
    if overall:
        st.error("⚠️ Drift Detected — retraining may be needed")
    else:
        st.success("✅ No Drift Detected — model is stable")

    # feature drift table
    drift_df = pd.DataFrame([
        {
            "Feature": feat,
            "PSI Score": res['psi'],
            "Severity": res['severity'].capitalize(),
            "Drift": "Yes" if res['drift_detected'] else "No"
        }
        for feat, res in feature_results.items()
    ])

    st.dataframe(drift_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Could not run drift detection: {e}")

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown