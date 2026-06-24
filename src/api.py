from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from src.features import build_feature_matrix

# initialize app
app = FastAPI(
    title="Volatility Forecasting API",
    description="Predicts next-day realized volatility for a given ticker",
    version="1.0.0"
)

# load model at startup so it's ready for requests
model = joblib.load("models/xgb_model.pkl")


# define what a request looks like
class PredictRequest(BaseModel):
    ticker: str
    lookback_days: int = 60  # how many days of history to fetch


# define what a response looks like
class PredictResponse(BaseModel):
    ticker: str
    forecast_volatility: float
    annualized_percent: str
    model: str
    timestamp: str


@app.get("/")
def root():
    return {"message": "Volatility Forecasting API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # fetch recent price data for the ticker
        df = yf.download(
            request.ticker,
            period=f"{request.lookback_days}d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {request.ticker}")

        # flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        # compute features
        feature_df = build_feature_matrix(df)

        if feature_df.empty:
            raise HTTPException(status_code=400, detail="Not enough data to compute features")

        # use the most recent row as input
        features = ['rv_daily', 'rv_weekly', 'rv_monthly']
        X = feature_df[features].iloc[[-1]]

        # generate prediction
        forecast = model.predict(X)[0]

        return PredictResponse(
            ticker=request.ticker.upper(),
            forecast_volatility=round(float(forecast), 6),
            annualized_percent=f"{round(float(forecast) * 100, 2)}%",
            model="XGBoost_v1",
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))