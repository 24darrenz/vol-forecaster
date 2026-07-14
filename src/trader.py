import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from src.database import load_from_db
from src.features import build_feature_matrix
from datetime import datetime, timedelta
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_trading_client():
    """Initialize Alpaca trading client."""
    return TradingClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True  # always paper trading
    )


def get_data_client():
    """Initialize Alpaca data client."""
    return StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY")
    )


def get_account_info():
    """Get current paper account status."""
    client = get_trading_client()
    account = client.get_account()

    return {
        "portfolio_value": float(account.portfolio_value),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "equity": float(account.equity)
    }


def get_current_position(ticker: str):
    """Get current position for a ticker, returns None if no position."""
    client = get_trading_client()
    try:
        position = client.get_open_position(ticker)
        return {
            "qty": float(position.qty),
            "market_value": float(position.market_value),
            "unrealized_pl": float(position.unrealized_pl),
            "unrealized_plpc": float(position.unrealized_plpc)
        }
    except Exception:
        return None


def get_forecast(ticker: str):
    """Get volatility forecast for a ticker using the trained model."""
    try:
        # try loading from database first
        df = load_from_db(f"features_{ticker.lower()}")
        model = joblib.load("models/xgb_model.pkl")
        features = ['rv_daily', 'rv_weekly', 'rv_monthly']
        X = df[features].iloc[[-1]]
        forecast = float(model.predict(X)[0])
        return forecast
    except Exception as e:
        logger.error(f"Could not get forecast for {ticker}: {e}")
        return None


def make_trading_decision(ticker: str, vol_threshold: float = 0.20):
    """
    Decide whether to buy, sell, or hold based on vol forecast.
    - predicted vol < threshold: BUY (low risk environment)
    - predicted vol >= threshold: SELL (high risk environment, move to cash)
    """
    forecast = get_forecast(ticker)
    if forecast is None:
        return "HOLD", None

    if forecast < vol_threshold:
        decision = "BUY"
    else:
        decision = "SELL"

    logger.info(f"{ticker} forecast vol: {forecast:.4f} | threshold: {vol_threshold} | decision: {decision}")
    return decision, forecast


def execute_trade(ticker: str, vol_threshold: float = 0.20):
    """
    Execute a trade based on volatility forecast.
    Uses fixed 90% of buying power for buys.
    """
    client = get_trading_client()
    decision, forecast = make_trading_decision(ticker, vol_threshold)

    account = get_account_info()
    current_position = get_current_position(ticker)

    result = {
        "ticker": ticker,
        "decision": decision,
        "forecast_vol": forecast,
        "timestamp": datetime.utcnow().isoformat(),
        "account_value": account['portfolio_value'],
        "order": None,
        "message": ""
    }

    if decision == "BUY" and current_position is None:
        # only buy if we don't already have a position
        try:
            # use 90% of available cash
            cash = account['cash']
            notional = round(cash * 0.90, 2)

            if notional < 1:
                result['message'] = "Not enough cash to buy"
                return result

            order = MarketOrderRequest(
                symbol=ticker,
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            submitted = client.submit_order(order)
            result['order'] = str(submitted.id)
            result['message'] = f"BUY order submitted — ${notional:.2f} notional"
            logger.info(result['message'])

        except Exception as e:
            result['message'] = f"BUY failed: {str(e)}"
            logger.error(result['message'])

    elif decision == "SELL" and current_position is not None:
        # only sell if we have a position
        try:
            client.close_position(ticker)
            result['message'] = f"SELL order submitted — closed position"
            logger.info(result['message'])

        except Exception as e:
            result['message'] = f"SELL failed: {str(e)}"
            logger.error(result['message'])

    else:
        # already in the right state
        if decision == "BUY" and current_position is not None:
            result['message'] = "Already invested — holding"
        elif decision == "SELL" and current_position is None:
            result['message'] = "Already in cash — holding"
        logger.info(result['message'])

    return result


def get_portfolio_history():
    """Get historical portfolio performance from Alpaca."""
    client = get_trading_client()
    try:
        history = client.get_portfolio_history(
            period="1M",
            timeframe="1D"
        )
        timestamps = [datetime.fromtimestamp(t) for t in history.timestamp]
        values = history.equity

        df = pd.DataFrame({
            "date": timestamps,
            "portfolio_value": values
        }).set_index("date")

        return df
    except Exception as e:
        logger.error(f"Could not fetch portfolio history: {e}")
        return None