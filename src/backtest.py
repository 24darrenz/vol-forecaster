import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from src.database import load_from_db


def run_backtest(
    ticker: str = "SPY",
    vol_threshold: float = 0.20,
    initial_capital: float = 10000.0
):
    """
    Simple volatility-timing strategy:
    - When predicted vol < threshold: invest (low risk environment)
    - When predicted vol >= threshold: move to cash (high risk environment)

    Args:
        ticker: ticker to backtest on
        vol_threshold: annualized vol level above which we exit to cash
        initial_capital: starting portfolio value
    """
    # load feature matrix
    df = load_from_db(f"features_{ticker.lower()}")

    # load model and generate predictions
    model = joblib.load("models/xgb_model.pkl")
    features = ['rv_daily', 'rv_weekly', 'rv_monthly']
    df['predicted_vol'] = model.predict(df[features])

    # use test set only — never backtest on training data
    split = int(len(df) * 0.8)
    df = df.iloc[split:].copy()

    # generate signal: 1 = invested, 0 = cash
    df['signal'] = (df['predicted_vol'] < vol_threshold).astype(int)

    # compute daily strategy returns
    # signal is based on previous day's forecast (avoid lookahead bias)
    df['signal'] = df['signal'].shift(1)
    df['strategy_return'] = df['signal'] * df['log_return']
    df['buyhold_return'] = df['log_return']

    # compute cumulative portfolio value
    df['strategy_value'] = initial_capital * np.exp(df['strategy_return'].cumsum())
    df['buyhold_value'] = initial_capital * np.exp(df['buyhold_return'].cumsum())

    return df


def compute_metrics(df: pd.DataFrame, initial_capital: float = 10000.0):
    """Compute key performance metrics for strategy vs buy-and-hold."""

    def sharpe(returns, periods=252):
        if returns.std() == 0:
            return 0
        return np.sqrt(periods) * returns.mean() / returns.std()

    def max_drawdown(values):
        peak = values.cummax()
        drawdown = (values - peak) / peak
        return drawdown.min()

    def total_return(values, capital):
        return (values.iloc[-1] - capital) / capital

    def annualized_return(returns, periods=252):
        total = np.exp(returns.sum())
        n_years = len(returns) / periods
        return total ** (1 / n_years) - 1

    strategy_returns = df['strategy_return'].dropna()
    buyhold_returns = df['buyhold_return'].dropna()

    metrics = {
        "Strategy": {
            "Total Return": f"{total_return(df['strategy_value'], initial_capital):.1%}",
            "Annualized Return": f"{annualized_return(strategy_returns):.1%}",
            "Sharpe Ratio": f"{sharpe(strategy_returns):.2f}",
            "Max Drawdown": f"{max_drawdown(df['strategy_value']):.1%}",
            "Days Invested": f"{int(df['signal'].sum())} / {len(df)}",
        },
        "Buy & Hold": {
            "Total Return": f"{total_return(df['buyhold_value'], initial_capital):.1%}",
            "Annualized Return": f"{annualized_return(buyhold_returns):.1%}",
            "Sharpe Ratio": f"{sharpe(buyhold_returns):.2f}",
            "Max Drawdown": f"{max_drawdown(df['buyhold_value']):.1%}",
            "Days Invested": f"{len(df)} / {len(df)}",
        }
    }

    return metrics


def plot_backtest(df: pd.DataFrame, ticker: str, vol_threshold: float):
    """Generate interactive backtest chart."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Portfolio Value",
            "Predicted Volatility vs Threshold",
            "Market Position (1=Invested, 0=Cash)"
        ),
        row_heights=[0.5, 0.3, 0.2]
    )

    # portfolio value
    fig.add_trace(go.Scatter(
        x=df.index, y=df['strategy_value'],
        name="Strategy", line=dict(color="#2ecc71", width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['buyhold_value'],
        name="Buy & Hold", line=dict(color="#3498db", width=2)
    ), row=1, col=1)

    # predicted vol vs threshold
    fig.add_trace(go.Scatter(
        x=df.index, y=df['predicted_vol'],
        name="Predicted Vol", line=dict(color="#e67e22", width=1.5)
    ), row=2, col=1)

    fig.add_hline(
        y=vol_threshold, line_dash="dash",
        line_color="red", annotation_text="Threshold",
        row=2, col=1
    )

    # position
    fig.add_trace(go.Scatter(
        x=df.index, y=df['signal'],
        name="Position", fill='tozeroy',
        line=dict(color="#9b59b6", width=1)
    ), row=3, col=1)

    fig.update_layout(
        title=f"{ticker} Volatility Timing Strategy vs Buy & Hold",
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig