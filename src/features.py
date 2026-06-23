import pandas as pd
import numpy as np

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns are the standard in finance.
    More statistically well-behaved than simple % returns.
    log(P_t / P_{t-1})
    """
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df.dropna()


def compute_realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Realized vol = rolling std of log returns, annualized.
    Window of 20 = ~1 trading month.
    Multiplying by sqrt(252) annualizes it (252 trading days/year).
    """
    df = df.copy()
    df['realized_vol'] = (
        df['log_return']
        .rolling(window=window)
        .std() * np.sqrt(252)
    )
    return df


def compute_har_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    HAR model features: daily, weekly, monthly realized vol.
    These become your X variables for the baseline model.
    
    The intuition: vol today is influenced by vol
    over the past day, week, and month — each captures
    a different type of market participant.
    """
    df = df.copy()

    # daily vol: just the squared log return (single day estimate)
    df['rv_daily'] = df['log_return'] ** 2

    # weekly: 5-day rolling average of daily vol
    df['rv_weekly'] = df['rv_daily'].rolling(window=5).mean()

    # monthly: 22-day rolling average of daily vol
    df['rv_monthly'] = df['rv_daily'].rolling(window=22).mean()

    return df


def build_feature_matrix(df: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
    """
    Combine everything into a clean feature matrix.
    Target is realized_vol shifted back by forecast_horizon
    so we're always predicting FUTURE vol from CURRENT features.
    """
    df = compute_log_returns(df)
    df = compute_realized_volatility(df)
    df = compute_har_features(df)

    # shift target so model learns to predict forward
    df['target'] = df['realized_vol'].shift(-forecast_horizon)

    # drop rows with any NaN (from rolling windows + shift)
    df = df.dropna()

    return df