# Volatility Forecaster

A production-style MLOps system that forecasts short-term market volatility, monitors itself for data drift, retrains automatically, and trades on its own predictions in a live paper-trading account.

Built to demonstrate end-to-end ML engineering — not just a model in a notebook, but the full lifecycle: ingestion, training, serving, monitoring, retraining, and deployment.

**Live components:** FastAPI prediction endpoint · Streamlit dashboard · Alpaca paper trading (daily automated runs)

---

## Why volatility, not price

Most student ML projects predict stock *price*, which is close to a random walk and notoriously unreliable to forecast. Volatility is different — it's mean-reverting, clusters predictably, and is what real financial systems (options pricing, risk management, position sizing) actually depend on. Forecasting it well is a more tractable and more honest problem.

---

## System overview

```
Data ingestion (yfinance)
        ↓
Feature engineering (HAR: daily/weekly/monthly realized vol)
        ↓
Model training + experiment tracking (MLflow)
        ↓
Model serving (FastAPI)
        ↓
Drift detection (manual PSI implementation) + performance monitoring
        ↓
Automated retraining (only promotes new model if it beats the current one)
        ↓
Paper trading (Alpaca) + Dashboard (Streamlit)
```

Everything above runs end-to-end from a single daily script (`run_daily.py`), designed to be scheduled and left running unattended.

---

## Tech stack

| Layer | Tool |
|---|---|
| Data | yfinance, SQLite |
| Modeling | scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Serving | FastAPI |
| Drift detection | Custom PSI (Population Stability Index) implementation |
| Trading | Alpaca (paper trading API) |
| Dashboard | Streamlit, Plotly |

---

## What the system actually does

**1. Ingests fresh SPY price data daily** and computes realized volatility using a rolling 20-day window, annualized.

**2. Trains and tracks models with MLflow**, comparing an XGBoost model against a HAR (Heterogeneous Autoregressive) baseline — the standard econometric benchmark for volatility forecasting, using daily/weekly/monthly realized vol as features.

**3. Serves live predictions via FastAPI** with auto-generated docs at `/docs`, returning a next-day annualized volatility forecast for any ticker.

**4. Detects data drift** using a from-scratch PSI implementation (no black-box library) — bucketing feature distributions and measuring how far current data has shifted from the training distribution. Currently flags significant drift in longer-horizon volatility features, consistent with the market environment since the training window.

**5. Retrains automatically when triggered** by drift or performance degradation, but only promotes the new model if it actually beats the current one on held-out RMSE — preventing silent model regressions.

**6. Backtests a volatility-timing strategy**: go long when predicted vol is below a threshold, move to cash when it's above. Evaluated against buy-and-hold on Sharpe ratio, max drawdown, and total return — not cherry-picked, reported honestly (see Results below).

**7. Trades on its own forecasts** through Alpaca's paper trading API, running daily and adjusting position based on the current model's output.

**8. Visualizes everything** in a Streamlit dashboard — live forecasts, prediction vs. actual volatility over time, model error trends, drift status, and paper portfolio performance.

---

## Results (backtest, out-of-sample)

| Metric | Strategy (vol-timing) | Buy & Hold |
|---|---|---|
| Total Return | 15.4% | 31.1% |
| Annualized Return | 8.9% | 17.5% |
| Sharpe Ratio | 0.65 | 0.94 |
| Max Drawdown | -15.9% | -18.8% |

**Honest read:** the strategy underperformed buy-and-hold on raw return during this test window, which spans an unusually strong bull market — exiting to cash during high-vol days has an opportunity cost when the market keeps climbing anyway. It did, however, reduce max drawdown, meaning it took less risk to get a lower return. The expected regime where this strategy adds real value is a higher-volatility or bear market period, where avoiding drawdowns matters more than capturing every point of upside. I'd rather report this accurately than cherry-pick a favorable window.

---

## What I'd improve next

- Hyperparameter tuning (current model uses reasonable defaults, not optimized)
- Additional features: VIX, volume-based signals, macro indicators
- Per-ticker models instead of one model applied across tickers
- Backtesting across a bear-market window to test the strategy's actual thesis
- Containerizing the full system with Docker

---

## Running it locally

```bash
pip install -r requirements.txt

# 1. Ingest data + train baseline models
python3 run_ingestion.py
python3 run_train.py

# 2. Start the API
python3 -m uvicorn src.api:app --reload

# 3. Launch the dashboard
python3 -m streamlit run dashboard.py

# 4. Run the full daily pipeline (ingest → drift check → retrain → trade)
python3 run_daily.py
```

Requires a free [Alpaca](https://alpaca.markets) paper trading account for the trading component; API keys go in a `.env` file (not committed).