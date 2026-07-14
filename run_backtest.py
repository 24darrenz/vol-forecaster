from src.backtest import run_backtest, compute_metrics, plot_backtest
import pandas as pd

TICKER = "SPY"
VOL_THRESHOLD = 0.20
INITIAL_CAPITAL = 10000.0

print(f"Running backtest for {TICKER}...")
df = run_backtest(TICKER, VOL_THRESHOLD, INITIAL_CAPITAL)

print("\n--- Performance Metrics ---")
metrics = compute_metrics(df, INITIAL_CAPITAL)
results = pd.DataFrame(metrics)
print(results.to_string())