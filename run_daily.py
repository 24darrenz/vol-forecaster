from src.pipeline import run_pipeline
from src.trader import execute_trade, get_account_info, get_current_position
from datetime import datetime

TICKER = "SPY"
VOL_THRESHOLD = 0.20

print(f"\n{'='*50}")
print(f"Daily Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*50}\n")

# step 1: run full MLOps pipeline
# (ingest fresh data, check drift, retrain if needed)
print("STEP 1: Running MLOps pipeline...")
run_pipeline(ticker=TICKER, start="2018-01-01")

# step 2: execute trade based on latest forecast
print("\nSTEP 2: Executing trade...")
result = execute_trade(TICKER, VOL_THRESHOLD)
print(f"  Decision: {result['decision']}")
print(f"  Forecast Vol: {result['forecast_vol']:.4f}")
print(f"  Message: {result['message']}")

# step 3: print account summary
print("\nSTEP 3: Account Summary:")
account = get_account_info()
for k, v in account.items():
    print(f"  {k}: ${v:,.2f}")

position = get_current_position(TICKER)
if position:
    print(f"\n  {TICKER} Position:")
    for k, v in position.items():
        print(f"    {k}: {v}")

print(f"\n{'='*50}")
print("Daily run complete.")
print(f"{'='*50}\n")