from src.trader import (
    get_account_info,
    get_current_position,
    execute_trade,
    get_portfolio_history
)

TICKER = "SPY"
VOL_THRESHOLD = 0.20

# check account
print("Account Info:")
account = get_account_info()
for k, v in account.items():
    print(f"  {k}: ${v:,.2f}")

# check current position
print(f"\nCurrent {TICKER} Position:")
position = get_current_position(TICKER)
if position:
    for k, v in position.items():
        print(f"  {k}: {v}")
else:
    print("  No position")

# execute trade
print(f"\nExecuting trade for {TICKER}...")
result = execute_trade(TICKER, VOL_THRESHOLD)
print(f"  Decision: {result['decision']}")
print(f"  Forecast Vol: {result['forecast_vol']:.4f}")
print(f"  Message: {result['message']}")
print(f"  Account Value: ${result['account_value']:,.2f}")