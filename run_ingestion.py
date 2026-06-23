from src.ingestion import fetch_price_data
from src.features import build_feature_matrix
from src.database import save_to_db


TICKER = "SPY"
START_DATE = "2018-01-01"

# fetch raw prices
print(f"Fetching data for {TICKER}...")
raw_df = fetch_price_data(TICKER, start=START_DATE)
save_to_db(raw_df, table_name=f"prices_{TICKER.lower()}")

# compute features + target
print("Computing features...")
feature_df = build_feature_matrix(raw_df)
save_to_db(feature_df, table_name=f"features_{TICKER.lower()}")

print(feature_df[['close', 'log_return', 'realized_vol', 'rv_daily', 'rv_weekly', 'rv_monthly', 'target']].tail(10))