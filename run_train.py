from src.train import train_har, train_xgboost, load_data
import mlflow

mlflow.set_experiment("volatility_forecasting")

print("Loading data...")
X_train, X_test, y_train, y_test = load_data("SPY")
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

print("\nTraining HAR baseline...")
har_model = train_har(X_train, X_test, y_train, y_test)

print("\nTraining XGBoost...")
xgb_model = train_xgboost(X_train, X_test, y_train, y_test)

print("\nDone. Run `mlflow ui` to view experiment results.")