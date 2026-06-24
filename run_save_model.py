import mlflow
import mlflow.xgboost
from src.train import load_data, train_xgboost

mlflow.set_experiment("volatility_forecasting")

print("Loading data...")
X_train, X_test, y_train, y_test = load_data("SPY")

print("Training and saving XGBoost model...")
model = train_xgboost(X_train, X_test, y_train, y_test)

# save model to disk so FastAPI can load it
import joblib
import os

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")
print("Model saved to models/xgb_model.pkl")