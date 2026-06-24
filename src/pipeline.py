import mlflow
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from src.ingestion import fetch_price_data
from src.features import build_feature_matrix
from src.database import save_to_db, load_from_db
from src.monitoring import run_drift_report, check_prediction_error
from src.train import train_xgboost
from datetime import datetime
import os


def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def ingest_data(ticker: str, start: str):
    """Fetch latest price data and update database."""
    log(f"Fetching latest data for {ticker}...")

    raw_df = fetch_price_data(ticker, start=start)
    save_to_db(raw_df, table_name=f"prices_{ticker.lower()}")

    feature_df = build_feature_matrix(raw_df)
    save_to_db(feature_df, table_name=f"features_{ticker.lower()}")

    log(f"Data updated: {len(feature_df)} rows")
    return len(feature_df)


def check_drift(ticker: str):
    """Run drift detection and return whether retraining is needed."""
    log("Running drift detection...")

    drift_result = run_drift_report(ticker, save=True)
    error_result = check_prediction_error(ticker)

    needs_retraining = (
        drift_result['drift_detected'] or
        error_result['needs_retraining']
    )

    log(f"Drift detected: {drift_result['drift_detected']}")
    log(f"Model degradation: {error_result['needs_retraining']}")
    log(f"Retraining needed: {needs_retraining}")

    return needs_retraining


def retrain_model(ticker: str):
    """Retrain model and promote only if it beats the current model."""
    log("Retraining model...")

    mlflow.set_experiment("volatility_forecasting")

    # load data
    df = load_from_db(f"features_{ticker.lower()}")
    features = ['rv_daily', 'rv_weekly', 'rv_monthly']
    target = 'target'

    X = df[features]
    y = df[target]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # evaluate current model
    current_model = joblib.load("models/xgb_model.pkl")
    current_preds = current_model.predict(X_test)
    current_rmse = np.sqrt(mean_squared_error(y_test, current_preds))
    log(f"Current model RMSE: {current_rmse:.4f}")

    # train new model
    run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.end_run()
        new_model = train_xgboost(X_train, X_test, y_train, y_test)
        new_preds = new_model.predict(X_test)
        new_rmse = np.sqrt(mean_squared_error(y_test, new_preds))

        mlflow.log_metric("current_model_rmse", current_rmse)
        mlflow.log_metric("new_model_rmse", new_rmse)
        mlflow.log_param("trigger", "drift_or_degradation")

        log(f"New model RMSE: {new_rmse:.4f}")

        # only promote if new model is better
        if new_rmse < current_rmse:
            os.makedirs("models", exist_ok=True)
            joblib.dump(new_model, "models/xgb_model.pkl")
            log("New model promoted — replaces current model")
            mlflow.log_param("promoted", True)
            return True
        else:
            log("New model not better — keeping current model")
            mlflow.log_param("promoted", False)
            return False


def run_pipeline(ticker: str = "SPY", start: str = "2018-01-01"):
    """
    Full automated pipeline:
    1. Ingest fresh data
    2. Check for drift / degradation
    3. Retrain if needed
    """
    log(f"Pipeline started for {ticker}")
    log("=" * 50)

    # step 1: always ingest fresh data
    ingest_data(ticker, start)

    # step 2: check if retraining is needed
    needs_retraining = check_drift(ticker)

    # step 3: retrain only if flagged
    if needs_retraining:
        log("Retraining triggered...")
        promoted = retrain_model(ticker)
        if promoted:
            log("Pipeline complete — new model deployed")
        else:
            log("Pipeline complete — current model retained")
    else:
        log("Pipeline complete — no retraining needed")

    log("=" * 50)


if __name__ == "__main__":
    run_pipeline()