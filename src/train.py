import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from src.database import load_from_db


def load_data(ticker: str):
    """Load feature matrix from database and split into train/test."""
    df = load_from_db(f"features_{ticker.lower()}")

    features = ['rv_daily', 'rv_weekly', 'rv_monthly']
    target = 'target'

    X = df[features]
    y = df[target]

    # 80/20 chronological split — never shuffle time series data
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred):
    """Compute evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # directional accuracy: did we get the direction of vol change right?
    direction_acc = np.mean(np.sign(y_pred - y_true.values) == 0)
    return {"rmse": rmse, "mae": mae, "direction_accuracy": direction_acc}


def train_har(X_train, X_test, y_train, y_test):
    """Train baseline HAR model (linear regression)."""
    with mlflow.start_run(run_name="HAR_baseline"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate(y_test, y_pred)

        # log parameters
        mlflow.log_param("model_type", "HAR_linear")
        mlflow.log_param("features", list(X_train.columns))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # log metrics
        mlflow.log_metrics(metrics)

        # log model artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"HAR baseline — RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | Dir. Acc: {metrics['direction_accuracy']:.4f}")

    return model


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model."""
    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }

    with mlflow.start_run(run_name="XGBoost"):
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        y_pred = model.predict(X_test)

        metrics = evaluate(y_test, y_pred)

        # log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_params(params)

        # log metrics
        mlflow.log_metrics(metrics)

        # log model artifact
        mlflow.xgboost.log_model(model, "model")

        print(f"XGBoost — RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | Dir. Acc: {metrics['direction_accuracy']:.4f}")

    return model


if __name__ == "__main__":
    mlflow.set_experiment("volatility_forecasting")

    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data("SPY")
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("\nTraining HAR baseline...")
    har_model = train_har(X_train, X_test, y_train, y_test)

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, X_test, y_train, y_test)

    print("\nDone. Run `mlflow ui` to view experiment results.")