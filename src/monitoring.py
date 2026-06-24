import pandas as pd
import numpy as np
from src.database import load_from_db
import joblib
import os
from datetime import datetime


def load_reference_and_current(ticker: str, current_window: int = 60):
    """
    Load feature matrix and split into:
    - reference: training data (first 80%)
    - current: recent data (last current_window days)
    """
    df = load_from_db(f"features_{ticker.lower()}")
    features = ['rv_daily', 'rv_weekly', 'rv_monthly', 'target']
    split = int(len(df) * 0.8)
    reference = df.iloc[:split][features].copy()
    current = df.iloc[-current_window:][features].copy()
    return reference, current


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.1):
    """
    Manual drift detection using Population Stability Index (PSI).
    PSI < 0.1: no drift
    PSI 0.1-0.2: moderate drift
    PSI > 0.2: significant drift
    """
    features = ['rv_daily', 'rv_weekly', 'rv_monthly']
    results = {}

    for feature in features:
        ref = reference[feature].dropna()
        cur = current[feature].dropna()

        # bin reference data into 10 buckets
        bins = np.percentile(ref, np.linspace(0, 100, 11))
        bins[0] -= 1e-10
        bins[-1] += 1e-10

        ref_counts = np.histogram(ref, bins=bins)[0]
        cur_counts = np.histogram(cur, bins=bins)[0]

        # convert to proportions, avoid division by zero
        ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / len(ref))
        cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / len(cur))

        # PSI formula
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        results[feature] = {
            "psi": round(float(psi), 4),
            "drift_detected": psi > threshold,
            "severity": "none" if psi < 0.1 else "moderate" if psi < 0.2 else "significant"
        }

    overall_drift = any(r["drift_detected"] for r in results.values())

    return {
        "overall_drift_detected": overall_drift,
        "features": results
    }


def run_drift_report(ticker: str, current_window: int = 60, save: bool = True):
    """
    Generate drift report and optionally save as HTML.
    """
    print(f"Running drift detection for {ticker}...")
    reference, current = load_reference_and_current(ticker, current_window)
    drift_results = compute_drift(reference, current)

    # print summary
    print(f"\nDrift Report for {ticker}:")
    print(f"  Overall drift detected: {drift_results['overall_drift_detected']}")
    for feature, result in drift_results['features'].items():
        print(f"  {feature}: PSI={result['psi']} | {result['severity']}")

    # save simple HTML report
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/drift_report_{ticker}_{timestamp}.html"

    if save:
        _save_html_report(ticker, drift_results, reference, current, report_path, timestamp)
        print(f"\nReport saved to {report_path}")

    return {
        "ticker": ticker,
        "drift_detected": drift_results['overall_drift_detected'],
        "feature_results": drift_results['features'],
        "report_path": report_path if save else None,
        "timestamp": timestamp
    }


def _save_html_report(ticker, drift_results, reference, current, path, timestamp):
    """Save a simple HTML drift report."""
    rows = ""
    for feature, result in drift_results['features'].items():
        color = "#d4edda" if not result['drift_detected'] else "#f8d7da"
        rows += f"""
        <tr style="background:{color}">
            <td>{feature}</td>
            <td>{result['psi']}</td>
            <td>{result['severity']}</td>
            <td>{'Yes' if result['drift_detected'] else 'No'}</td>
        </tr>"""

    overall_color = "#d4edda" if not drift_results['overall_drift_detected'] else "#f8d7da"
    overall_text = "Drift Detected" if drift_results['overall_drift_detected'] else "No Drift Detected"

    html = f"""
    <html>
    <head>
        <title>Drift Report - {ticker}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 30px; }}
            table {{ border-collapse: collapse; width: 60%; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background: #343a40; color: white; }}
            .badge {{ padding: 8px 16px; border-radius: 4px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Volatility Forecasting — Drift Report</h1>
        <p><b>Ticker:</b> {ticker} | <b>Timestamp:</b> {timestamp}</p>
        <p><b>Reference size:</b> {len(reference)} rows |
           <b>Current window:</b> {len(current)} rows</p>
        <p><span class="badge" style="background:{overall_color}">{overall_text}</span></p>
        <h2>Feature Drift (PSI)</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>PSI Score</th>
                <th>Severity</th>
                <th>Drift?</th>
            </tr>
            {rows}
        </table>
        <br>
        <p style="color:gray; font-size:12px">
            PSI &lt; 0.1: No drift | 0.1–0.2: Moderate | &gt; 0.2: Significant
        </p>
    </body>
    </html>
    """

    with open(path, "w") as f:
        f.write(html)


def check_prediction_error(ticker: str, threshold: float = 0.02):
    """
    Compare recent prediction error vs historical error.
    Flags for retraining if degradation exceeds threshold.
    """
    df = load_from_db(f"features_{ticker.lower()}")
    features = ['rv_daily', 'rv_weekly', 'rv_monthly']

    model = joblib.load("models/xgb_model.pkl")

    df['prediction'] = model.predict(df[features])
    df['error'] = np.abs(df['prediction'] - df['target'])

    split = int(len(df) * 0.8)
    historical_error = df.iloc[:split]['error'].mean()
    recent_error = df.iloc[-30:]['error'].mean()

    degradation = recent_error - historical_error
    needs_retraining = degradation > threshold

    print(f"\nModel Error Check for {ticker}:")
    print(f"  Historical MAE: {historical_error:.4f}")
    print(f"  Recent MAE (last 30 days): {recent_error:.4f}")
    print(f"  Degradation: {degradation:.4f}")
    print(f"  Needs retraining: {needs_retraining}")

    return {
        "ticker": ticker,
        "historical_mae": float(historical_error),
        "recent_mae": float(recent_error),
        "degradation": float(degradation),
        "needs_retraining": bool(needs_retraining)
    }