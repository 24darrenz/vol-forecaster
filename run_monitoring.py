from src.monitoring import run_drift_report, check_prediction_error

TICKER = "SPY"

# run drift detection
drift_result = run_drift_report(TICKER)
print(f"\nDrift detected: {drift_result['drift_detected']}")
print(f"Report saved to: {drift_result['report_path']}")

# check model degradation
error_result = check_prediction_error(TICKER)