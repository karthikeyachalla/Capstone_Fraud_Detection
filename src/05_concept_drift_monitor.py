"""
05_concept_drift_monitor.py
===========================
Automated Statistical Concept Drift & Covariate Shift Surveillance.
IEEE-CIS Global Financial Fraud Benchmark.

STATISTICAL ENGINES:
  1. Two-sample Kolmogorov-Smirnov (KS) Test:
     D_KS = sup_x |F_ref(x) - F_stream(x)|
     Rejects null hypothesis H0 (same distribution) when p_value < 0.05.
  2. Population Stability Index (PSI):
     PSI = sum( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) )
     Thresholds:
       - PSI < 0.10: Stable, no action needed.
       - 0.10 <= PSI < 0.25: Moderate shift, alert triggered.
       - PSI >= 0.25: Severe drift, automated retraining required.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train_transaction.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index between reference and streaming windows."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_edges = np.percentile(expected, percentiles)
    bucket_edges = np.unique(bucket_edges)
    if len(bucket_edges) < 2:
        return 0.0

    bucket_edges[0] = -np.inf
    bucket_edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bucket_edges)
    act_counts, _ = np.histogram(actual, bins=bucket_edges)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    eps = 1e-4
    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(np.clip(psi, 0.0, 50.0))


def run_drift_monitor():
    print("=" * 70)
    print("  IEEE-CIS STATISTICAL CONCEPT DRIFT & COVARIATE SURVEILLANCE")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at {DATA_PATH}")
        return

    print("Loading 45,000 temporal transactions from train_transaction.csv...")
    cols = ["TransactionID", "TransactionDT", "TransactionAmt", "card1", "C1", "C2", "isFraud"]
    df = pd.read_csv(DATA_PATH, nrows=45000, usecols=cols)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Chronological sliding windows:
    # Baseline: First 15,000 transactions
    # Window 1: Next 15,000 transactions (In-Distribution stream)
    # Window 2: Next 15,000 transactions (Temporal shifted stream)
    baseline_df = df.iloc[:15000]
    stream_w1 = df.iloc[15000:30000]
    stream_w2 = df.iloc[30000:45000]

    monitored_features = ["TransactionAmt", "card1", "C1", "C2"]
    windows = [
        ("Window-1: Mid-Cycle Operational Stream", stream_w1),
        ("Window-2: End-of-Cycle Temporal Shift Stream", stream_w2)
    ]

    report_summary = []

    for w_name, w_df in windows:
        print(f"\n🔍 Evaluating {w_name} ({len(w_df):,} transactions)...")
        w_metrics = {}
        max_psi = 0.0
        drifted_count = 0

        for col in monitored_features:
            base_vals = baseline_df[col].dropna().values
            stream_vals = w_df[col].dropna().values

            ks_stat, p_val = ks_2samp(base_vals, stream_vals)
            psi = calculate_psi(base_vals, stream_vals)

            if psi > max_psi:
                max_psi = psi

            is_drift = bool(p_val < 0.05 or psi >= 0.10)
            if is_drift:
                drifted_count += 1

            tier = "STABLE" if psi < 0.10 else ("MODERATE_DRIFT" if psi < 0.25 else "SEVERE_DRIFT_RETRAIN")

            w_metrics[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(p_val), 5),
                "psi_score": round(float(psi), 4),
                "drift_detected": is_drift,
                "severity_tier": tier
            }

            status_symbol = "🟢" if tier == "STABLE" else ("🟡" if tier == "MODERATE_DRIFT" else "🔴")
            print(f"   {status_symbol} {col:<16} | KS: {ks_stat:.4f} (p={p_val:.4f}) | PSI: {psi:.4f} | Status: {tier}")

        overall_status = "STABLE" if max_psi < 0.10 else ("MODERATE_DRIFT_ALERT" if max_psi < 0.25 else "CRITICAL_DRIFT_ALERT")
        recommendation = "✅ Pipeline operating within normal bounds." if max_psi < 0.10 else (
            "🟡 Recalibrate decision thresholds." if max_psi < 0.25 else "🚨 Dispatch automated model retraining pipeline."
        )

        w_entry = {
            "window_name": w_name,
            "window_size": len(w_df),
            "overall_status": overall_status,
            "max_psi": round(max_psi, 4),
            "drifted_features_count": drifted_count,
            "recommended_action": recommendation,
            "feature_metrics": w_metrics
        }
        report_summary.append(w_entry)

    out_file = os.path.join(REPORTS_DIR, "drift_report.json")
    with open(out_file, "w") as f:
        json.dump(report_summary, f, indent=4)
    print(f"\n💾 Concept drift report successfully saved to {out_file}")


if __name__ == "__main__":
    run_drift_monitor()
