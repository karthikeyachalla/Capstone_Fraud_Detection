"""
01_exploratory_data_analysis.py
================================
Capstone Project I — Real-Time Financial Fraud & Credit Risk Intelligence System
Karthikeya Challa | 23WU0104013 | Woxsen University

PURPOSE:
  Load the IEEE-CIS Fraud Detection dataset, understand its structure,
  check class imbalance, identify missing values, and explore key feature distributions.
  This is Step 1 before any model is trained.

RUN:
  python src/01_exploratory_data_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TRAIN_TRANSACTION = os.path.join(DATA_DIR, "train_transaction.csv")
TRAIN_IDENTITY    = os.path.join(DATA_DIR, "train_identity.csv")

def load_data():
    print("Loading transaction data...")
    df_trans = pd.read_csv(TRAIN_TRANSACTION)
    print(f"  Transactions shape: {df_trans.shape}")

    print("Loading identity data...")
    df_id = pd.read_csv(TRAIN_IDENTITY)
    print(f"  Identity shape:     {df_id.shape}")

    print("Merging on TransactionID...")
    df = df_trans.merge(df_id, on="TransactionID", how="left")
    print(f"  Merged shape:       {df.shape}")
    return df

def class_distribution(df):
    print("\n─── Class Distribution ───────────────────────────────")
    counts  = df["isFraud"].value_counts()
    pct     = df["isFraud"].value_counts(normalize=True) * 100
    print(f"  Legitimate (0):  {counts[0]:,}  ({pct[0]:.2f}%)")
    print(f"  Fraudulent  (1): {counts[1]:,}  ({pct[1]:.2f}%)")
    print(f"  Imbalance ratio: {counts[0]/counts[1]:.1f}:1")

def missing_values(df):
    print("\n─── Missing Values (top 20 columns) ──────────────────")
    miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(miss.head(20).to_string())
    high_miss = miss[miss > 50].index.tolist()
    print(f"\n  Columns with >50% missing: {len(high_miss)}")
    return high_miss

def basic_stats(df):
    print("\n─── Transaction Amount Stats ─────────────────────────")
    print(df.groupby("isFraud")["TransactionAmt"].describe().round(2).to_string())

    print("\n─── Product Code Distribution ────────────────────────")
    if "ProductCD" in df.columns:
        print(df.groupby(["ProductCD", "isFraud"]).size().unstack(fill_value=0).to_string())

def card_analysis(df):
    print("\n─── Card Features ────────────────────────────────────")
    for col in ["card4", "card6"]:
        if col in df.columns:
            print(f"\n  {col}:")
            print(df.groupby([col, "isFraud"]).size().unstack(fill_value=0).to_string())

def email_analysis(df):
    print("\n─── Email Domain Analysis ────────────────────────────")
    if "P_emaildomain" in df.columns:
        top_domains = df["P_emaildomain"].value_counts().head(10)
        print("  Top 10 buyer email domains:")
        print(top_domains.to_string())

        fraud_by_domain = df.groupby("P_emaildomain")["isFraud"].mean().sort_values(ascending=False)
        print("\n  Top 10 email domains by fraud rate:")
        print(fraud_by_domain.head(10).to_string())

def time_analysis(df):
    print("\n─── Time-Based Patterns ──────────────────────────────")
    df["hour"] = (df["TransactionDT"] / 3600).astype(int) % 24
    hourly_fraud = df.groupby("hour")["isFraud"].mean()
    peak_hour = hourly_fraud.idxmax()
    print(f"  Peak fraud hour: {peak_hour}:00 (rate: {hourly_fraud[peak_hour]:.3f})")
    print(f"  Lowest fraud hour: {hourly_fraud.idxmin()}:00")

def graph_potential(df):
    print("\n─── Graph Construction Potential ─────────────────────")
    print("  Features available for building transaction graph:")

    graph_features = {
        "card1": "Card number (primary) — high cardinality node connector",
        "card2": "Card number (secondary) — edge between shared cards",
        "P_emaildomain": "Buyer email domain — links transactions via email",
        "R_emaildomain": "Seller email domain — merchant-side connections",
        "DeviceType": "Device type — mobile/desktop clustering",
        "DeviceInfo": "Device fingerprint — specific device identification",
        "addr1": "Billing zip code — geographic clustering",
        "addr2": "Billing country — international fraud clustering",
    }

    for col, desc in graph_features.items():
        if col in df.columns:
            unique_vals = df[col].nunique()
            print(f"  ✅ {col}: {unique_vals:,} unique values — {desc}")
        else:
            print(f"  ❌ {col}: not in dataset")

def main():
    print("=" * 60)
    print("CAPSTONE PROJECT I — Exploratory Data Analysis")
    print("IEEE-CIS Fraud Detection Dataset")
    print("=" * 60)

    df = load_data()
    class_distribution(df)
    high_miss = missing_values(df)
    basic_stats(df)
    card_analysis(df)
    email_analysis(df)
    time_analysis(df)
    graph_potential(df)

    print("\n" + "=" * 60)
    print("✅ EDA Complete.")
    print(f"   Dataset size: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Fraud rate:   {df['isFraud'].mean()*100:.2f}%")
    print(f"   High-miss columns to drop: {len(high_miss)}")
    print("   Ready to proceed to: 02_feature_engineering.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
