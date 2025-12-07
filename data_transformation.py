# 1. Library imports and configuration

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# 2. Data ingestion and initial cleaning

# Read synthetic open banking style data
# Expected columns:
# business_id, account_id, date, amount, transaction_type,
# merchant_category, balance_after_transaction, risk_label

raw_df = pd.read_csv("open_banking_data.csv")

# Basic type conversions
raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")

# Remove rows with invalid or missing dates
raw_df = raw_df.dropna(subset=["date"])

# Normalise transaction type strings
raw_df["transaction_type"] = (
    raw_df["transaction_type"]
    .str.strip()
    .str.lower()
)

# Normalise merchant category strings
raw_df["merchant_category"] = (
    raw_df["merchant_category"]
    .str.strip()
    .str.lower()
)

# Remove obviously erroneous transactions
# For example extremely large values that exceed domain expectations
upper_amount_quantile = raw_df["amount"].quantile(0.999)
raw_df = raw_df[raw_df["amount"].between(-upper_amount_quantile, upper_amount_quantile)]
