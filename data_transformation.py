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


# 3. Derived fields at transaction level

# Create debit credit indicator
raw_df["is_credit"] = np.where(raw_df["amount"] > 0, 1, 0)
raw_df["is_debit"] = np.where(raw_df["amount"] < 0, 1, 0)

# Month key for later aggregation
raw_df["year_month"] = raw_df["date"].dt.to_period("M").astype(str)

# Days since previous transaction per business
raw_df = raw_df.sort_values(["business_id", "date"])
raw_df["prev_date"] = raw_df.groupby("business_id")["date"].shift(1)
raw_df["days_since_prev_txn"] = (
    raw_df["date"] - raw_df["prev_date"]
).dt.days.fillna(0)


# 4. Feature engineering at business level

# Helper functions for aggregation

def pct_negative_balances(series: pd.Series) -> float:
    return np.mean(series < 0)


def concentration_index(series: pd.Series) -> float:
    """
    Very simple Hirschman Herfindahl like index for payment concentration.
    """
    abs_vals = series.abs()
    total = abs_vals.sum()
    if total == 0:
        return 0.0
    shares = abs_vals / total
    return np.square(shares).sum()


# Aggregate to monthly level first
monthly = (
    raw_df
    .groupby(["business_id", "year_month"])
    .agg(
        monthly_revenue=("amount", lambda x: x[x > 0].sum()),
        monthly_expenses=("amount", lambda x: -x[x < 0].sum()),
        net_cashflow=("amount", "sum"),
        avg_balance=("balance_after_transaction", "mean"),
        min_balance=("balance_after_transaction", "min"),
        max_balance=("balance_after_transaction", "max"),
        pct_negative_balance=("balance_after_transaction", pct_negative_balances),
        payment_concentration=("amount", concentration_index),
        avg_days_between_txn=("days_since_prev_txn", "mean"),
        txn_count=("amount", "count"),
    )
    .reset_index()
)

# Now aggregate to business level features
business_features = (
    monthly
    .groupby("business_id")
    .agg(
        avg_monthly_revenue=("monthly_revenue", "mean"),
        avg_monthly_expenses=("monthly_expenses", "mean"),
        avg_net_cashflow=("net_cashflow", "mean"),
        cashflow_volatility=("net_cashflow", "std"),
        liquidity_buffer=("avg_balance", "mean"),
        worst_drawdown=("min_balance", "min"),
        pct_negative_balance_mean=("pct_negative_balance", "mean"),
        payment_concentration_mean=("payment_concentration", "mean"),
        avg_days_between_txn=("avg_days_between_txn", "mean"),
        avg_txn_count=("txn_count", "mean"),
    )
    .reset_index()
)

# Replace missing values that arise from variance computations
business_features = business_features.fillna(0.0)

# Attach risk label at business level for validation purposes
risk_labels = (
    raw_df[["business_id", "risk_label"]]
    .drop_duplicates(subset="business_id")
    .set_index("business_id")
)

business_features = (
    business_features
    .set_index("business_id")
    .join(risk_labels, how="inner")
    .reset_index()
)

business_features.head()

# 5. Train test split and standardisation
#    (for validation only, not for deployment)

X = business_features.drop(columns=["business_id", "risk_label"])
y = business_features["risk_label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 6. Logistic regression model for validation

log_reg = LogisticRegression(max_iter=1000)

log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

print("Logistic regression validation report")
print(classification_report(y_test, y_pred))
print("Area under ROC curve:", roc_auc_score(y_test, y_prob))


# 7. Random forest model for sensitivity analysis

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Random forest validation report")
print(classification_report(y_test, y_pred_rf))
print("Area under ROC curve:", roc_auc_score(y_test, y_prob_rf))

# Simple feature importance inspection
feature_importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("Top ten features by importance")
print(feature_importances.head(10))
