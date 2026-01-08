import os
import pandas as pd
import numpy as np

np.random.seed(42)

# Number of businesses
n_business = 200

industries = [
    "Manufacturing",
    "Retail",
    "Professional services",
    "Construction",
    "Hospitality and leisure",
    "Technology and digital",
    "Transport and logistics"
]

regions = [
    "London",
    "South East",
    "South West",
    "Midlands",
    "North West",
    "North East",
    "Scotland",
    "Wales",
    "Northern Ireland"
]

size_categories = ["Micro", "Small", "Medium"]
turnover_bands = ["Under 500k", "500k to 2m", "2m to 10m", "Over 10m"]

def make_business_id(i: int) -> str:
    return f"BIZ{i:04d}"

# Create business table
business_df = pd.DataFrame({
    "business_id": [make_business_id(i) for i in range(1, n_business + 1)],
    "business_name": [f"Business_{i:04d}" for i in range(1, n_business + 1)],
    "industry": np.random.choice(industries, n_business),
    "size_category": np.random.choice(size_categories, n_business, p=[0.45, 0.35, 0.20]),
    "turnover_band": np.random.choice(turnover_bands, n_business, p=[0.25, 0.35, 0.30, 0.10]),
    "years_trading": np.random.randint(1, 26, n_business),
    "region": np.random.choice(regions, n_business),
})

# Filing timeliness and ESG
business_df["filing_timeliness_score"] = np.round(np.clip(np.random.normal(0.85, 0.12, n_business), 0, 1), 2)
business_df["esg_rating"] = np.round(np.random.uniform(30, 90, n_business), 1)

# Create account table: 1 to 3 accounts per business
account_rows = []
account_types = ["Current", "Savings", "Loan", "Asset finance"]

account_id_counter = 1
for _, row in business_df.iterrows():
    num_accounts = np.random.randint(1, 4)
    for _ in range(num_accounts):
        bal = float(np.round(np.random.normal(50000, 75000), 2))
        # Optional: avoid extreme negatives at inception
        # bal = max(bal, -50000.0)

        account_rows.append({
            "account_id": f"ACC{account_id_counter:05d}",
            "business_id": row["business_id"],
            "account_type": np.random.choice(account_types, p=[0.55, 0.15, 0.20, 0.10]),
            "current_balance": bal
        })
        account_id_counter += 1

account_df = pd.DataFrame(account_rows)

# Create transaction table
transaction_rows = []
transaction_types = ["Credit", "Debit"]
merchant_categories = [
    "Utilities", "Payroll", "Supplier", "Rent", "Tax", "Loan repayment",
    "Card receipts", "Online sales", "Other"
]

transaction_id_counter = 1
for _, acc in account_df.iterrows():
    num_tx = np.random.randint(30, 120)
    balance = float(acc["current_balance"])

    for _ in range(num_tx):
        t_type = np.random.choice(transaction_types, p=[0.45, 0.55])
        amount_abs = float(np.round(abs(np.random.normal(2500, 4000)), 2))

        if t_type == "Debit":
            signed_amount = -amount_abs
        else:
            signed_amount = amount_abs

        balance = float(np.round(balance + signed_amount, 2))

        transaction_rows.append({
            "transaction_id": f"TX{transaction_id_counter:07d}",
            "account_id": acc["account_id"],
            "business_id": acc["business_id"],
            "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="D"),
            "amount": signed_amount,
            "transaction_type": t_type,
            "merchant_category": np.random.choice(merchant_categories),
            "balance_after_transaction": balance
        })
        transaction_id_counter += 1

transaction_df = pd.DataFrame(transaction_rows)

# Aggregate features at business level
transaction_df["month"] = transaction_df["date"].dt.to_period("M")

rev_exp = transaction_df.groupby(["business_id", "month"]).agg(
    monthly_revenue=("amount", lambda x: x[x > 0].sum()),
    monthly_expenses=("amount", lambda x: -x[x < 0].sum())
).reset_index()

features_rows = []
for biz_id, group in rev_exp.groupby("business_id"):
    avg_rev = float(group["monthly_revenue"].mean()) if not group.empty else 0.0
    avg_exp = float(group["monthly_expenses"].mean()) if not group.empty else 0.0
    net_cf = avg_rev - avg_exp
    cf_vol = float(group["monthly_revenue"].std(ddof=0)) if len(group) > 1 else 0.0

    biz_accounts = account_df.loc[account_df["business_id"] == biz_id, "account_id"]
    neg_days = transaction_df.loc[
        transaction_df["account_id"].isin(biz_accounts) & (transaction_df["balance_after_transaction"] < 0),
        :
    ].shape[0]

    debits = transaction_df[(transaction_df["business_id"] == biz_id) & (transaction_df["amount"] < 0)]
    if not debits.empty:
        counts = debits["merchant_category"].value_counts(normalize=True)
        concentration = float((counts ** 2).sum())
    else:
        concentration = 0.0

    features_rows.append({
        "business_id": biz_id,
        "avg_monthly_revenue": round(avg_rev, 2),
        "avg_monthly_expenses": round(avg_exp, 2),
        "avg_net_cashflow": round(net_cf, 2),
        "cashflow_volatility_score": round(cf_vol, 3),
        "days_negative_balance": int(neg_days),
        "payment_concentration_score": round(concentration, 3)
    })

features_df = pd.DataFrame(features_rows)

# Ensure all businesses appear in features table
features_df = business_df[["business_id"]].merge(features_df, on="business_id", how="left").fillna({
    "avg_monthly_revenue": 0.0,
    "avg_monthly_expenses": 0.0,
    "avg_net_cashflow": 0.0,
    "cashflow_volatility_score": 0.0,
    "days_negative_balance": 0,
    "payment_concentration_score": 0.0
})

features_df["days_negative_balance"] = features_df["days_negative_balance"].astype(int)

# Save to CSV files (Windows path)
out_dir = r"C:\Users\kpanchal009\OneDrive - pwc\QMPLUS\Year 4\IOT653U"
os.makedirs(out_dir, exist_ok=True)

business_path = os.path.join(out_dir, "business_table.csv")
account_path = os.path.join(out_dir, "account_table.csv")
transaction_path = os.path.join(out_dir, "transaction_table.csv")
features_path = os.path.join(out_dir, "features_table.csv")

business_df.to_csv(business_path, index=False)
account_df.to_csv(account_path, index=False)
transaction_df.to_csv(transaction_path, index=False)
features_df.to_csv(features_path, index=False)

print("Saved files:")
print(business_path)
print(account_path)
print(transaction_path)
print(features_path)

# Quick preview
print("\nBusiness sample:")
print(business_df.head())

print("\nAccount sample:")
print(account_df.head())

print("\nTransaction sample:")
print(transaction_df.head())

print("\nFeatures sample:")
print(features_df.head())
