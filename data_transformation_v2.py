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

def make_business_id(i):
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
        account_rows.append({
            "account_id": f"ACC{account_id_counter:05d}",
            "business_id": row["business_id"],
            "account_type": np.random.choice(account_types, p=[0.55, 0.15, 0.2, 0.1]),
            "current_balance": np.round(np.random.normal(50000, 75000), 2)
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
    balance = acc["current_balance"]
    for _ in range(num_tx):
        t_type = np.random.choice(transaction_types, p=[0.45, 0.55])
        amount = np.round(abs(np.random.normal(2500, 4000)), 2)
        if t_type == "Debit":
            balance_after = balance - amount
            balance = balance_after
        else:
            balance_after = balance + amount
            balance = balance_after

        transaction_rows.append({
            "transaction_id": f"TX{transaction_id_counter:07d}",
            "account_id": acc["account_id"],
            "business_id": acc["business_id"],
            "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="D"),
            "amount": amount if t_type == "Credit" else -amount,
            "transaction_type": t_type,
            "merchant_category": np.random.choice(merchant_categories),
            "balance_after_transaction": round(balance_after, 2)
        })
        transaction_id_counter += 1

transaction_df = pd.DataFrame(transaction_rows)

# Aggregate features at business level
# Compute monthly revenue and expenses approximations from transactions
transaction_df["month"] = transaction_df["date"].dt.to_period("M")

# Approximate "revenue" as positive cash inflows, "expenses" as negative outflows
rev_exp = transaction_df.groupby(["business_id", "month"]).agg(
    monthly_revenue=("amount", lambda x: x[x > 0].sum()),
    monthly_expenses=("amount", lambda x: -x[x < 0].sum())
).reset_index()

features_rows = []
for biz_id, group in rev_exp.groupby("business_id"):
    avg_rev = group["monthly_revenue"].mean()
    avg_exp = group["monthly_expenses"].mean()
    net_cf = avg_rev - avg_exp
    cf_vol = group["monthly_revenue"].std(ddof=0) if len(group) > 1 else 0.0

    # Days in negative balance approximation
    biz_accounts = account_df[account_df["business_id"] == biz_id]["account_id"]
    neg_days = transaction_df[transaction_df["account_id"].isin(biz_accounts)]
    neg_days = neg_days[neg_days["balance_after_transaction"] < 0].shape[0]

    # Payment concentration score: index like measure on merchant categories for debits
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
missing_biz = set(business_df["business_id"]) - set(features_df["business_id"])
for biz in missing_biz:
    features_df = pd.concat([
        features_df,
        pd.DataFrame([{
            "business_id": biz,
            "avg_monthly_revenue": 0.0,
            "avg_monthly_expenses": 0.0,
            "avg_net_cashflow": 0.0,
            "cashflow_volatility_score": 0.0,
            "days_negative_balance": 0,
            "payment_concentration_score": 0.0
        }])
    ], ignore_index=True)

# Save to CSV
business_path = "/mnt/data/business_table.csv"
account_path = "/mnt/data/account_table.csv"
transaction_path = "/mnt/data/transaction_table.csv"
features_path = "/mnt/data/features_table.csv"

business_df.to_csv(business_path, index=False)
account_df.to_csv(account_path, index=False)
transaction_df.to_csv(transaction_path, index=False)
features_df.to_csv(features_path, index=False)

business_df.head(), account_df.head(), transaction_df.head(), features_df.head(), (business_path, account_path, transaction_path, features_path)
