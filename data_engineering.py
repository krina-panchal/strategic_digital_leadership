import pandas as pd
import numpy as np

np.random.seed(42)

# parameters for small dataset
n_businesses = 30
n_accounts_per_business = 2
n_transactions = 800

# generate business and account ids
business_ids = np.random.choice(range(1000, 1000 + n_businesses), size=n_transactions)
account_ids = [
    f"A{bid}_{np.random.choice(range(n_accounts_per_business))}" for bid in business_ids
]

# generate dates over a year
dates = pd.date_range(start="2024-01-01", end="2024-12-31", periods=n_transactions)

# generate transaction amounts
amounts = np.random.normal(loc=0, scale=200, size=n_transactions).round(2)

# transaction type labels
txn_types = np.where(amounts > 0, "credit", "debit")

# merchant categories
merchants = np.random.choice(
    ["utilities", "salary", "services", "retail", "transport", "misc"],
    size=n_transactions
)

# balance after transaction as a noisy cumulative process
balances = np.cumsum(amounts) + np.random.normal(0, 300, n_transactions)

# assign binary risk label at business level
risk_map = {bid: np.random.choice([0, 1]) for bid in set(business_ids)}
risk_labels = [risk_map[bid] for bid in business_ids]

# assemble dataframe
df = pd.DataFrame({
    "business_id": business_ids,
    "account_id": account_ids,
    "date": dates,
    "amount": amounts,
    "transaction_type": txn_types,
    "merchant_category": merchants,
    "balance_after_transaction": balances.round(2),
    "risk_label": risk_labels
})

# save to CSV
df.to_csv("open_banking_data.csv", index=False)

print(df.head())
print("Saved to synthetic_open_banking_data_small.csv")
