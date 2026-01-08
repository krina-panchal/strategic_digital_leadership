import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

OUTPUT_PATH = "./"

# -----------------------------
# Dimension Tables
# -----------------------------

def create_dim_date():
    dates = pd.date_range(start="2015-01-01", end="2025-12-31", freq="D")
    df = pd.DataFrame({
        "date_id": range(1, len(dates) + 1),
        "date": dates,
        "year": dates.year,
        "quarter": dates.quarter,
        "month": dates.month,
        "month_name": dates.strftime("%B")
    })
    df.to_csv(f"{OUTPUT_PATH}dim_date.csv", index=False)


def create_dim_region():
    regions = ["UK", "Europe", "North America", "Asia Pacific"]
    df = pd.DataFrame({
        "region_id": range(1, len(regions) + 1),
        "region_name": regions
    })
    df.to_csv(f"{OUTPUT_PATH}dim_region.csv", index=False)


def create_dim_sector():
    sectors = ["Technology", "Healthcare", "Financial Services", "Retail", "Energy"]
    df = pd.DataFrame({
        "sector_id": range(1, len(sectors) + 1),
        "sector_name": sectors
    })
    df.to_csv(f"{OUTPUT_PATH}dim_sector.csv", index=False)


# -----------------------------
# Entity Tables
# -----------------------------

def create_sme():
    df = pd.DataFrame({
        "sme_id": range(1, 201),
        "company_name": [f"SME_{i}" for i in range(1, 201)],
        "sector_id": np.random.randint(1, 6, 200),
        "region_id": np.random.randint(1, 5, 200),
        "employee_count": np.random.randint(20, 1000, 200)
    })
    df.to_csv(f"{OUTPUT_PATH}sme.csv", index=False)


def create_leadership_profiles():
    df = pd.DataFrame({
        "sme_id": range(1, 201),
        "ceo_tenure_years": np.round(np.random.uniform(1, 15, 200), 1),
        "board_stability_score": np.round(np.random.uniform(0.4, 1.0, 200), 2),
        "prior_exit_experience": np.random.choice([0, 1], 200)
    })
    df.to_csv(f"{OUTPUT_PATH}leadership_profiles.csv", index=False)


# -----------------------------
# Fact Tables
# -----------------------------

def create_financials():
    df = pd.DataFrame({
        "sme_id": np.repeat(range(1, 201), 5),
        "year": list(range(2020, 2025)) * 200,
        "revenue": np.random.normal(10_000_000, 3_000_000, 1000).round(0),
        "ebitda_margin": np.round(np.random.uniform(0.05, 0.35, 1000), 2),
        "debt_to_equity": np.round(np.random.uniform(0.1, 2.5, 1000), 2)
    })
    df.to_csv(f"{OUTPUT_PATH}financials.csv", index=False)


def create_transactions():
    df = pd.DataFrame({
        "transaction_id": range(1, 401),
        "sme_id": np.random.randint(1, 201, 400),
        "deal_value_m": np.round(np.random.uniform(5, 500, 400), 1),
        "deal_type": np.random.choice(["Minority", "Majority", "Full Buyout"], 400),
        "deal_success": np.random.choice([0, 1], 400, p=[0.35, 0.65])
    })
    df.to_csv(f"{OUTPUT_PATH}transactions.csv", index=False)


def create_credit_risk_scores():
    df = pd.DataFrame({
        "sme_id": range(1, 201),
        "credit_score": np.random.randint(300, 850, 200),
        "probability_of_default": np.round(np.random.uniform(0.01, 0.25, 200), 3)
    })
    df.to_csv(f"{OUTPUT_PATH}credit_risk_scores.csv", index=False)


def create_esg_scores():
    df = pd.DataFrame({
        "sme_id": range(1, 201),
        "environment_score": np.round(np.random.uniform(40, 90, 200), 1),
        "social_score": np.round(np.random.uniform(45, 95, 200), 1),
        "governance_score": np.round(np.random.uniform(50, 95, 200), 1)
    })
    df["esg_overall"] = df[[
        "environment_score",
        "social_score",
        "governance_score"
    ]].mean(axis=1).round(1)

    df.to_csv(f"{OUTPUT_PATH}esg_scores.csv", index=False)


def create_features_snapshot():
    df = pd.DataFrame({
        "sme_id": range(1, 201),
        "avg_revenue_growth": np.round(np.random.uniform(-0.05, 0.35, 200), 3),
        "avg_ebitda_margin": np.round(np.random.uniform(0.1, 0.3, 200), 2),
        "esg_overall": np.round(np.random.uniform(50, 90, 200), 1),
        "credit_score": np.random.randint(300, 850, 200),
        "deal_success_label": np.random.choice([0, 1], 200)
    })
    df.to_csv(f"{OUTPUT_PATH}features_snapshot.csv", index=False)


# -----------------------------
# Run All Generators
# -----------------------------

def main():
    create_dim_date()
    create_dim_region()
    create_dim_sector()
    create_sme()
    create_leadership_profiles()
    create_financials()
    create_transactions()
    create_credit_risk_scores()
    create_esg_scores()
    create_features_snapshot()


if __name__ == "__main__":
    main()
