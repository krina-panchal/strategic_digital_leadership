from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationConfig:
    output_dir: Path = Path(".")
    strict: bool = True  # if False, allow extra columns beyond schema


# ---------------------------------------------------------------------
# Schemas (expected columns and their basic dtypes)
# Note: Pandas dtypes can vary on read; we validate "compatibility" rather than exact dtype.
# ---------------------------------------------------------------------

SCHEMAS: Dict[str, List[str]] = {
    "dim_date.csv": ["date_id", "date", "year", "quarter", "month", "month_name"],
    "dim_region.csv": ["region_id", "region_name"],
    "dim_sector.csv": ["sector_id", "sector_name"],
    "sme.csv": ["sme_id", "company_name", "sector_id", "region_id", "employee_count"],
    "leadership_profiles.csv": ["sme_id", "ceo_tenure_years", "board_stability_score", "prior_exit_experience"],
    "financials.csv": ["sme_id", "year", "revenue", "ebitda_margin", "debt_to_equity"],
    "transactions.csv": ["transaction_id", "sme_id", "deal_value_m", "deal_type", "deal_success"],
    "credit_risk_scores.csv": ["sme_id", "credit_score", "probability_of_default"],
    "esg_scores.csv": [
        "sme_id",
        "environment_score",
        "social_score",
        "governance_score",
        "esg_overall",
    ],
    "features_snapshot.csv": [
        "sme_id",
        "avg_revenue_growth",
        "avg_ebitda_margin",
        "esg_overall",
        "credit_score",
        "deal_success_label",
    ],
}

ALLOWED_DEAL_TYPES = {"Minority", "Majority", "Full Buyout"}


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------

class ValidationError(Exception):
    pass


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _err(errors: List[str], message: str) -> None:
    errors.append(message)


def _require_files_exist(cfg: ValidationConfig, filenames: Iterable[str]) -> List[str]:
    errors: List[str] = []
    for name in filenames:
        path = cfg.output_dir / name
        if not path.exists():
            _err(errors, f"Missing file: {path}")
    return errors


def _read_csv(path: Path) -> pd.DataFrame:
    # Keep defaults; parsing done per-table where needed
    return pd.read_csv(path)


def _validate_schema(
    df: pd.DataFrame,
    filename: str,
    cfg: ValidationConfig,
    errors: List[str],
) -> None:
    expected = SCHEMAS[filename]
    actual = list(df.columns)

    missing = [c for c in expected if c not in df.columns]
    if missing:
        _err(errors, f"{filename}: missing columns: {missing}")

    if cfg.strict:
        extra = [c for c in actual if c not in expected]
        if extra:
            _err(errors, f"{filename}: unexpected extra columns: {extra}")


def _validate_not_null(
    df: pd.DataFrame,
    filename: str,
    columns: Iterable[str],
    errors: List[str],
) -> None:
    for col in columns:
        if col in df.columns and df[col].isna().any():
            n = int(df[col].isna().sum())
            _err(errors, f"{filename}: column '{col}' contains {n} null value(s)")


def _validate_unique(
    df: pd.DataFrame,
    filename: str,
    column: str,
    errors: List[str],
) -> None:
    if column not in df.columns:
        return
    dup = df[df[column].duplicated(keep=False)][column]
    if not dup.empty:
        examples = dup.head(5).tolist()
        _err(errors, f"{filename}: '{column}' not unique. Example duplicate ids: {examples}")


def _validate_fk_in_set(
    df: pd.DataFrame,
    filename: str,
    fk_col: str,
    valid_values: set,
    errors: List[str],
) -> None:
    if fk_col not in df.columns:
        return
    bad = df.loc[~df[fk_col].isin(valid_values), fk_col]
    if not bad.empty:
        examples = bad.head(5).tolist()
        _err(errors, f"{filename}: '{fk_col}' contains invalid FK(s). Examples: {examples}")


def _validate_numeric_range(
    df: pd.DataFrame,
    filename: str,
    col: str,
    min_value: Optional[float],
    max_value: Optional[float],
    errors: List[str],
    allow_null: bool = False,
) -> None:
    if col not in df.columns:
        return

    series = pd.to_numeric(df[col], errors="coerce")
    if not allow_null and series.isna().any():
        n = int(series.isna().sum())
        _err(errors, f"{filename}: '{col}' has {n} non-numeric or null value(s)")
        return

    non_null = series.dropna()
    if min_value is not None:
        bad = non_null[non_null < min_value]
        if not bad.empty:
            _err(errors, f"{filename}: '{col}' has values < {min_value}. Example: {float(bad.iloc[0])}")
    if max_value is not None:
        bad = non_null[non_null > max_value]
        if not bad.empty:
            _err(errors, f"{filename}: '{col}' has values > {max_value}. Example: {float(bad.iloc[0])}")


def _validate_allowed_values(
    df: pd.DataFrame,
    filename: str,
    col: str,
    allowed: set,
    errors: List[str],
) -> None:
    if col not in df.columns:
        return
    bad = df.loc[~df[col].isin(allowed), col]
    if not bad.empty:
        examples = bad.head(5).tolist()
        _err(errors, f"{filename}: '{col}' has disallowed value(s). Examples: {examples}")


def _validate_esg_overall_mean(
    df: pd.DataFrame,
    filename: str,
    errors: List[str],
    tolerance: float = 0.2,
) -> None:
    # Tolerance accounts for rounding differences.
    required = {"environment_score", "social_score", "governance_score", "esg_overall"}
    if not required.issubset(df.columns):
        return

    comps = df[["environment_score", "social_score", "governance_score"]].apply(pd.to_numeric, errors="coerce")
    overall = pd.to_numeric(df["esg_overall"], errors="coerce")

    expected = comps.mean(axis=1)
    diff = (overall - expected).abs()

    bad = diff[diff > tolerance]
    if not bad.empty:
        idx = int(bad.index[0])
        _err(
            errors,
            f"{filename}: esg_overall not consistent with component mean at row index {idx}. "
            f"abs_diff={float(bad.iloc[0]):.3f} (tolerance {tolerance})",
        )


# ---------------------------------------------------------------------
# Table-specific validations
# ---------------------------------------------------------------------

def _validate_dim_date(df: pd.DataFrame, errors: List[str]) -> None:
    filename = "dim_date.csv"
    _validate_unique(df, filename, "date_id", errors)
    _validate_not_null(df, filename, ["date_id", "date"], errors)

    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        if parsed.isna().any():
            n = int(parsed.isna().sum())
            _err(errors, f"{filename}: 'date' contains {n} unparsable date(s)")

    _validate_numeric_range(df, filename, "month", 1, 12, errors)
    _validate_numeric_range(df, filename, "quarter", 1, 4, errors)


def _validate_dim_region(df: pd.DataFrame, errors: List[str]) -> None:
    filename = "dim_region.csv"
    _validate_unique(df, filename, "region_id", errors)
    _validate_not_null(df, filename, ["region_id", "region_name"], errors)


def _validate_dim_sector(df: pd.DataFrame, errors: List[str]) -> None:
    filename = "dim_sector.csv"
    _validate_unique(df, filename, "sector_id", errors)
    _validate_not_null(df, filename, ["sector_id", "sector_name"], errors)


def _validate_sme(df: pd.DataFrame, errors: List[str]) -> None:
    filename = "sme.csv"
    _validate_unique(df, filename, "sme_id", errors)
    _validate_not_null(df, filename, ["sme_id", "company_name", "sector_id", "region_id"], errors)
    _validate_numeric_range(df, filename, "employee_count", 1, None, errors)


def _validate_leadership(df: pd.DataFrame, errors: List[str]) -> None:
    filename = "leadership_profiles.csv"
    _validate_not_null(df, filename, ["sme_id"], errors)
    _validate_numeric_range(df, filename, "ceo_tenure_years", 0, 60, errors)
    _validate_numeric_range(df, filename, "board_stability_score", 0, 1, errors)
    _validate_allowed_values(df, filename, "prior_exit_experience", {0, 1}, errors)


def _validate_financials(df: pd.DataFrame, errors: List[str]) -> None:
    filename = "financials.csv"
    _validate_not_null(df, filename, ["sme_id", "year"], errors)
    _validate_numeric_range(df, filename, "year", 1900, 2100, errors)
    _validate_numeric_range(df, filename, "revenue", 0, None, errors)
    _validate_numeric_range(df, filename, "ebitda_margin", -1, 1, errors)  # allow negatives in edge cases
    _validate_numeric_range(df, filename, "debt_to_equity", 0, None, errors)


def _validate_transactions(df: pd.DataFrame, errors: List[str]) -> None:
    filename = "transactions.csv"
    _validate_unique(df, filename, "transaction_id", errors)
    _validate_not_null(df, filename, ["transaction_id", "sme_id", "deal_success"], errors)
    _validate_numeric_range(df, filename, "deal_value_m", 0, None, errors)
    _validate_allowed_values(df, filename, "deal_type", ALLOWED_DEAL_TYPES, errors)
    _validate_allowed_values(df, filename, "deal_success", {0, 1}, errors)


def _validate_credit_risk(df: pd.DataFrame, errors: List[str]) -> None:
    filename = "credit_risk_scores.csv"
    _validate_not_null(df, filename, ["sme_id", "credit_score"], errors)
    _validate_numeric_range(df, filename, "credit_score", 300, 850, errors)
    _validate_numeric_range(df, filename, "probability_of_default", 0, 1, errors)


def _validate_esg(df: pd.DataFrame, errors: List[str]) -> None:
    filename = "esg_scores.csv"
    _validate_not_null(df, filename, ["sme_id"], errors)
    _validate_numeric_range(df, filename, "environment_score", 0, 100, errors)
    _validate_numeric_range(df, filename, "social_score", 0, 100, errors)
    _validate_numeric_range(df, filename, "governance_score", 0, 100, errors)
    _validate_numeric_range(df, filename, "esg_overall", 0, 100, errors)
    _validate_esg_overall_mean(df, filename, errors, tolerance=0.25)


def _validate_features_snapshot(df: pd.DataFrame, errors: List[str]) -> None:
    filename = "features_snapshot.csv"
    _validate_unique(df, filename, "sme_id", errors)
    _validate_not_null(df, filename, ["sme_id"], errors)
    _validate_numeric_range(df, filename, "avg_revenue_growth", -1, 5, errors)
    _validate_numeric_range(df, filename, "avg_ebitda_margin", -1, 1, errors)
    _validate_numeric_range(df, filename, "esg_overall", 0, 100, errors)
    _validate_numeric_range(df, filename, "credit_score", 300, 850, errors)
    _validate_allowed_values(df, filename, "deal_success_label", {0, 1}, errors)


# ---------------------------------------------------------------------
# Cross-table relational checks
# ---------------------------------------------------------------------

def _validate_relations(tables: Dict[str, pd.DataFrame], errors: List[str]) -> None:
    sme = tables.get("sme.csv")
    dim_region = tables.get("dim_region.csv")
    dim_sector = tables.get("dim_sector.csv")

    if sme is None or dim_region is None or dim_sector is None:
        return

    sme_ids = set(sme["sme_id"].dropna().astype(int).tolist()) if "sme_id" in sme.columns else set()
    region_ids = set(dim_region["region_id"].dropna().astype(int).tolist()) if "region_id" in dim_region.columns else set()
    sector_ids = set(dim_sector["sector_id"].dropna().astype(int).tolist()) if "sector_id" in dim_sector.columns else set()

    # SME foreign keys
    _validate_fk_in_set(sme, "sme.csv", "region_id", region_ids, errors)
    _validate_fk_in_set(sme, "sme.csv", "sector_id", sector_ids, errors)

    # Other tables FK to SME
    for fname, df in tables.items():
        if fname == "sme.csv":
            continue
        if "sme_id" in df.columns:
            _validate_fk_in_set(df, fname, "sme_id", sme_ids, errors)


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------

def validate_all_csvs(cfg: ValidationConfig) -> None:
    filenames = list(SCHEMAS.keys())

    errors = _require_files_exist(cfg, filenames)
    if errors:
        raise ValidationError("\n".join(errors))

    # Load and schema validate
    tables: Dict[str, pd.DataFrame] = {}
    for fname in filenames:
        df = _read_csv(cfg.output_dir / fname)
        tables[fname] = df
        _validate_schema(df, fname, cfg, errors)

    # Table-level checks
    if "dim_date.csv" in tables:
        _validate_dim_date(tables["dim_date.csv"], errors)
    if "dim_region.csv" in tables:
        _validate_dim_region(tables["dim_region.csv"], errors)
    if "dim_sector.csv" in tables:
        _validate_dim_sector(tables["dim_sector.csv"], errors)
    if "sme.csv" in tables:
        _validate_sme(tables["sme.csv"], errors)
    if "leadership_profiles.csv" in tables:
        _validate_leadership(tables["leadership_profiles.csv"], errors)
    if "financials.csv" in tables:
        _validate_financials(tables["financials.csv"], errors)
    if "transactions.csv" in tables:
        _validate_transactions(tables["transactions.csv"], errors)
    if "credit_risk_scores.csv" in tables:
        _validate_credit_risk(tables["credit_risk_scores.csv"], errors)
    if "esg_scores.csv" in tables:
        _validate_esg(tables["esg_scores.csv"], errors)
    if "features_snapshot.csv" in tables:
        _validate_features_snapshot(tables["features_snapshot.csv"], errors)

    # Cross-table checks
    _validate_relations(tables, errors)

    if errors:
        raise ValidationError("Validation failed:\n" + "\n".join(f"- {e}" for e in errors))


if __name__ == "__main__":
    # Example usage:
    # Put this file next to your generated CSVs, or set output_dir to the folder containing them.
    config = ValidationConfig(output_dir=Path("./"), strict=True)
    validate_all_csvs(config)
    print("All CSV validations passed.")
