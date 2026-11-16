
"""Reusable data quality assessment utilities.

This module provides non-destructive profiling and validation
functions for structured customer datasets.
"""

from typing import Any

import pandas as pd


# Business constraints are centralized so they can be maintained
# consistently across notebooks, APIs, and future application services.
BUSINESS_RULES = {
    "age": {
        "description": "Customer age must be between 18 and 100.",
        "invalid_condition": lambda series: (series < 18) | (series > 100),
    },
    "annual_income": {
        "description": "Annual income must be greater than zero.",
        "invalid_condition": lambda series: series <= 0,
    },
    "website_visits": {
        "description": "Website visits cannot be negative.",
        "invalid_condition": lambda series: series < 0,
    },
    "previous_purchases": {
        "description": "Previous purchases cannot be negative.",
        "invalid_condition": lambda series: series < 0,
    },
    "avg_session_minutes": {
        "description": "Average session duration cannot be negative.",
        "invalid_condition": lambda series: series < 0,
    },
    "marketing_emails_opened": {
        "description": "Opened emails must be between 0 and 10.",
        "invalid_condition": lambda series: (series < 0) | (series > 10),
    },
    "purchased": {
        "description": "Purchase target must be either 0 or 1.",
        "invalid_condition": lambda series: ~series.isin([0, 1]),
    },
}


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return basic structural and statistical information about a dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def assess_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing-value counts and percentages for every column."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    total_rows = len(df)

    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna().sum().div(total_rows).mul(100)
            if total_rows > 0
            else 0
        ),
    })

    return report.round(2)


def assess_duplicates(
    df: pd.DataFrame,
    identifier_column: str | None = None,
) -> dict[str, Any]:
    """Assess exact duplicate rows and repeated identifier values."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    result = {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_including_first_occurrence": int(
            df.duplicated(keep=False).sum()
        ),
    }

    if identifier_column is not None:
        if identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' does not exist."
            )

        repeated_mask = df[identifier_column].duplicated(keep=False)

        result.update({
            "identifier_column": identifier_column,
            "duplicate_identifier_rows": int(repeated_mask.sum()),
            "unique_duplicated_identifiers": int(
                df.loc[repeated_mask, identifier_column].nunique()
            ),
        })

    return result


def validate_business_rules(
    df: pd.DataFrame,
    rules: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Evaluate business constraints and report invalid records."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    active_rules = rules if rules is not None else BUSINESS_RULES
    results = []

    for column, rule in active_rules.items():

        if column not in df.columns:
            results.append({
                "column": column,
                "business_rule": rule["description"],
                "invalid_records": None,
                "affected_rows_percentage": None,
                "status": "COLUMN_MISSING",
            })
            continue

        invalid_mask = rule["invalid_condition"](df[column]).fillna(False)
        invalid_count = int(invalid_mask.sum())

        percentage = (
            invalid_count / len(df) * 100
            if len(df) > 0
            else 0
        )

        results.append({
            "column": column,
            "business_rule": rule["description"],
            "invalid_records": invalid_count,
            "affected_rows_percentage": round(percentage, 2),
            "status": "ISSUES_FOUND" if invalid_count else "PASS",
        })

    return pd.DataFrame(results)


def generate_quality_report(
    df: pd.DataFrame,
    identifier_column: str | None = None,
) -> dict[str, Any]:
    """Generate a consolidated, non-destructive dataset quality report."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    return {
        "profile": profile_dataset(df),
        "missing_values": assess_missing_values(df),
        "duplicates": assess_duplicates(df, identifier_column),
        "business_rules": validate_business_rules(df),
    }
