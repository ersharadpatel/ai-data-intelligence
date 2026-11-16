
"""Dataset quality scoring utilities."""

from typing import Any

import pandas as pd

from ai_data_intelligence.data.quality import (
    BUSINESS_RULES,
)


def calculate_quality_scores(
    df: pd.DataFrame,
    identifier_column: str | None = None,
) -> dict[str, Any]:
    """Calculate completeness, validity, and exact-row uniqueness scores."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    total_cells = df.size

    # Completeness measures the proportion of non-missing cells.
    completeness = (
        (1 - df.isna().sum().sum() / total_cells) * 100
        if total_cells > 0 else 100.0
    )

    # Validate business constraints only for rules whose columns exist.
    total_checked_values = 0
    total_invalid_values = 0

    for column, rule in BUSINESS_RULES.items():

        if column not in df.columns:
            continue

        # Missing values are counted separately from invalid values.
        non_missing = df[column].notna()
        invalid_mask = (
            rule["invalid_condition"](df[column]).fillna(False)
            & non_missing
        )

        total_checked_values += int(non_missing.sum())
        total_invalid_values += int(invalid_mask.sum())

    validity = (
        (1 - total_invalid_values / total_checked_values) * 100
        if total_checked_values > 0 else 100.0
    )

    # Exact-row uniqueness is distinct from identifier uniqueness.
    duplicate_count = int(df.duplicated().sum())

    uniqueness = (
        (1 - duplicate_count / len(df)) * 100
        if len(df) > 0 else 100.0
    )

    scores = {
        "completeness_percentage": round(float(completeness), 2),
        "validity_percentage": round(float(validity), 2),
        "exact_row_uniqueness_percentage": round(float(uniqueness), 2),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "invalid_values": total_invalid_values,
        "exact_duplicate_rows": duplicate_count,
    }

    if identifier_column is not None:
        if identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' does not exist."
            )

        non_missing_ids = df[identifier_column].dropna()

        scores["duplicate_identifier_rows"] = int(
            non_missing_ids.duplicated(keep=False).sum()
        )

    return scores
