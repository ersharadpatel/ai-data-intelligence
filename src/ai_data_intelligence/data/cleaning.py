
"""Non-destructive customer data cleaning utilities.

The cleaning pipeline preserves the original DataFrame and returns
a cleaned copy with an auditable record of every major operation.
"""

from typing import Any

import pandas as pd

from ai_data_intelligence.data.quality import BUSINESS_RULES


def clean_customer_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean customer records using explicit business constraints.

    Cleaning policy:
    1. Remove exact duplicate records.
    2. Convert invalid predictor values into missing values.
    3. Remove records with missing customer identifiers.
    4. Remove records with missing or invalid target labels.
    5. Impute numeric predictor values using their median.
    6. Preserve non-identical records with repeated customer IDs.

    Returns:
        A tuple containing the cleaned DataFrame and an audit report.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_columns = [
        "customer_id",
        "age",
        "annual_income",
        "website_visits",
        "previous_purchases",
        "avg_session_minutes",
        "marketing_emails_opened",
        "purchased",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # Work on a copy to protect the original raw dataset.
    cleaned_df = df.copy(deep=True)

    audit_records: list[dict[str, Any]] = []

    def record_action(
        operation: str,
        affected_records: int,
        reason: str,
    ) -> None:
        """Append an auditable record for a cleaning operation."""

        audit_records.append({
            "operation": operation,
            "affected_records": int(affected_records),
            "reason": reason,
        })

    # Remove only rows that are identical across every column.
    before_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().copy()

    record_action(
        "remove_exact_duplicates",
        before_count - len(cleaned_df),
        "Removed fully identical records.",
    )

    # Convert values that violate predictor business rules into missing
    # values so that they can be handled by an explicit imputation policy.
    predictor_columns = [
        "age",
        "annual_income",
        "website_visits",
        "previous_purchases",
        "avg_session_minutes",
        "marketing_emails_opened",
    ]

    for column in predictor_columns:
        invalid_condition = BUSINESS_RULES[column]["invalid_condition"]
        invalid_mask = invalid_condition(cleaned_df[column]).fillna(False)

        invalid_count = int(invalid_mask.sum())

        cleaned_df.loc[invalid_mask, column] = pd.NA

        record_action(
            f"invalidate_{column}",
            invalid_count,
            BUSINESS_RULES[column]["description"],
        )

    # A customer record without an identifier cannot be reliably tracked.
    before_count = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=["customer_id"]).copy()

    record_action(
        "remove_missing_customer_ids",
        before_count - len(cleaned_df),
        "Removed records without a customer identifier.",
    )

    # The target label must be trustworthy for supervised learning.
    # We do not invent or impute purchase outcomes.
    target_invalid_mask = (
        BUSINESS_RULES["purchased"]["invalid_condition"](
            cleaned_df["purchased"]
        ).fillna(False)
        | cleaned_df["purchased"].isna()
    )

    invalid_target_count = int(target_invalid_mask.sum())

    cleaned_df = cleaned_df.loc[~target_invalid_mask].copy()

    record_action(
        "remove_invalid_target_records",
        invalid_target_count,
        "Removed rows with missing or invalid purchase labels.",
    )

    # Impute missing predictor values using the median.
    # Median imputation is less sensitive to extreme values than mean
    # imputation, but it can still reduce natural data variability.
    for column in predictor_columns:
        missing_count = int(cleaned_df[column].isna().sum())

        if missing_count == 0:
            continue

        if cleaned_df[column].notna().sum() == 0:
            raise ValueError(
                f"Cannot impute '{column}': no valid values remain."
            )

        median_value = cleaned_df[column].median()

        cleaned_df[column] = cleaned_df[column].fillna(median_value)

        record_action(
            f"median_imputation_{column}",
            missing_count,
            f"Filled missing values using median: {median_value}.",
        )

    # Recheck identifier uniqueness without silently deleting records.
    # Different rows with the same ID may represent different events.
    repeated_id_rows = int(
        cleaned_df["customer_id"].duplicated(keep=False).sum()
    )

    record_action(
        "review_repeated_customer_ids",
        repeated_id_rows,
        "Non-identical records with repeated IDs are preserved for review.",
    )

    # Restore integer types for columns whose business definitions
    # require whole numbers.
    integer_columns = [
        "customer_id",
        "age",
        "website_visits",
        "previous_purchases",
        "marketing_emails_opened",
        "purchased",
    ]

    for column in integer_columns:
        cleaned_df[column] = cleaned_df[column].astype("int64")

    # Reset the index after row removals for a clean output DataFrame.
    cleaned_df = cleaned_df.reset_index(drop=True)

    audit_report = pd.DataFrame(audit_records)

    return cleaned_df, audit_report
