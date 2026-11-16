
import pandas as pd
import pytest

from ai_data_intelligence.data.quality import (
    profile_dataset,
    assess_missing_values,
    assess_duplicates,
    validate_business_rules,
    generate_quality_report,
)


@pytest.fixture
def sample_customer_data():
    """Create a controlled dataset for testing data quality functions."""

    return pd.DataFrame({
        "customer_id": [101, 102, 103, 103],
        "age": [25, 35, None, 35],
        "annual_income": [50000, 60000, -1000, 60000],
        "website_visits": [10, 20, -5, 20],
        "previous_purchases": [2, 4, 1, 4],
        "avg_session_minutes": [5.5, 8.0, 3.0, 8.0],
        "marketing_emails_opened": [3, 5, 2, 5],
        "purchased": [1, 0, 1, 0],
    })


def test_profile_dataset(sample_customer_data):
    result = profile_dataset(sample_customer_data)

    assert result["row_count"] == 4
    assert result["column_count"] == 8

    # No complete row is duplicated in this test dataset.
    assert result["duplicate_rows"] == 0



def test_missing_value_assessment(sample_customer_data):
    result = assess_missing_values(sample_customer_data)

    assert result.loc["age", "missing_count"] == 1


def test_duplicate_assessment(sample_customer_data):
    result = assess_duplicates(
        sample_customer_data,
        identifier_column="customer_id",
    )

    # No complete records are identical.
    assert result["duplicate_rows"] == 0

    # Two rows share the same customer identifier.
    assert result["duplicate_identifier_rows"] == 2

    # Only one unique customer identifier is repeated.
    assert result["unique_duplicated_identifiers"] == 1


def test_business_rule_validation(sample_customer_data):
    result = validate_business_rules(sample_customer_data)

    income_result = result.loc[
        result["column"] == "annual_income"
    ].iloc[0]

    assert income_result["invalid_records"] == 1
    assert income_result["status"] == "ISSUES_FOUND"


def test_generate_quality_report(sample_customer_data):
    result = generate_quality_report(
        sample_customer_data,
        identifier_column="customer_id",
    )

    assert "profile" in result
    assert "missing_values" in result
    assert "duplicates" in result
    assert "business_rules" in result


def test_invalid_input_raises_error():
    with pytest.raises(TypeError):
        profile_dataset([1, 2, 3])


def test_missing_identifier_column_raises_error(sample_customer_data):
    with pytest.raises(ValueError):
        assess_duplicates(
            sample_customer_data,
            identifier_column="unknown_id",
        )
