
import pandas as pd
import pytest

from ai_data_intelligence.data.quality_score import (
    calculate_quality_scores,
)


@pytest.fixture
def valid_customer_data():
    """Create a small dataset containing valid customer records."""

    return pd.DataFrame({
        "customer_id": [101, 102, 103],
        "age": [25, 35, 45],
        "annual_income": [50000, 60000, 70000],
        "website_visits": [10, 20, 30],
        "previous_purchases": [2, 4, 6],
        "avg_session_minutes": [5.5, 8.0, 10.0],
        "marketing_emails_opened": [3, 5, 7],
        "purchased": [1, 0, 1],
    })


def test_perfect_dataset_scores(valid_customer_data):
    result = calculate_quality_scores(
        valid_customer_data,
        identifier_column="customer_id",
    )

    assert result["completeness_percentage"] == 100.0
    assert result["validity_percentage"] == 100.0
    assert result["exact_row_uniqueness_percentage"] == 100.0
    assert result["missing_cells"] == 0
    assert result["invalid_values"] == 0
    assert result["exact_duplicate_rows"] == 0
    assert result["duplicate_identifier_rows"] == 0


def test_missing_values_reduce_completeness(valid_customer_data):
    test_df = valid_customer_data.copy()

    test_df.loc[0, "age"] = float("nan")

    result = calculate_quality_scores(test_df)

    assert result["completeness_percentage"] < 100.0
    assert result["missing_cells"] == 1


def test_invalid_values_reduce_validity(valid_customer_data):
    test_df = valid_customer_data.copy()

    test_df.loc[0, "annual_income"] = -5000

    result = calculate_quality_scores(test_df)

    assert result["validity_percentage"] < 100.0
    assert result["invalid_values"] == 1


def test_duplicate_rows_reduce_uniqueness(valid_customer_data):
    test_df = pd.concat(
        [valid_customer_data, valid_customer_data.iloc[[0]]],
        ignore_index=True,
    )

    result = calculate_quality_scores(test_df)

    assert result["exact_row_uniqueness_percentage"] < 100.0
    assert result["exact_duplicate_rows"] == 1


def test_invalid_identifier_column_raises_error(valid_customer_data):
    with pytest.raises(ValueError):
        calculate_quality_scores(
            valid_customer_data,
            identifier_column="unknown_id",
        )


def test_invalid_input_raises_error():
    with pytest.raises(TypeError):
        calculate_quality_scores([1, 2, 3])
