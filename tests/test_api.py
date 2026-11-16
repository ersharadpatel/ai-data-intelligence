# Import pytest for automated testing
import pytest

# Import FastAPI TestClient
from fastapi.testclient import TestClient

# Import the FastAPI application
from ai_data_intelligence.api import app
# Import patch to temporarily replace the prediction method
from unittest.mock import patch
# Create a test client for the API
client = TestClient(app)


# Define valid customer input for testing
valid_customer = {
    "age": 32,
    "annual_income": 65000,
    "website_visits": 12,
    "previous_purchases": 4,
    "avg_session_minutes": 8.5,
    "marketing_emails_opened": 6
}


# Test whether the API is running
def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Test whether valid customer data gets a prediction
def test_predict_valid_customer():
    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "purchase_probability" in response.json()["data"]


# Test whether invalid customer category is rejected

def test_predict_invalid_purchase_count():

    customer = valid_customer.copy()
    customer["previous_purchases"] = -1

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


# Test whether negative income is rejected
def test_predict_negative_income():
    customer = valid_customer.copy()
    customer["annual_income"] = -65000

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


# Test whether decimal website visits are rejected
def test_predict_decimal_visits():
    customer = valid_customer.copy()
    customer["website_visits"] = 12.5

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

# Test whether unexpected prediction errors are handled safely

def test_predict_internal_error():

    with patch(
        "ai_data_intelligence.api.trained_model.predict_proba",
        side_effect=ValueError("Sensitive internal test error")
    ):
        response = client.post("/predict", json=valid_customer)

    assert response.status_code == 500

    assert response.json() == {
        "detail": "An internal error occurred while processing the prediction."
    }

    assert "Sensitive internal test error" not in response.text


# Test prediction history endpoint
def test_prediction_history():

    response = client.get("/predictions")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "total_records" in data
    assert "total_pages" in data
    assert isinstance(data["data"], list)


# Test prediction history pagination
def test_prediction_history_pagination():

    response = client.get("/predictions?page=1&limit=2")

    assert response.status_code == 200

    data = response.json()

    assert data["page"] == 1
    assert data["limit"] == 2
    assert len(data["data"]) <= 2



# Test dashboard statistics endpoint
def test_dashboard_statistics():

    response = client.get("/dashboard/stats")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "total_predictions" in data["data"]
    assert "total_buyers" in data["data"]
    assert "total_non_buyers" in data["data"]
    assert "buyer_percentage" in data["data"]
    assert "average_purchase_probability" in data["data"]
    assert "average_processing_time_seconds" in data["data"]


# Test date-wise dashboard analytics
def test_dashboard_analytics():

    response = client.get("/dashboard/analytics")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    analytics = data["data"]

    assert "today" in analytics
    assert "last_7_days" in analytics
    assert "last_30_days" in analytics

    for period in analytics.values():
        assert "total_predictions" in period
        assert "buyers" in period
        assert "non_buyers" in period
        assert "buyer_percentage" in period

        assert period["total_predictions"] == (
            period["buyers"] + period["non_buyers"]
        )


