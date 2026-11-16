
# Import Path for handling model file paths
from pathlib import Path

# Import joblib for loading the trained machine learning model
import joblib

# Import pandas for handling customer input data
import pandas as pd


class CustomerPurchasePredictor:
    """Predict customer purchase probability using a trained ML pipeline."""

    def __init__(self, model_path: str, threshold: float = 0.556522):
        # Convert the model path into a Path object
        self.model_path = Path(model_path)

        # Validate the classification threshold
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        # Store the business decision threshold
        self.threshold = threshold

        # Check whether the model file exists
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        # Load the trained machine learning pipeline
        self.model = joblib.load(self.model_path)

    
    def predict(self, customer_data: pd.DataFrame) -> list:
        """Validate customer features and return predictions."""

        # Define the expected input feature columns
        expected_columns = [
            "age",
            "annual_income",
            "website_visits",
            "avg_session_minutes",
            "marketing_emails_opened",
            "customer_engagement_score",
            "purchase_history_segment"
        ]

        # Validate the input data type
        if not isinstance(customer_data, pd.DataFrame):
            raise TypeError("customer_data must be a pandas DataFrame.")

        # Reject empty customer data
        if customer_data.empty:
            raise ValueError("customer_data cannot be empty.")

        # Identify missing and unexpected columns
        missing_columns = set(expected_columns) - set(customer_data.columns)
        unexpected_columns = set(customer_data.columns) - set(expected_columns)

        # Reject missing columns
        if missing_columns:
            raise ValueError(
                f"Missing columns: {sorted(missing_columns)}"
            )

        # Reject unexpected columns
        if unexpected_columns:
            raise ValueError(
                f"Unexpected columns: {sorted(unexpected_columns)}"
            )

        # Arrange columns in the expected order
        customer_data = customer_data[expected_columns].copy()

        # Generate purchase probabilities
        probabilities = self.model.predict_proba(customer_data)[:, 1]

        # Convert probabilities into binary predictions
        predictions = (probabilities >= self.threshold).astype(int)

        # Build structured prediction results
        results = []

        for probability, prediction in zip(probabilities, predictions):
            results.append({
                "purchase_probability": round(float(probability), 4),
                "prediction": int(prediction),
                "decision": (
                    "Likely Buyer" if prediction == 1
                    else "Unlikely Buyer"
                )
            })

        return results

