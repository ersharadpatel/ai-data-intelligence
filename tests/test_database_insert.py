# Import the database session factory
from ai_data_intelligence.database import SessionLocal

# Import the prediction history database model
from ai_data_intelligence.database_models import PredictionHistory


# Test inserting a prediction record into the database
def test_insert_prediction_record():

    # Create a new database session
    db = SessionLocal()

    try:

        # Create a sample prediction record
        prediction = PredictionHistory(
            request_id="test-request-001",
            age=32,
            annual_income=65000,
            purchase_probability=0.6161,
            prediction=1,
            decision="Likely Buyer",
            processing_time=0.0611
        )

        # Add the prediction record to the session
        db.add(prediction)

        # Save the record permanently in the database
        db.commit()

        # Refresh the object to get its database-generated ID
        db.refresh(prediction)

        # Verify that the record has a database ID
        assert prediction.id is not None

        # Verify the saved prediction decision
        assert prediction.decision == "Likely Buyer"

    finally:

        # Always close the database session
        db.close()