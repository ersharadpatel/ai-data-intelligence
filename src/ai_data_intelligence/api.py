
# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field

# Standard library imports
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
import io
import logging
import time
import uuid

# Third-party imports
import joblib
import pandas as pd
from sqlalchemy import desc, func
from ai_data_intelligence.database import engine
from ai_data_intelligence.database_models import (
    Base,
    ModelRegistry,
    TrainingExperiment,
)
# Application imports
from ai_data_intelligence.data_quality import DataQualityAnalyzer
from ai_data_intelligence.database import SessionLocal
from ai_data_intelligence.database_models import (
    DatasetQualityReport,
    PredictionHistory,
)
from ai_data_intelligence.data_cleaner import DataCleaner
from ai_data_intelligence.feature_engineering import FeatureEngineer
from ai_data_intelligence.training import ModelTrainer
from typing import Any
from ai_data_intelligence.database_models import TrainingExperiment
# Define the directory for uploaded datasets
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Public-demo friendly upload limits (also enforced in the browser UI)
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_UPLOAD_ROWS = 5000
MAX_UPLOAD_COLUMNS = 40
ALLOWED_UPLOAD_EXTENSIONS = {".csv"}
# Configure the application logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create a logger for this API module
logger = logging.getLogger(__name__)

# Create newly introduced registry table without dropping existing tables.
Base.metadata.create_all(bind=engine)

# Create a context variable for the current request ID
request_id_context = ContextVar("request_id", default="N/A")
# Create the FastAPI application
app = FastAPI(
    title="AI Data Intelligence API",
    description="Customer purchase prediction service",
    version="1.0.0"
)

# Define middleware to track every incoming API request
@app.middleware("http")
async def track_request(request, call_next):

    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())

    # Store the request ID and keep a reset token
    context_token = request_id_context.set(request_id)

    try:

        # Log the incoming request
        logger.info(
            "Request received | ID: %s | Method: %s | Path: %s",
            request_id,
            request.method,
            request.url.path
        )

        # Process the request and get the response
        response = await call_next(request)

        # Attach the request ID to the response headers
        response.headers["X-Request-ID"] = request_id

        # Log the completed request
        logger.info(
            "Request completed | ID: %s | Status: %s",
            request_id,
            response.status_code
        )

        # Return the response to the client
        return response

    finally:

        # Always reset the request-specific context
        request_id_context.reset(context_token)
# Load the active model recorded in the model registry, falling back to the stable path.
MODEL_PATH = Path("models/best_customer_model.joblib")
trained_model = None
MODEL_THRESHOLD = 0.5
MODEL_NAME = "unavailable"
MODEL_FEATURE_COLUMNS = []
MODEL_VERSION = None


def load_active_model():
    global trained_model, MODEL_THRESHOLD, MODEL_NAME
    global MODEL_FEATURE_COLUMNS, MODEL_VERSION

    db = SessionLocal()

    try:
        active_record = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.is_active.is_(True))
            .order_by(ModelRegistry.id.desc())
            .first()
        )

        selected_path = (
            Path(active_record.artifact_path)
            if active_record
            else MODEL_PATH
        )

        if not selected_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {selected_path}. "
                "Train a model first."
            )

        # Load artifact into local variables first
        bundle = joblib.load(selected_path)

        new_model = bundle["model"]
        new_threshold = float(bundle["threshold"])
        new_model_name = bundle["model_name"]
        new_feature_columns = bundle["feature_columns"]
        new_model_version = (
            bundle.get("version")
            or (active_record.version if active_record else None)
        )

        if not hasattr(new_model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba.")

        if not new_feature_columns:
            raise ValueError("Model feature columns are missing.")

        if not 0 <= new_threshold <= 1:
            raise ValueError("Invalid model threshold.")

        # Update application state only after complete validation
        trained_model = new_model
        MODEL_THRESHOLD = new_threshold
        MODEL_NAME = new_model_name
        MODEL_FEATURE_COLUMNS = new_feature_columns
        MODEL_VERSION = new_model_version

        logger.info(
            "Model loaded | Version: %s | Model: %s",
            MODEL_VERSION,
            MODEL_NAME
        )

    finally:
        db.close()

load_active_model()

# Customer fields required to build the same features used during training.
class CustomerInput(BaseModel):
    age: float = Field(gt=0, le=120)
    annual_income: float = Field(ge=0)
    website_visits: int = Field(ge=0)
    previous_purchases: int = Field(ge=0)
    avg_session_minutes: float = Field(ge=0)
    marketing_emails_opened: int = Field(ge=0)


class ExperimentCreate(BaseModel):
    experiment_name: str = Field(min_length=3, max_length=200)
    experiment_group: str | None = None
    dataset_id: str
    model_name: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    training_config: dict[str, Any] = Field(default_factory=dict)


class ExperimentUpdate(BaseModel):
    validation_metrics: dict[str, Any] | None = None
    test_metrics: dict[str, Any] | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    status: str | None = None
    error_message: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    model_version: str | None = None
    is_best: bool | None = None


class AutoTuneRequest(BaseModel):
    c_values: list[float] = Field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0]
    )
    random_state: int = 42
    test_size: float = 0.15
    validation_size: float = 0.15
    threshold_target_precision: float = 0.60


# Define the root endpoint
@app.get("/")
def home():

    return {
        "message": "AI Data Intelligence API is running",
        "status": "healthy"
    }


# Predict customer purchase likelihood using the latest trained pipeline.
@app.post("/predict")
def predict_customer(customer: CustomerInput):
    db = SessionLocal()

    try:
        start_time = time.perf_counter()
        customer_data = customer.model_dump()

        # Recreate the engineered features used by the training dataset.
        customer_data["customer_engagement_score"] = (
            customer_data["website_visits"]
            + customer_data["marketing_emails_opened"]
        )

        previous_purchases = customer_data["previous_purchases"]
        if previous_purchases == 0:
            customer_data["purchase_history_segment"] = "New Customer"
        elif previous_purchases <= 2:
            customer_data["purchase_history_segment"] = "Occasional Buyer"
        elif previous_purchases <= 4:
            customer_data["purchase_history_segment"] = "Repeat Buyer"
        else:
            customer_data["purchase_history_segment"] = "Frequent Buyer"

        annual_income = customer_data["annual_income"]
        if annual_income <= 30000:
            customer_data["income_segment"] = "Low Income"
        elif annual_income <= 60000:
            customer_data["income_segment"] = "Medium Income"
        else:
            customer_data["income_segment"] = "High Income"

        # Keep exactly the feature names and order used during model training.
        customer_dataframe = pd.DataFrame([customer_data]).reindex(
            columns=MODEL_FEATURE_COLUMNS
        )

        purchase_probability = float(
            trained_model.predict_proba(customer_dataframe)[0][1]
        )
        prediction = int(purchase_probability >= MODEL_THRESHOLD)
        decision = (
            "Likely to Purchase" if prediction == 1
            else "Unlikely to Purchase"
        )
        processing_time = time.perf_counter() - start_time

        prediction_result = {
            "prediction": prediction,
            "decision": decision,
            "purchase_probability": round(purchase_probability, 4),
            "threshold": MODEL_THRESHOLD,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
        }

        # Save the prediction to the existing prediction history table.
        prediction_record = PredictionHistory(
            request_id=request_id_context.get(),
            age=customer_data["age"],
            annual_income=customer_data["annual_income"],
            purchase_probability=purchase_probability,
            prediction=prediction,
            decision=decision,
            processing_time=processing_time,
        )
        db.add(prediction_record)
        db.commit()

        logger.info(
            "Prediction completed | Request ID: %s | Decision: %s | Processing time: %.4f seconds",
            request_id_context.get(),
            decision,
            processing_time,
        )

        return {
            "success": True,
            "data": prediction_result,
            "processing_time_seconds": round(processing_time, 4),
        }

    except Exception:
        db.rollback()
        logger.exception(
            "Prediction failed | Request ID: %s",
            request_id_context.get(),
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the prediction.",
        )

    finally:
        db.close()


# Define the prediction history endpoint with pagination
@app.get("/predictions")
def get_prediction_history(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=10, ge=1, le=100)
):

    db = SessionLocal()

    try:
        # Calculate the number of records to skip
        skip = (page - 1) * limit

        # Get the total number of prediction records
        total_records = db.query(PredictionHistory).count()

        # Fetch prediction records for the requested page
        records = (
            db.query(PredictionHistory)
            .order_by(PredictionHistory.id.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

        # Convert database records into JSON-compatible dictionaries
        prediction_data = [
            {
                "id": record.id,
                "request_id": record.request_id,
                "age": record.age,
                "annual_income": record.annual_income,
                "purchase_probability": record.purchase_probability,
                "prediction": record.prediction,
                "decision": record.decision,
                "processing_time": record.processing_time,
                "created_at": record.created_at.isoformat()
            }
            for record in records
        ]

        # Return paginated prediction history
        return {
            "success": True,
            "total_records": total_records,
            "page": page,
            "limit": limit,
            "total_pages": (total_records + limit - 1) // limit,
            "data": prediction_data
        }

    except Exception:
        logger.exception("Failed to retrieve prediction history")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while retrieving prediction history."
        )

    finally:
        db.close()


# Define the dashboard statistics endpoint
@app.get("/dashboard/stats")
def get_dashboard_stats():

    db = SessionLocal()

    try:
        # Calculate total prediction records
        total_predictions = db.query(PredictionHistory).count()

        # Count customers predicted as buyers
        total_buyers = db.query(PredictionHistory).filter(
            PredictionHistory.prediction == 1
        ).count()

        # Count customers predicted as non-buyers
        total_non_buyers = db.query(PredictionHistory).filter(
            PredictionHistory.prediction == 0
        ).count()

        # Calculate average purchase probability
        average_probability = db.query(
            func.avg(PredictionHistory.purchase_probability)
        ).scalar()

        # Calculate average processing time
        average_processing_time = db.query(
            func.avg(PredictionHistory.processing_time)
        ).scalar()

        # Calculate buyer percentage
        buyer_percentage = (
            (total_buyers / total_predictions) * 100
            if total_predictions > 0 else 0
        )

        return {
            "success": True,
            "data": {
                "total_predictions": total_predictions,
                "total_buyers": total_buyers,
                "total_non_buyers": total_non_buyers,
                "buyer_percentage": round(buyer_percentage, 2),
                "average_purchase_probability": round(average_probability or 0, 4),
                "average_processing_time_seconds": round(average_processing_time or 0, 4)
            }
        }

    except Exception:
        logger.exception("Failed to retrieve dashboard statistics")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while retrieving dashboard statistics."
        )

    finally:
        db.close()


# Define the date-wise analytics endpoint
@app.get("/dashboard/analytics")
def get_dashboard_analytics():

    db = SessionLocal()

    try:
        # Get the current UTC time
        now = datetime.now(timezone.utc)

        # Define date ranges
        today_start = now.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        seven_days_ago = now - timedelta(days=7)
        thirty_days_ago = now - timedelta(days=30)

        # Calculate analytics for a date range
        def calculate_period_stats(start_date):
            records = db.query(PredictionHistory).filter(
                PredictionHistory.created_at >= start_date.replace(tzinfo=None)
            )

            total = records.count()

            buyers = records.filter(
                PredictionHistory.prediction == 1
            ).count()

            non_buyers = records.filter(
                PredictionHistory.prediction == 0
            ).count()

            return {
                "total_predictions": total,
                "buyers": buyers,
                "non_buyers": non_buyers,
                "buyer_percentage": round(
                    (buyers / total) * 100, 2
                ) if total else 0
            }

        # Generate date-wise analytics
        return {
            "success": True,
            "data": {
                "today": calculate_period_stats(today_start),
                "last_7_days": calculate_period_stats(seven_days_ago),
                "last_30_days": calculate_period_stats(thirty_days_ago)
            }
        }

    except Exception:
        logger.exception("Failed to retrieve date-wise analytics")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while retrieving analytics."
        )

    finally:
        db.close()



# Define the dataset upload endpoint
@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):

    try:
        original_name = Path(file.filename or "upload.csv").name
        file_extension = Path(original_name).suffix.lower()

        if file_extension not in ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported for public demo uploads."
            )

        # Read the uploaded file content
        file_content = await file.read()

        # Validate the file size
        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 2 MB limit."
            )

        if b"\x00" in file_content[:4096]:
            raise HTTPException(
                status_code=400,
                detail="File content looks invalid for CSV."
            )

        # Load the dataset into a Pandas DataFrame
        import pandas as pd

        dataframe = pd.read_csv(io.BytesIO(file_content))

        # Validate the dataset structure
        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="Dataset contains no records."
            )

        if len(dataframe.columns) == 0:
            raise HTTPException(
                status_code=400,
                detail="Dataset contains no columns."
            )

        if len(dataframe) > MAX_UPLOAD_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset exceeds the {MAX_UPLOAD_ROWS} row limit."
            )

        if len(dataframe.columns) > MAX_UPLOAD_COLUMNS:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset exceeds the {MAX_UPLOAD_COLUMNS} column limit."
            )

        # Generate a unique filename (never trust client path/name on disk)
        dataset_id = str(uuid.uuid4())
        safe_filename = f"{dataset_id}.csv"

        # Save the uploaded dataset
        file_path = UPLOAD_DIR / safe_filename
        file_path.write_bytes(file_content)

        # Generate dataset preview
        preview = dataframe.head(5).fillna("").to_dict(
            orient="records"
        )

        # Calculate missing values
        missing_values = int(dataframe.isnull().sum().sum())

        # Calculate duplicate rows
        duplicate_rows = int(dataframe.duplicated().sum())
        
        # Calculate data quality metrics
        total_cells = dataframe.shape[0] * dataframe.shape[1]

        completeness_score = (((total_cells - missing_values) / total_cells) * 100 if total_cells else 0)

        uniqueness_score = (((len(dataframe) - duplicate_rows) / len(dataframe)) * 100 if len(dataframe) else 0)
        # Generate comprehensive data quality report
        quality_report = DataQualityAnalyzer.analyze(dataframe)
        
        # Store dataset quality report in the database  
        # Save dataset quality report to database
        db = SessionLocal()

        try:
            report = DatasetQualityReport(
            dataset_id=dataset_id,
            filename=original_name[:120],
            total_rows=len(dataframe),
            total_columns=len(dataframe.columns),
            quality_report=quality_report
        )
            db.add(report)
            db.commit()

        except Exception:
            db.rollback()
            logger.exception("Failed to save dataset quality report")
            raise HTTPException(status_code=500,detail="Failed to save dataset quality report.")
        finally:
            db.close()

        return {
            "success": True,
            "message": "Dataset uploaded successfully.",
            "data": {
                "dataset_id": dataset_id,
                "filename": original_name[:120],
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "column_names": dataframe.columns.tolist(),
                "missing_values": missing_values,
                "duplicate_rows": duplicate_rows,
                "preview": preview,
                "quality":quality_report
            }
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Dataset upload failed")

        raise HTTPException(
            status_code=400,
            detail="Unable to process the uploaded dataset."
        )


@app.get("/datasets")
def get_datasets(page: int = 1, limit: int = 10):
    if page < 1 or limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Invalid pagination parameters."
        )

    db = SessionLocal()

    try:
        total = db.query(DatasetQualityReport).count()

        datasets = (
            db.query(DatasetQualityReport)
            .order_by(desc(DatasetQualityReport.created_at))
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )

        return {
            "success": True,
            "total": total,
            "page": page,
            "limit": limit,
            "datasets": [
                {
                    "dataset_id": item.dataset_id,
                    "filename": item.filename,
                    "total_rows": item.total_rows,
                    "total_columns": item.total_columns,
                    "quality_report": item.quality_report,
                    "created_at": item.created_at.isoformat()
                }
                for item in datasets
            ]
        }

    except Exception:
        logger.exception("Failed to retrieve datasets")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve datasets."
        )

    finally:
        db.close()


@app.get("/datasets/{dataset_id}")
def get_dataset_details(dataset_id: str):
    db = SessionLocal()

    try:
        dataset = (
            db.query(DatasetQualityReport)
            .filter(DatasetQualityReport.dataset_id == dataset_id)
            .first()
        )

        if not dataset:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found."
            )

        return {
            "success": True,
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "filename": dataset.filename,
                "total_rows": dataset.total_rows,
                "total_columns": dataset.total_columns,
                "quality_report": dataset.quality_report,
                "created_at": dataset.created_at.isoformat()
            }
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Failed to retrieve dataset details")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve dataset details."
        )

    finally:
        db.close()
@app.post("/datasets/{dataset_id}/clean")
def clean_dataset(dataset_id: str):
    db = SessionLocal()

    try:
        dataset = db.query(DatasetQualityReport).filter(
            DatasetQualityReport.dataset_id == dataset_id
        ).first()

        if not dataset:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found"
            )

        extension = Path(dataset.filename).suffix.lower()
        upload_path = Path("data/uploads") / f"{dataset_id}{extension}"

        if not upload_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Uploaded file not found"
            )

        if extension == ".csv":
            dataframe = pd.read_csv(upload_path)
        elif extension == ".xlsx":
            dataframe = pd.read_excel(upload_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format"
            )

        cleaned_df, cleaning_report = DataCleaner.clean(dataframe)

        cleaned_dir = Path("data/cleaned")
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        cleaned_path = cleaned_dir / f"{dataset_id}_cleaned.csv"
        cleaned_df.to_csv(cleaned_path, index=False)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "cleaned_file": str(cleaned_path),
            "cleaning_report": cleaning_report,
            "quality_after_cleaning": DataQualityAnalyzer.analyze(cleaned_df),
            "preview": cleaned_df.head(5).to_dict(orient="records")
        }

    finally:
        db.close()

@app.post("/datasets/{dataset_id}/features")
def create_dataset_features(dataset_id: str):

    cleaned_path = (
        Path("data/cleaned")
        / f"{dataset_id}_cleaned.csv"
    )

    if not cleaned_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Cleaned dataset not found. Clean dataset first."
        )

    dataframe = pd.read_csv(cleaned_path)

    engineered_df = FeatureEngineer.transform(dataframe)

    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    output_path = features_dir / f"{dataset_id}_features.csv"

    engineered_df.to_csv(output_path, index=False)

    new_features = [
        "customer_engagement_score",
        "purchase_history_segment",
        "income_segment"
    ]

    return {
        "success": True,
        "dataset_id": dataset_id,
        "original_rows": len(dataframe),
        "engineered_rows": len(engineered_df),
        "new_features": new_features,
        "total_columns": len(engineered_df.columns),
        "output_file": str(output_path),
        "preview": engineered_df.head(5).to_dict(orient="records")
    }

@app.post("/datasets/{dataset_id}/train")
def train_dataset_model(dataset_id: str):

    features_path = (
        Path("data/features")
        / f"{dataset_id}_features.csv"
    )

    if not features_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Feature dataset not found. Run feature engineering first."
        )

    dataframe = pd.read_csv(features_path)

    try:
        result = ModelTrainer.train(dataframe, dataset_id=dataset_id)
        load_active_model()

        return {
            "success": True,
            "dataset_id": dataset_id,
            "training_rows": len(dataframe),
            "result": result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {str(error)}"
        )

# List registered model versions.
@app.get("/models")
def list_models():
    db = SessionLocal()
    try:
        records = db.query(ModelRegistry).order_by(ModelRegistry.id.desc()).all()
        return {
            "success": True,
            "total": len(records),
            "models": [
                {
                    "id": item.id,
                    "version": item.version,
                    "model_name": item.model_name,
                    "dataset_id": item.dataset_id,
                    "training_rows": item.training_rows,
                    "threshold": item.threshold,
                    "metrics": item.metrics,
                    "feature_columns": item.feature_columns,
                    "artifact_path": item.artifact_path,
                    "is_active": item.is_active,
                    "created_at": item.created_at.isoformat(),
                }
                for item in records
            ],
        }
    finally:
        db.close()


# Return the currently active production model.
@app.get("/models/active")
def get_active_model():
    db = SessionLocal()
    try:
        item = db.query(ModelRegistry).filter(
            ModelRegistry.is_active.is_(True)
        ).order_by(ModelRegistry.id.desc()).first()
        if not item:
            raise HTTPException(status_code=404, detail="No active registered model found.")
        return {
            "success": True,
            "model": {
                "version": item.version,
                "model_name": item.model_name,
                "dataset_id": item.dataset_id,
                "training_rows": item.training_rows,
                "threshold": item.threshold,
                "metrics": item.metrics,
                "artifact_path": item.artifact_path,
                "created_at": item.created_at.isoformat(),
            },
        }
    finally:
        db.close()


# Compare two registered model versions using their held-out test metrics.
@app.get("/models/compare")
def compare_models(
    version_a: str = Query(..., description="First model version, e.g. v1"),
    version_b: str = Query(..., description="Second model version, e.g. v2"),
):
    db = SessionLocal()
    try:
        first = db.query(ModelRegistry).filter(
            ModelRegistry.version == version_a
        ).first()
        second = db.query(ModelRegistry).filter(
            ModelRegistry.version == version_b
        ).first()

        if not first or not second:
            raise HTTPException(
                status_code=404,
                detail="One or both model versions were not found in the registry.",
            )

        metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        metric_comparison = {}
        for metric in metric_names:
            value_a = float((first.metrics or {}).get(metric, 0.0))
            value_b = float((second.metrics or {}).get(metric, 0.0))
            metric_comparison[metric] = {
                version_a: value_a,
                version_b: value_b,
                "difference_b_minus_a": round(value_b - value_a, 4),
            }

        return {
            "success": True,
            "comparison": {
                "version_a": {
                    "version": first.version,
                    "model_name": first.model_name,
                    "threshold": first.threshold,
                    "dataset_id": first.dataset_id,
                    "created_at": first.created_at.isoformat(),
                },
                "version_b": {
                    "version": second.version,
                    "model_name": second.model_name,
                    "threshold": second.threshold,
                    "dataset_id": second.dataset_id,
                    "created_at": second.created_at.isoformat(),
                },
                "metrics": metric_comparison,
                "higher_is_better": metric_names,
                "note": "Metrics are held-out test results; compare models trained/evaluated on compatible datasets and splits.",
            },
        }
    finally:
        db.close()


@app.post("/models/{version}/activate")
def activate_model(version: str):
    global trained_model, MODEL_THRESHOLD, MODEL_NAME
    global MODEL_FEATURE_COLUMNS, MODEL_VERSION

    db = SessionLocal()

    try:
        # 1. Find requested model version
        record = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.version == version)
            .first()
        )

        if not record:
            raise HTTPException(
                status_code=404,
                detail="Model version not found."
            )

        artifact_path = Path(record.artifact_path)

        if not artifact_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Model artifact file not found."
            )

        # 2. Load and validate model BEFORE changing production
        bundle = joblib.load(artifact_path)

        new_model = bundle["model"]
        new_threshold = float(bundle["threshold"])
        new_model_name = bundle["model_name"]
        new_feature_columns = bundle["feature_columns"]
        new_model_version = bundle.get("version") or record.version

        if new_model_version != record.version:
            raise ValueError(
                "Model artifact version does not match registry version."
            )

        if not hasattr(new_model, "predict_proba"):
            raise ValueError(
                "Model does not support predict_proba."
            )

        if not new_feature_columns:
            raise ValueError("Model feature columns are missing.")

        if not 0 <= new_threshold <= 1:
            raise ValueError("Invalid model threshold.")

        # 3. Update registry inside database transaction
        db.query(ModelRegistry).update(
            {ModelRegistry.is_active: False},
            synchronize_session=False
        )

        record.is_active = True

        db.commit()

        # 4. Update application memory only after successful commit
        trained_model = new_model
        MODEL_THRESHOLD = new_threshold
        MODEL_NAME = new_model_name
        MODEL_FEATURE_COLUMNS = new_feature_columns
        MODEL_VERSION = new_model_version

        logger.info(
            "Model activated | Version: %s | Model: %s",
            MODEL_VERSION,
            MODEL_NAME
        )

        return {
            "success": True,
            "message": f"Model {version} is now active.",
            "active_version": MODEL_VERSION,
            "model_name": MODEL_NAME,
            "threshold": MODEL_THRESHOLD
        }

    except HTTPException:
        db.rollback()
        raise

    except Exception as error:
        db.rollback()

        logger.exception(
            "Failed to activate model version %s",
            version
        )

        raise HTTPException(
            status_code=500,
            detail=f"Unable to activate model: {str(error)}"
        )

    finally:
        db.close()


# --------------------------------------------------
# EXPERIMENT TRACKING APIs
# --------------------------------------------------


# Create a new training experiment.
@app.post("/experiments")
def create_experiment(payload: ExperimentCreate):

    db = SessionLocal()

    try:
        experiment = TrainingExperiment(
            experiment_name=payload.experiment_name,
            experiment_group=payload.experiment_group,
            dataset_id=payload.dataset_id,
            model_name=payload.model_name,
            hyperparameters=payload.hyperparameters,
            training_config=payload.training_config,
            status="pending",
        )

        db.add(experiment)
        db.commit()
        db.refresh(experiment)

        return {
            "success": True,
            "message": "Experiment created successfully.",
            "experiment": {
                "id": experiment.id,
                "experiment_name": experiment.experiment_name,
                "experiment_group": experiment.experiment_group,
                "dataset_id": experiment.dataset_id,
                "model_name": experiment.model_name,
                "hyperparameters": experiment.hyperparameters,
                "training_config": experiment.training_config,
                "status": experiment.status,
                "created_at": experiment.created_at.isoformat(),
            },
        }

    except Exception:
        db.rollback()
        logger.exception("Failed to create experiment")
        raise HTTPException(
            status_code=500,
            detail="Unable to create experiment.",
        )

    finally:
        db.close()


# List all experiments with pagination.
@app.get("/experiments")
def list_experiments(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=10, ge=1, le=100),
):

    db = SessionLocal()

    try:
        total = db.query(TrainingExperiment).count()

        records = (
            db.query(TrainingExperiment)
            .order_by(TrainingExperiment.id.desc())
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )

        return {
            "success": True,
            "total": total,
            "page": page,
            "limit": limit,
            "experiments": [
                {
                    "id": item.id,
                    "experiment_name": item.experiment_name,
                    "experiment_group": item.experiment_group,
                    "dataset_id": item.dataset_id,
                    "model_name": item.model_name,
                    "hyperparameters": item.hyperparameters,
                    "training_config": item.training_config,
                    "validation_metrics": item.validation_metrics,
                    "test_metrics": item.test_metrics,
                    "best_metric_name": item.best_metric_name,
                    "best_metric_value": item.best_metric_value,
                    "status": item.status,
                    "duration_seconds": item.duration_seconds,
                    "model_version": item.model_version,
                    "is_best": item.is_best,
                    "created_at": item.created_at.isoformat(),
                    "completed_at": (
                        item.completed_at.isoformat()
                        if item.completed_at else None
                    ),
                }
                for item in records
            ],
        }

    except Exception:
        logger.exception("Failed to list experiments")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve experiments.",
        )

    finally:
        db.close()


# Compare two experiments using a selected validation metric.
@app.get("/experiments/compare")
def compare_experiments(
    experiment_a: int = Query(..., ge=1),
    experiment_b: int = Query(..., ge=1),
    metric_name: str = Query(default="roc_auc"),
):

    db = SessionLocal()

    try:
        first = db.query(TrainingExperiment).filter(
            TrainingExperiment.id == experiment_a
        ).first()

        second = db.query(TrainingExperiment).filter(
            TrainingExperiment.id == experiment_b
        ).first()

        if not first or not second:
            raise HTTPException(
                status_code=404,
                detail="One or both experiments were not found.",
            )

        if first.dataset_id != second.dataset_id:
            raise HTTPException(
                status_code=400,
                detail="Compare experiments trained on the same dataset.",
            )

        metrics_a = first.validation_metrics or {}
        metrics_b = second.validation_metrics or {}

        if metric_name not in metrics_a or metric_name not in metrics_b:
            raise HTTPException(
                status_code=400,
                detail=f"Metric '{metric_name}' is unavailable in both experiments.",
            )

        value_a = float(metrics_a[metric_name])
        value_b = float(metrics_b[metric_name])

        return {
            "success": True,
            "comparison": {
                "experiment_a": first.id,
                "experiment_b": second.id,
                "metric": metric_name,
                "value_a": value_a,
                "value_b": value_b,
                "difference_b_minus_a": round(value_b - value_a, 6),
                "higher_value": (
                    "experiment_a" if value_a > value_b
                    else "experiment_b" if value_b > value_a
                    else "equal"
                ),
            },
        }

    finally:
        db.close()


# Get one experiment by ID.
@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: int):

    db = SessionLocal()

    try:
        item = db.query(TrainingExperiment).filter(
            TrainingExperiment.id == experiment_id
        ).first()

        if not item:
            raise HTTPException(
                status_code=404,
                detail="Experiment not found.",
            )

        return {
            "success": True,
            "experiment": {
                "id": item.id,
                "experiment_name": item.experiment_name,
                "experiment_group": item.experiment_group,
                "dataset_id": item.dataset_id,
                "model_name": item.model_name,
                "hyperparameters": item.hyperparameters,
                "training_config": item.training_config,
                "validation_metrics": item.validation_metrics,
                "test_metrics": item.test_metrics,
                "best_metric_name": item.best_metric_name,
                "best_metric_value": item.best_metric_value,
                "status": item.status,
                "error_message": item.error_message,
                "duration_seconds": item.duration_seconds,
                "model_version": item.model_version,
                "is_best": item.is_best,
                "created_at": item.created_at.isoformat(),
                "completed_at": (
                    item.completed_at.isoformat()
                    if item.completed_at else None
                ),
            },
        }

    finally:
        db.close()


# Update experiment results and status.
@app.patch("/experiments/{experiment_id}")
def update_experiment(
    experiment_id: int,
    payload: ExperimentUpdate,
):

    db = SessionLocal()

    try:
        item = db.query(TrainingExperiment).filter(
            TrainingExperiment.id == experiment_id
        ).first()

        if not item:
            raise HTTPException(
                status_code=404,
                detail="Experiment not found.",
            )

        updates = payload.model_dump(exclude_unset=True)

        allowed_statuses = {
            "pending",
            "running",
            "completed",
            "failed",
        }

        if (
            "status" in updates
            and updates["status"] not in allowed_statuses
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid experiment status.",
            )

        if updates.get("model_version"):
            registered_model = db.query(ModelRegistry).filter(
                ModelRegistry.version == updates["model_version"]
            ).first()

            if not registered_model:
                raise HTTPException(
                    status_code=404,
                    detail="Registered model version not found.",
                )

        if updates.get("is_best") is True:
            db.query(TrainingExperiment).filter(
                TrainingExperiment.id != experiment_id,
                TrainingExperiment.dataset_id == item.dataset_id,
            ).update(
                {TrainingExperiment.is_best: False},
                synchronize_session=False,
            )

        for key, value in updates.items():
            setattr(item, key, value)

        if updates.get("status") in {"completed", "failed"}:
            item.completed_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(item)

        return {
            "success": True,
            "message": "Experiment updated successfully.",
            "experiment_id": item.id,
            "status": item.status,
            "is_best": item.is_best,
            "validation_metrics": item.validation_metrics,
            "test_metrics": item.test_metrics,
            "model_version": item.model_version,
        }

    except HTTPException:
        db.rollback()
        raise

    except Exception:
        db.rollback()
        logger.exception("Failed to update experiment")
        raise HTTPException(
            status_code=500,
            detail="Unable to update experiment.",
        )

    finally:
        db.close()


@app.post("/experiments/{experiment_id}/run")
def run_training_experiment(experiment_id: int):

    db = SessionLocal()

    try:
        experiment = (
            db.query(TrainingExperiment)
            .filter(TrainingExperiment.id == experiment_id)
            .first()
        )

        if not experiment:
            raise HTTPException(
                status_code=404,
                detail="Experiment not found."
            )

        dataset_id = experiment.dataset_id

    finally:
        db.close()

    features_path = (
        Path("data/features")
        / f"{dataset_id}_features.csv"
    )

    if not features_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Feature dataset not found."
        )

    dataframe = pd.read_csv(features_path)

    try:
        result = ModelTrainer.run_experiment(
            dataframe,
            experiment_id
        )

        return result

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Experiment training failed: {str(error)}"
        )


@app.post("/datasets/{dataset_id}/tune")
def auto_tune_dataset(dataset_id: str, request: AutoTuneRequest):

    # Validate tuning configuration
    if not request.c_values:
        raise HTTPException(
            status_code=400,
            detail="At least one C value is required."
        )

    if len(request.c_values) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 experiments allowed per tuning request."
        )

    if any(value <= 0 for value in request.c_values):
        raise HTTPException(
            status_code=400,
            detail="All C values must be greater than zero."
        )

    if len(set(request.c_values)) != len(request.c_values):
        raise HTTPException(
            status_code=400,
            detail="Duplicate C values are not allowed."
        )

    if (
        request.test_size <= 0
        or request.validation_size <= 0
        or request.test_size + request.validation_size >= 1
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid train/validation/test split."
        )

    features_path = (
        Path("data/features") / f"{dataset_id}_features.csv"
    )

    if not features_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Feature dataset not found. Run feature engineering first."
        )

    dataframe = pd.read_csv(features_path)

    experiment_ids = []
    experiment_group = f"auto_tune_{uuid.uuid4().hex[:8]}"

    # Step 1: Create experiments
    db = SessionLocal()

    try:
        for c_value in request.c_values:
            experiment = TrainingExperiment(
                experiment_name=f"Auto Logistic Regression C={c_value}",
                experiment_group=experiment_group,
                dataset_id=dataset_id,
                model_name="logistic_regression",
                hyperparameters={
                    "C": c_value,
                    "max_iter": 1000,
                    "solver": "lbfgs",
                },
                training_config={
                    "test_size": request.test_size,
                    "validation_size": request.validation_size,
                    "random_state": request.random_state,
                    "threshold_target_precision":
                        request.threshold_target_precision,
                },
                status="pending",
                is_best=False,
            )

            db.add(experiment)
            db.flush()
            experiment_ids.append(experiment.id)

        db.commit()

    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Experiment creation failed: {str(error)}"
        )

    finally:
        db.close()

    # Step 2: Run all experiments
    results = []

    for experiment_id in experiment_ids:
        try:
            result = ModelTrainer.run_experiment(
                dataframe,
                experiment_id
            )
            results.append(result)

        except Exception as error:
            results.append({
                "success": False,
                "experiment_id": experiment_id,
                "error": str(error),
            })

    # Step 3: Select best using validation ROC-AUC only
    db = SessionLocal()

    try:
        completed = (
            db.query(TrainingExperiment)
            .filter(
                TrainingExperiment.id.in_(experiment_ids),
                TrainingExperiment.status == "completed",
                TrainingExperiment.best_metric_value.isnot(None),
            )
            .order_by(TrainingExperiment.best_metric_value.desc())
            .all()
        )

        if not completed:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "All experiments failed.",
                    "results": results,
                }
            )

        best_experiment = completed[0]

        # Clear previous best for this dataset
        db.query(TrainingExperiment).filter(
            TrainingExperiment.dataset_id == dataset_id
        ).update(
            {TrainingExperiment.is_best: False},
            synchronize_session=False,
        )

        best_experiment.is_best = True
        db.commit()

        return {
            "success": True,
            "message": "Automated hyperparameter tuning completed.",
            "dataset_id": dataset_id,
            "experiment_group": experiment_group,
            "total_experiments": len(experiment_ids),
            "completed_experiments": len(completed),
            "failed_experiments": len(experiment_ids) - len(completed),
            "best_experiment": {
                "id": best_experiment.id,
                "name": best_experiment.experiment_name,
                "hyperparameters": best_experiment.hyperparameters,
                "validation_metrics": best_experiment.validation_metrics,
                "model_version": best_experiment.model_version,
            },
            "results": results,
            "production_model_changed": False,
        }

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Same-origin hosting: /api/* → existing routes, and serve the React build
# ---------------------------------------------------------------------------
import os
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIST = Path(
    os.environ.get(
        "FRONTEND_DIST",
        str(Path(__file__).resolve().parents[2] / "frontend" / "dist"),
    )
)


@app.middleware("http")
async def strip_api_prefix(request, call_next):
    """Allow the UI to call /api/... on the same host as FastAPI."""
    path = request.scope.get("path", "")
    if path == "/api" or path.startswith("/api/"):
        new_path = path[4:] or "/"
        request.scope["path"] = new_path
        if "raw_path" in request.scope:
            request.scope["raw_path"] = new_path.encode("utf-8")
    return await call_next(request)


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

    samples_dir = FRONTEND_DIST / "samples"
    if samples_dir.exists():
        app.mount("/samples", StaticFiles(directory=str(samples_dir)), name="frontend-samples")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """SPA fallback — API routes registered above take priority."""
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(index)
        raise HTTPException(
            status_code=404,
            detail="Frontend build not found. Run npm run build in frontend/.",
        )

