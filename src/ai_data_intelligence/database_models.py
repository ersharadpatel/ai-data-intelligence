
# SQLAlchemy database models for the AI Data Intelligence Platform.

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    JSON,
    Boolean,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), nullable=False, index=True)
    age = Column(Float, nullable=False)
    annual_income = Column(Float, nullable=False)
    purchase_probability = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False)
    decision = Column(String(50), nullable=False)
    processing_time = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class DatasetQualityReport(Base):
    __tablename__ = "dataset_quality_reports"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(String(100), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    total_rows = Column(Integer, nullable=False)
    total_columns = Column(Integer, nullable=False)
    quality_report = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ModelRegistry(Base):
    """Metadata and lifecycle state for every trained model artifact."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    artifact_path = Column(String(500), nullable=False)
    dataset_id = Column(String(100), nullable=True, index=True)
    training_rows = Column(Integer, nullable=False)
    threshold = Column(Float, nullable=False)
    metrics = Column(JSON, nullable=False)
    feature_columns = Column(JSON, nullable=False)
    is_active = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class TrainingExperiment(Base):
    """
    Stores model training experiment configurations and results.
    """

    __tablename__ = "training_experiments"

    id = Column(Integer, primary_key=True, index=True)

    experiment_name = Column(String(200), nullable=False, index=True)

    experiment_group = Column(String(100), nullable=True, index=True)

    dataset_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    model_name = Column(String(100), nullable=False, index=True)

    hyperparameters = Column(JSON, nullable=False)

    training_config = Column(JSON, nullable=False)

    validation_metrics = Column(JSON, nullable=True)

    test_metrics = Column(JSON, nullable=True)

    best_metric_name = Column(String(100), nullable=True)

    best_metric_value = Column(Float, nullable=True)

    status = Column(
        String(30),
        nullable=False,
        default="pending",
        index=True,
    )

    error_message = Column(Text, nullable=True)

    duration_seconds = Column(Float, nullable=True)

    model_version = Column(
        String(50),
        ForeignKey("model_registry.version"),
        nullable=True,
        index=True,
    )

    is_best = Column(Boolean, nullable=False, default=False, index=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    completed_at = Column(DateTime(timezone=True), nullable=True)
