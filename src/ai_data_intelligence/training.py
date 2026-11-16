from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import func
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)

from sklearn.base import clone
from ai_data_intelligence.database import SessionLocal, engine
from ai_data_intelligence.database_models import (
    Base,
    ModelRegistry,
    TrainingExperiment,
)

class ModelTrainer:
    @staticmethod
    def _build_preprocessor(numeric_columns, categorical_columns):
        """Create a fresh preprocessing pipeline for each candidate model."""
        transformers = []
        if numeric_columns:
            transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_columns))
        if categorical_columns:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_columns))
        return ColumnTransformer(transformers, remainder="drop")


    @staticmethod
    def cross_validate_model(
        estimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        numeric_columns: list[str],
        categorical_columns: list[str],
        n_splits: int = 5,
        random_state: int = 42,
    ) -> dict:
        """
        Evaluate an estimator with stratified K-fold cross-validation.

        Preprocessing remains inside the Pipeline, so each fold fits its
        own imputers, scaler, and encoder. This prevents preprocessing
        leakage from a fold's validation portion.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        class_counts = y_train.value_counts()
        if class_counts.min() < n_splits:
            raise ValueError(
                f"Each target class must contain at least {n_splits} "
                "rows for StratifiedKFold."
            )

        pipeline = Pipeline([
            (
                "preprocessor",
                ModelTrainer._build_preprocessor(
                    numeric_columns, categorical_columns
                ),
            ),
            ("model", clone(estimator)),
        ])

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "accuracy": "accuracy",
            },
            return_train_score=False,
        )

        return {
            "folds": n_splits,
            "roc_auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
            "roc_auc_std": round(float(scores["test_roc_auc"].std()), 4),
            "accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
            "accuracy_std": round(float(scores["test_accuracy"].std()), 4),
        }

    @staticmethod
    def train(dataframe: pd.DataFrame, dataset_id: str | None = None):
        if "purchased" not in dataframe.columns:
            raise ValueError("Training dataset must contain 'purchased'.")

        df = dataframe.copy()
        drop_columns = [column for column in ["purchased", "customer_id"] if column in df.columns]
        X = df.drop(columns=drop_columns)
        y = df["purchased"].astype(int)
        if y.nunique() != 2:
            raise ValueError("Target 'purchased' must contain both classes 0 and 1.")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp,
        )

        numeric = X.select_dtypes(include="number").columns.tolist()
        categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42,
                class_weight="balanced",
            ),
            # Additional candidate for the v2 experiment.
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            ),
        }

        pipelines = {}
        validation_results = {}
        for name, estimator in models.items():
            pipeline = Pipeline([
                ("preprocessor", ModelTrainer._build_preprocessor(numeric, categorical)),
                ("model", estimator),
            ])
            pipeline.fit(X_train, y_train)
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            validation_results[name] = {
                "roc_auc": round(float(roc_auc_score(y_val, val_prob)), 4),
            }
            pipelines[name] = pipeline

        best_name = max(validation_results, key=lambda name: validation_results[name]["roc_auc"])
        best_pipeline = pipelines[best_name]
        val_prob = best_pipeline.predict_proba(X_val)[:, 1]

        threshold_candidates = []
        for candidate in [i / 100 for i in range(10, 91)]:
            val_pred = (val_prob >= candidate).astype(int)
            precision = precision_score(y_val, val_pred, zero_division=0)
            recall = recall_score(y_val, val_pred, zero_division=0)
            if precision >= 0.60:
                threshold_candidates.append((candidate, precision, recall))

        if threshold_candidates:
            threshold, val_precision, val_recall = max(
                threshold_candidates, key=lambda item: item[2]
            )
        else:
            threshold = 0.50
            val_pred = (val_prob >= threshold).astype(int)
            val_precision = precision_score(y_val, val_pred, zero_division=0)
            val_recall = recall_score(y_val, val_pred, zero_division=0)

        test_prob = best_pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        test_metrics = {
            "accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
            "precision": round(float(precision_score(y_test, test_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, test_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, test_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, test_prob)), 4),
        }

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        try:
            previous_count = db.query(func.count(ModelRegistry.id)).scalar() or 0
            version_number = previous_count + 1
            version = f"v{version_number}"
            artifact_path = models_dir / f"customer_model_{version}.joblib"
            bundle = {
                "model": best_pipeline,
                "model_name": best_name,
                "version": version,
                "threshold": float(threshold),
                "metrics": test_metrics,
                "feature_columns": X.columns.tolist(),
                "dataset_id": dataset_id,
            }
            joblib.dump(bundle, artifact_path)
            # Keep the existing API's stable model path in sync with the newest training run.
            joblib.dump(bundle, models_dir / "best_customer_model.joblib")

            db.query(ModelRegistry).update({ModelRegistry.is_active: False})
            registry_record = ModelRegistry(
                version=version,
                model_name=best_name,
                artifact_path=str(artifact_path),
                dataset_id=dataset_id,
                training_rows=len(df),
                threshold=float(threshold),
                metrics=test_metrics,
                feature_columns=X.columns.tolist(),
                is_active=True,
                created_at=datetime.now(timezone.utc),
            )
            db.add(registry_record)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        return {
            "version": version,
            "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
            "validation_model_results": validation_results,
            "best_model": best_name,
            "selected_threshold": float(threshold),
            "validation_precision": round(float(val_precision), 4),
            "validation_recall": round(float(val_recall), 4),
            "final_test_metrics": test_metrics,
            "model_path": str(artifact_path),
            "active": True,
        }


    @staticmethod
    def run_experiment(dataframe: pd.DataFrame, experiment_id: int):
        db = SessionLocal()

        try:
            experiment = (
                db.query(TrainingExperiment)
                .filter(TrainingExperiment.id == experiment_id)
                .first()
            )

            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found.")

            if experiment.status == "completed":
                raise ValueError("This experiment is already completed.")

            if experiment.status == "running":
                raise ValueError("This experiment is already running.")

            experiment.status = "running"
            experiment.error_message = None
            db.commit()

            start_time = datetime.now(timezone.utc)

            try:
                if "purchased" not in dataframe.columns:
                    raise ValueError("Dataset must contain 'purchased'.")

                df = dataframe.copy()
                X = df.drop(
                    columns=[
                        c for c in ["purchased", "customer_id"]
                        if c in df.columns
                    ]
                )
                y = df["purchased"].astype(int)

                if y.nunique() != 2:
                    raise ValueError("Target must contain both classes 0 and 1.")

                config = experiment.training_config or {}
                params = experiment.hyperparameters or {}

                test_size = float(config.get("test_size", 0.15))
                validation_size = float(config.get("validation_size", 0.15))
                random_state = int(config.get("random_state", 42))
                target_precision = float(
                    config.get("threshold_target_precision", 0.60)
                )

                if test_size <= 0 or validation_size <= 0:
                    raise ValueError("Test and validation sizes must be positive.")

                if test_size + validation_size >= 1:
                    raise ValueError(
                        "Test size + validation size must be less than 1."
                    )

                X_train, X_temp, y_train, y_temp = train_test_split(
                    X,
                    y,
                    test_size=test_size + validation_size,
                    random_state=random_state,
                    stratify=y,
                )

                validation_fraction = (
                    validation_size / (test_size + validation_size)
                )

                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=1 - validation_fraction,
                    random_state=random_state,
                    stratify=y_temp,
                )

                numeric = X.select_dtypes(include="number").columns.tolist()
                categorical = X.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                if experiment.model_name == "logistic_regression":
                    estimator = LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    )
                elif experiment.model_name == "random_forest":
                    estimator = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        random_state=random_state,
                        class_weight="balanced",
                    )
                elif experiment.model_name == "gradient_boosting":
                    estimator = GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=random_state,
                    )
                else:
                    raise ValueError(
                        f"Unsupported model: {experiment.model_name}"
                    )

                allowed_params = estimator.get_params()
                unknown_params = set(params) - set(allowed_params)

                if unknown_params:
                    raise ValueError(
                        f"Unsupported hyperparameters: {sorted(unknown_params)}"
                    )

                estimator.set_params(**params)

                pipeline = Pipeline([
                    (
                        "preprocessor",
                        ModelTrainer._build_preprocessor(
                            numeric, categorical
                        ),
                    ),
                    ("model", estimator),
                ])

                pipeline.fit(X_train, y_train)

                # Validation: threshold selection happens only here.
                val_prob = pipeline.predict_proba(X_val)[:, 1]

                validation_roc_auc = float(
                    roc_auc_score(y_val, val_prob)
                )

                candidates = []

                for threshold_candidate in [
                    i / 100 for i in range(10, 91)
                ]:
                    val_pred = (
                        val_prob >= threshold_candidate
                    ).astype(int)

                    precision = precision_score(
                        y_val, val_pred, zero_division=0
                    )
                    recall = recall_score(
                        y_val, val_pred, zero_division=0
                    )

                    if precision >= target_precision:
                        candidates.append(
                            (threshold_candidate, precision, recall)
                        )

                if candidates:
                    threshold, val_precision, val_recall = max(
                        candidates, key=lambda item: item[2]
                    )
                else:
                    threshold = 0.50
                    val_pred = (val_prob >= threshold).astype(int)
                    val_precision = precision_score(
                        y_val, val_pred, zero_division=0
                    )
                    val_recall = recall_score(
                        y_val, val_pred, zero_division=0
                    )

                validation_metrics = {
                    "roc_auc": round(validation_roc_auc, 4),
                    "precision": round(float(val_precision), 4),
                    "recall": round(float(val_recall), 4),
                    "threshold": float(threshold),
                }

                # Final test evaluation: do not use this to tune parameters.
                test_prob = pipeline.predict_proba(X_test)[:, 1]
                test_pred = (test_prob >= threshold).astype(int)

                test_metrics = {
                    "accuracy": round(
                        float(accuracy_score(y_test, test_pred)), 4
                    ),
                    "precision": round(
                        float(precision_score(
                            y_test, test_pred, zero_division=0
                        )), 4
                    ),
                    "recall": round(
                        float(recall_score(
                            y_test, test_pred, zero_division=0
                        )), 4
                    ),
                    "f1_score": round(
                        float(f1_score(
                            y_test, test_pred, zero_division=0
                        )), 4
                    ),
                    "roc_auc": round(
                        float(roc_auc_score(y_test, test_prob)), 4
                    ),
                }

                # Save experiment artifact separately.
                models_dir = Path("models")
                models_dir.mkdir(parents=True, exist_ok=True)

                previous_count = (
                    db.query(func.count(ModelRegistry.id)).scalar() or 0
                )
                version = f"v{previous_count + 1}"

                artifact_path = (
                    models_dir
                    / f"experiment_{experiment_id}_{version}.joblib"
                )

                bundle = {
                    "model": pipeline,
                    "model_name": experiment.model_name,
                    "version": version,
                    "threshold": float(threshold),
                    "metrics": test_metrics,
                    "feature_columns": X.columns.tolist(),
                    "dataset_id": experiment.dataset_id,
                    "experiment_id": experiment_id,
                }

                joblib.dump(bundle, artifact_path)

                registry_record = ModelRegistry(
                    version=version,
                    model_name=experiment.model_name,
                    artifact_path=str(artifact_path),
                    dataset_id=experiment.dataset_id,
                    training_rows=len(df),
                    threshold=float(threshold),
                    metrics=test_metrics,
                    feature_columns=X.columns.tolist(),
                    is_active=False,
                    created_at=datetime.now(timezone.utc),
                )

                db.add(registry_record)
                db.flush()

                experiment.validation_metrics = validation_metrics
                experiment.test_metrics = test_metrics
                experiment.best_metric_name = "roc_auc"
                experiment.best_metric_value = validation_roc_auc
                experiment.model_version = version
                experiment.status = "completed"
                experiment.duration_seconds = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()
                experiment.completed_at = datetime.now(timezone.utc)

                db.commit()

                return {
                    "success": True,
                    "experiment_id": experiment_id,
                    "model_name": experiment.model_name,
                    "hyperparameters": params,
                    "version": version,
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                    "split": {
                        "train": len(X_train),
                        "validation": len(X_val),
                        "test": len(X_test),
                    },
                    "model_path": str(artifact_path),
                    "active": False,
                }

            except Exception as error:
                db.rollback()

                failed_experiment = (
                    db.query(TrainingExperiment)
                    .filter(TrainingExperiment.id == experiment_id)
                    .first()
                )

                if failed_experiment:
                    failed_experiment.status = "failed"
                    failed_experiment.error_message = str(error)
                    failed_experiment.completed_at = datetime.now(timezone.utc)
                    db.commit()

                raise

        finally:
            db.close()

