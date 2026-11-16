import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATA_PATH = "data/features/e729bb41-8a87-49c0-8053-32a665d900dd_features.csv"

df = pd.read_csv(DATA_PATH)

if "purchased" not in df.columns:
    raise ValueError("Target column 'purchased' not found.")

X = df.drop(columns=[c for c in ["purchased", "customer_id"] if c in df.columns])
y = df["purchased"].astype(int)

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
], remainder="drop")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=42, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05,
        max_depth=2, random_state=42
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\nDataset rows: {len(df)}")
print(f"CV training rows: {len(X_train)}")
print("\n5-Fold Cross Validation Results")
print("-" * 65)

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "accuracy": "accuracy"},
        return_train_score=False,
        n_jobs=-1,
    )

    print(f"\n{name}")
    print(f"ROC-AUC Mean : {scores['test_roc_auc'].mean():.4f}")
    print(f"ROC-AUC Std  : {scores['test_roc_auc'].std():.4f}")
    print(f"Accuracy Mean: {scores['test_accuracy'].mean():.4f}")
    print(f"Accuracy Std : {scores['test_accuracy'].std():.4f}")

print("\nCross Validation completed.")
print("Production model was not changed.")
