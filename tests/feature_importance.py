import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

df = pd.read_csv(
    "data/features/e729bb41-8a87-49c0-8053-32a665d900dd_features.csv"
)

X = df.drop(columns=["purchased", "customer_id"], errors="ignore")
y = df["purchased"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
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
])

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

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)

    result = permutation_importance(
        pipeline, X_val, y_val,
        scoring="roc_auc", n_repeats=10,
        random_state=42, n_jobs=-1
    )

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False)

    print(f"\n{name}")
    print(importance.to_string(index=False))

print("\nFeature importance completed.")
print("Production model unchanged.")
