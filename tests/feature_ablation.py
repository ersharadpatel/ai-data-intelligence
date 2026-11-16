exec(open("tests/run_cross_validation.py").read().replace(
    'for name, estimator in models.items():',
    'models = {"All Features": models["Logistic Regression"], "Without previous_purchases": models["Logistic Regression"]}\n\nfor name, estimator in models.items():'
).replace(
    'pipeline = Pipeline([',
    'X_eval = X_train.drop(columns=["previous_purchases"]) if name == "Without previous_purchases" else X_train\n    current_numeric = X_eval.select_dtypes(include="number").columns.tolist()\n    current_categorical = X_eval.select_dtypes(include=["object", "category"]).columns.tolist()\n    current_preprocessor = ColumnTransformer([\n        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), current_numeric),\n        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), current_categorical),\n    ], remainder="drop")\n    pipeline = Pipeline(['
).replace(
    '("preprocessor", preprocessor),',
    '("preprocessor", current_preprocessor),'
).replace(
    '        X_train,\n        y_train,',
    '        X_eval,\n        y_train,'
))
