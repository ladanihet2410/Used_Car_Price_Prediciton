
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

DATA_FILE = "dummy_used_cars.csv"
MODEL_FILE = "used_car_price_model.pkl"

data = pd.read_csv(DATA_FILE)
target = "selling_price"
features = [
    "brand", "year", "km_driven", "fuel", "transmission",
    "owner", "engine_cc", "seats"
]
X = data[features]
y = data[target]

numeric_features = ["year", "km_driven", "engine_cc", "seats"]
categorical_features = ["brand", "fuel", "transmission", "owner"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

print("\\nMODEL COMPARISON (5-FOLD CV, MAE)")
print("-" * 48)
results = []
for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    scores = cross_val_score(
        pipeline, X_train, y_train, cv=5,
        scoring="neg_mean_absolute_error"
    )
    mean_mae = -scores.mean()
    results.append((name, mean_mae))
    print(f"{name:22s} MAE: ₹{mean_mae:,.0f}")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 2]
}
search = GridSearchCV(
    rf_pipeline, param_grid, cv=5,
    scoring="neg_mean_absolute_error", n_jobs=-1
)
search.fit(X_train, y_train)


best_name, best_cv_mae = min(results, key=lambda item: item[1])
if -search.best_score_ < best_cv_mae:
    final_model = search.best_estimator_
    selected = "Tuned Random Forest"
else:
    selected_estimator = models[best_name]
    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", selected_estimator)
    ])
    final_model.fit(X_train, y_train)
    selected = best_name

predictions = final_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\\nFINAL TEST RESULTS")
print("-" * 48)
print("Selected model:", selected)
print(f"MAE : ₹{mae:,.0f}")
print(f"RMSE: ₹{rmse:,.0f}")
print(f"R²  : {r2:.3f}")

joblib.dump(final_model, MODEL_FILE)
print(f"\\nModel saved as {MODEL_FILE}")
print("You can now run: streamlit run app.py")
