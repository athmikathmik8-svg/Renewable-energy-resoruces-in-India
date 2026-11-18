from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "renewable_energy_dataset.csv"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CAPACITY_COLUMNS = [
    "Installed - Bio-Mass Power",
    "Installed - Small Hydro Power",
    "Installed - Solar Power",
    "Installed - Wind Power",
    "Installed - Waste to Energy",
]

TARGET = "Total_Renewable_Capacity"
PREVIOUS_TOTAL = "Previous Total Capacity"
LAG_COLUMNS = [f"Previous Year - {col}" for col in CAPACITY_COLUMNS]

# ---------------------------------------------------------------------
# 1. Load and validate source data
# ---------------------------------------------------------------------
raw_df = pd.read_csv(DATA_PATH)
source_shape = raw_df.shape
required_columns = [
    "State", "Year", "Population", "GSDP (crore INR)", *CAPACITY_COLUMNS
]
missing_columns = [col for col in required_columns if col not in raw_df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# The five installed-capacity columns are summed to create the prediction target.
raw_df[TARGET] = raw_df[CAPACITY_COLUMNS].sum(axis=1)

# ---------------------------------------------------------------------
# 2. Preprocessing and lag feature engineering
# ---------------------------------------------------------------------
model_df = raw_df.sort_values(["State", "Year"]).copy()
for source_col, lag_col in zip(CAPACITY_COLUMNS, LAG_COLUMNS):
    model_df[lag_col] = model_df.groupby("State")[source_col].shift(1)
model_df[PREVIOUS_TOTAL] = model_df.groupby("State")[TARGET].shift(1)
model_df["Capacity Change"] = model_df[TARGET] - model_df[PREVIOUS_TOTAL]

# First observed year for each state has no previous-year capacity, so it cannot
# be used for a one-year-ahead model. Keep all valid state-year rows thereafter.
model_df = model_df.dropna(subset=[*LAG_COLUMNS, PREVIOUS_TOTAL]).copy()

FEATURES = [
    "State", "Year", "Population", "GSDP (crore INR)", PREVIOUS_TOTAL, *LAG_COLUMNS
]
NUMERIC_FEATURES = [
    "Year", "Population", "GSDP (crore INR)", PREVIOUS_TOTAL, *LAG_COLUMNS
]
CATEGORICAL_FEATURES = ["State"]

# ---------------------------------------------------------------------
# 3. Chronological train/test split: no random row shuffling
# ---------------------------------------------------------------------
train_df = model_df[model_df["Year"] <= 2021].copy()
test_df = model_df[model_df["Year"] >= 2022].copy()
if train_df.empty or test_df.empty:
    raise ValueError("Expected data from both the training period (<=2021) and test period (>=2022).")

X_train = train_df[FEATURES]
y_train_change = train_df["Capacity Change"]
X_test = test_df[FEATURES]
y_test_total = test_df[TARGET]
y_test_change = test_df["Capacity Change"]

# Numeric values are median-imputed; State is imputed (if ever required) and
# one-hot encoded. The tree-based estimator does not need feature scaling.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
])

# The model learns the year-over-year change. Its forecast is added to the
# previous year's actual total capacity to produce next-year total capacity.
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=4,
        loss="huber",
        random_state=42,
    )),
])
model_pipeline.fit(X_train, y_train_change)

predicted_change = model_pipeline.predict(X_test)
previous_capacity = test_df[PREVIOUS_TOTAL].to_numpy()
model_predictions = np.maximum(previous_capacity + predicted_change, 0)
persistence_predictions = previous_capacity.copy()

# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------
def calculate_metrics(actual, predicted, model_name):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
        "WAPE (%)": np.abs(actual - predicted).sum() / np.maximum(np.abs(actual).sum(), 1e-9) * 100,
    }

metrics_df = pd.DataFrame([
    calculate_metrics(y_test_total, persistence_predictions, "Persistence baseline (last year = this year)"),
    calculate_metrics(y_test_total, model_predictions, "Gradient Boosting (predict capacity change)"),
])
metrics_df.to_csv(BASE_DIR / "renewable_energy_model_metrics.csv", index=False)

predictions_df = test_df[["State", "Year", TARGET, PREVIOUS_TOTAL]].copy()
predictions_df["Predicted Capacity Change"] = predicted_change
predictions_df["Model Predicted Total Capacity"] = model_predictions
predictions_df["Persistence Baseline Prediction"] = persistence_predictions
predictions_df["Model Absolute Error"] = np.abs(
    predictions_df[TARGET] - predictions_df["Model Predicted Total Capacity"]
)
predictions_df["Baseline Absolute Error"] = np.abs(
    predictions_df[TARGET] - predictions_df["Persistence Baseline Prediction"]
)
predictions_df.to_csv(BASE_DIR / "renewable_energy_backtest_predictions.csv", index=False)

# ---------------------------------------------------------------------
# 5. Permutation importance on held-out change values
# ---------------------------------------------------------------------
importance = permutation_importance(
    model_pipeline,
    X_test,
    y_test_change,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Mean importance": importance.importances_mean,
    "Importance standard deviation": importance.importances_std,
}).sort_values("Mean importance", ascending=False)
importance_df.to_csv(BASE_DIR / "renewable_energy_feature_importance.csv", index=False)

# ---------------------------------------------------------------------
# 6. Export prepared data and final model trained on all historical rows
# ---------------------------------------------------------------------
model_df.to_csv(BASE_DIR / "renewable_energy_preprocessed.csv", index=False)

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=4,
        loss="huber",
        random_state=42,
    )),
])
final_model.fit(model_df[FEATURES], model_df["Capacity Change"])
joblib.dump({
    "model": final_model,
    "feature_columns": FEATURES,
    "capacity_columns": CAPACITY_COLUMNS,
    "target_column": TARGET,
    "previous_total_column": PREVIOUS_TOTAL,
    "lag_columns": LAG_COLUMNS,
    "training_data_through_year": int(model_df["Year"].max()),
}, BASE_DIR / "renewable_energy_model.joblib")

# ---------------------------------------------------------------------
# 7. Save key EDA charts
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 5))
annual_total = raw_df.groupby("Year")[TARGET].sum()
plt.plot(annual_total.index, annual_total.values, marker="o")
plt.title("Total Installed Renewable Capacity by Year")
plt.xlabel("Year")
plt.ylabel("Sum of capacity values (dataset units)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "annual_capacity_trend.png", dpi=160)
plt.close()

plt.figure(figsize=(9, 5))
source_totals = raw_df[CAPACITY_COLUMNS].sum().sort_values(ascending=False)
plt.bar(source_totals.index, source_totals.values)
plt.title("Total Installed Capacity by Renewable Technology")
plt.xlabel("Technology")
plt.ylabel("Sum of capacity values (dataset units)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "capacity_by_technology.png", dpi=160)
plt.close()

latest_year = int(raw_df["Year"].max())
top_states = (raw_df[raw_df["Year"] == latest_year]
              .nlargest(10, TARGET)
              .sort_values(TARGET))
plt.figure(figsize=(9, 6))
plt.barh(top_states["State"], top_states[TARGET])
plt.title(f"Top 10 States/Regions by Total Capacity ({latest_year})")
plt.xlabel("Total capacity (dataset units)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "top_states_latest_year.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(raw_df[TARGET], bins=35, edgecolor="black")
plt.title("Distribution of Total Renewable Capacity")
plt.xlabel("Total capacity (dataset units)")
plt.ylabel("Number of state-year records")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "target_distribution.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(y_test_total, model_predictions, alpha=0.7)
limit = max(float(np.max(y_test_total)), float(np.max(model_predictions)))
plt.plot([0, limit], [0, limit], linestyle="--")
plt.title("Actual vs Predicted Capacity (2022–2024 Test Set)")
plt.xlabel("Actual capacity (dataset units)")
plt.ylabel("Predicted capacity (dataset units)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=160)
plt.close()

# ---------------------------------------------------------------------
# 8. Print concise run summary
# ---------------------------------------------------------------------
print("ORIGINAL SOURCE DATA SHAPE:", source_shape)
print("UNIQUE STATES:", raw_df["State"].nunique())
print("YEAR RANGE:", int(raw_df["Year"].min()), "to", int(raw_df["Year"].max()))
print("MISSING CELLS:", int(raw_df.isna().sum().sum()))
print("DUPLICATE STATE-YEAR PAIRS:", int(raw_df.duplicated(["State", "Year"]).sum()))
print("PREPARED MODELING ROWS:", len(model_df))
print("TRAIN ROWS:", len(train_df), "TEST ROWS:", len(test_df))
print("\nMODEL METRICS:")
print(metrics_df.round(3).to_string(index=False))
print("\nTOP PERMUTATION IMPORTANCE FEATURES:")
print(importance_df.head(6).round(3).to_string(index=False))
print("\nOUTPUT FILES WRITTEN TO:", BASE_DIR)