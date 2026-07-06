print("### TESTMARKER 12345 ###")

# ============================================================


# Random Forest & XGBoost - Airline Ticket Prices vs Oil & Fuel Costs
# Forecast-Variante: Preisänderung 12 Monate in der Zukunft
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

RANDOM_STATE = 42
WANDB_PROJECT = "airline-price-prediction"  # mit Marie/Kollege absprechen, gleicher Name für alle!

# ------------------------------------------------------------
# 1. Daten laden
# ------------------------------------------------------------
df = pd.read_csv("airline_ticket_prices.csv")

df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
df["year"] = df["month"].dt.year
df["month_num"] = df["month"].dt.month

group_cols = ["airline", "route_class"]
df = df.sort_values(group_cols + ["month"]).reset_index(drop=True)

# ------------------------------------------------------------
# 1b. Lag-Features
# ------------------------------------------------------------
lag_source_cols = [
    "brent_crude_usd", "jet_fuel_usd_barrel",
    "fuel_surcharge_usd", "fuel_cost_pct_opex",
]
lag_steps = [1, 3, 6]

lag_feature_cols = []
for col in lag_source_cols:
    for lag in lag_steps:
        new_col = f"{col}_lag{lag}"
        df[new_col] = df.groupby(group_cols)[col].shift(lag)
        lag_feature_cols.append(new_col)

# ------------------------------------------------------------
# 1e. Forecast-Target: Preisänderung 12 Monate in der Zukunft
# ------------------------------------------------------------
FORECAST_HORIZON = 12
target = "target_future_12m"
df[target] = df.groupby(group_cols)["yoy_price_change_pct"].shift(-FORECAST_HORIZON)

df = df.dropna(subset=[target] + lag_feature_cols).reset_index(drop=True)

print("Shape nach Drop (inkl. Lag-NaNs):", df.shape)
print("Verwendetes Target:", target)

# ------------------------------------------------------------
# 2. Features / Target festlegen
# ------------------------------------------------------------
categorical_features = [
    "airline", "country", "region", "airline_type",
    "route_class", "conflict_phase"
]

numerical_features = [
    "avg_route_km", "base_fare_usd", "fuel_surcharge_usd", "taxes_fees_usd",
    "brent_crude_usd", "jet_fuel_usd_barrel",
    "load_factor_pct", "fuel_cost_pct_opex", "year", "month_num",
] + lag_feature_cols

X = df[categorical_features + numerical_features]
y = df[target]

# ------------------------------------------------------------
# 3. Train/Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("Train:", X_train.shape, " Test:", X_test.shape)

# ------------------------------------------------------------
# 4. Preprocessing-Pipeline
# ------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

# ============================================================
# 5. Random Forest (Baseline-Modell)
# ============================================================
wandb.init(
    project=WANDB_PROJECT,
    name="random_forest_forecast",
    config={"model_type": "RandomForest", "target": target, "forecast_horizon": FORECAST_HORIZON},
)

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
])

rf_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 5],
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

rf_grid.fit(X_train, y_train)

print("\nBeste RF-Parameter:", rf_grid.best_params_)
rf_best = rf_grid.best_estimator_
wandb.config.update(rf_grid.best_params_)

y_pred_rf = rf_best.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)

print(f"Random Forest -> MSE: {rf_mse:.3f} | MAE: {rf_mae:.3f} | R2: {rf_r2:.3f} | RMSE: {rf_rmse:.3f}")

feature_names = rf_best.named_steps["preprocessor"].get_feature_names_out()
importances = rf_best.named_steps["model"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(data=fi_df, x="importance", y="feature")
plt.title("Random Forest - Top 15 Feature Importances (Forecast Target)")
plt.tight_layout()
plt.savefig("rf_feature_importance_forecast.png", dpi=150)
plt.close()

# Robustheits-Check ohne COVID (siehe weiter unten, hier vorgezogen für RF-Logging)
mask_no_covid = X_test["conflict_phase"] != "COVID-19 Collapse"
r2_no_covid_rf = r2_score(y_test[mask_no_covid], y_pred_rf[mask_no_covid])
mae_no_covid_rf = mean_absolute_error(y_test[mask_no_covid], y_pred_rf[mask_no_covid])

wandb.log({
    "mse": rf_mse,
    "mae": rf_mae,
    "r2": rf_r2,
    "rmse": rf_rmse,
    "r2_no_covid": r2_no_covid_rf,
    "mae_no_covid": mae_no_covid_rf,
    "feature_importance": wandb.Table(dataframe=fi_df),
})
wandb.log({"feature_importance_plot": wandb.Image("rf_feature_importance_forecast.png")})
wandb.finish()

# ============================================================
# 6. XGBoost (Verbessertes Modell)
# ============================================================
wandb.init(
    project=WANDB_PROJECT,
    name="xgboost_forecast",
    config={"model_type": "XGBoost", "target": target, "forecast_horizon": FORECAST_HORIZON},
)

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
    ))
])

xgb_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 6, 10],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample": [0.8, 1.0],
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

xgb_grid.fit(X_train, y_train)

print("\nBeste XGBoost-Parameter:", xgb_grid.best_params_)
xgb_best = xgb_grid.best_estimator_
wandb.config.update(xgb_grid.best_params_)

y_pred_xgb = xgb_best.predict(X_test)

xgb_mse = mean_squared_error(y_test, y_pred_xgb)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_r2 = r2_score(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(xgb_mse)

print(f"XGBoost -> MSE: {xgb_mse:.3f} | MAE: {xgb_mae:.3f} | R2: {xgb_r2:.3f} | RMSE: {xgb_rmse:.3f}")
xgb_importances = xgb_best.named_steps["model"].feature_importances_
fi_xgb_df = pd.DataFrame({"feature": feature_names, "importance": xgb_importances})
fi_xgb_df = fi_xgb_df.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(data=fi_xgb_df, x="importance", y="feature")
plt.title("XGBoost - Top 15 Feature Importances (Forecast Target)")
plt.tight_layout()
plt.savefig("xgb_feature_importance_forecast.png", dpi=150)
plt.close()

r2_no_covid_xgb = r2_score(y_test[mask_no_covid], y_pred_xgb[mask_no_covid])
mae_no_covid_xgb = mean_absolute_error(y_test[mask_no_covid], y_pred_xgb[mask_no_covid])

wandb.log({
    "mse": xgb_mse,
    "mae": xgb_mae,
    "r2": xgb_r2,
    "rmse": xgb_rmse,
    "r2_no_covid": r2_no_covid_xgb,
    "mae_no_covid": mae_no_covid_xgb,
    "feature_importance": wandb.Table(dataframe=fi_xgb_df),
})
wandb.log({"feature_importance_plot": wandb.Image("xgb_feature_importance_forecast.png")})
wandb.finish()

# ============================================================
# 7. Modellvergleich (lokal, unverändert)
# ============================================================
results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "MSE": [rf_mse, xgb_mse],
    "MAE": [rf_mae, xgb_mae],
    "R2": [rf_r2, xgb_r2],
})
print("\n=== Modellvergleich (Forecast Target) ===")
print(results.to_string(index=False))

print("\n=== Robustheits-Check: ohne COVID-19 Collapse ===")
print(f"RF  -> R2: {r2_no_covid_rf:.3f} | MAE: {mae_no_covid_rf:.3f}  (gesamt R2: {rf_r2:.3f})")
print(f"XGB -> R2: {r2_no_covid_xgb:.3f} | MAE: {mae_no_covid_xgb:.3f}  (gesamt R2: {xgb_r2:.3f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_pred, name in zip(axes, [y_pred_rf, y_pred_xgb], ["Random Forest", "XGBoost"]):
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Tatsächlich (target_future_12m)")
    ax.set_ylabel("Vorhergesagt")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("predicted_vs_actual_forecast.png", dpi=150)
plt.close()

print("\nFertig. Plots gespeichert und nach W&B geloggt.")
