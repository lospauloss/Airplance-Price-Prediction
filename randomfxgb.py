# ============================================================
# Random Forest & XGBoost - Airline Ticket Prices vs Oil & Fuel Costs
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

RANDOM_STATE = 42

# ------------------------------------------------------------
# 1. Daten laden
# ------------------------------------------------------------
df = pd.read_csv("airline_ticket_prices.csv")

# month -> echtes Datum, daraus year/month_num extrahieren
df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
df["year"] = df["month"].dt.year
df["month_num"] = df["month"].dt.month

# WICHTIG: Eindeutiger Schlüssel im Datensatz ist (airline, route_class, month) -
# jede Airline hat pro Monat 5 Zeilen (eine je route_class). Lags MÜSSEN daher
# pro (airline, route_class) berechnet werden, sonst mischt man z.B. Long-Haul-
# Werte von letztem Monat mit Short-Haul-Werten von diesem Monat.
group_cols = ["airline", "route_class"]
df = df.sort_values(group_cols + ["month"]).reset_index(drop=True)

# ------------------------------------------------------------
# 1b. Lag-Features
#    Annahme im Proposal: Ölpreis-Änderungen wirken sich erst zeitversetzt
#    auf Ticketpreise aus (Airlines reagieren nicht sofort). Daher Lags von
#    1, 3 und 6 Monaten für die zentralen Kostentreiber.
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

# yoy_price_change_pct ist erst nach 12 Monaten Historie berechenbar, UND die
# Lag6-Spalten brauchen selbst 6 Monate Vorlauf pro Gruppe -> beide NaN-Quellen droppen
df = df.dropna(subset=["yoy_price_change_pct"] + lag_feature_cols).reset_index(drop=True)

print("Shape nach Drop (inkl. Lag-NaNs):", df.shape)

# ------------------------------------------------------------
# 2. Features / Target festlegen
# ------------------------------------------------------------
target = "yoy_price_change_pct"

categorical_features = [
    "airline", "country", "region", "airline_type",
    "route_class", "conflict_phase"
]

numerical_features = [
    "avg_route_km", "base_fare_usd", "fuel_surcharge_usd", "taxes_fees_usd",
    "brent_crude_usd", "jet_fuel_usd_barrel",
    "load_factor_pct", "fuel_cost_pct_opex", "year", "month_num",
] + lag_feature_cols
# Hinweis: total_fare_usd bewusst NICHT als Feature, da es sich direkt aus
# base_fare_usd + fuel_surcharge_usd + taxes_fees_usd zusammensetzt und damit
# potenziell zu nah am Target liegt (Data-Leakage-Risiko).

X = df[categorical_features + numerical_features]
y = df[target]

# ------------------------------------------------------------
# 3. Train/Test Split (zeitbasiert wäre für YoY-Daten saubererer,
#    hier zunächst zufällig, siehe Hinweis am Ende des Skripts)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("Train:", X_train.shape, " Test:", X_test.shape)

# ------------------------------------------------------------
# 4. Preprocessing-Pipeline
#    - kategorisch: One-Hot
#    - numerisch: unverändert (Tree-Modelle brauchen kein Scaling,
#      Scaler trotzdem drin falls später ein NN denselben Preprocessor nutzt)
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

y_pred_rf = rf_best.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)  # <-- hier einfügen

print(f"Random Forest -> MSE: {rf_mse:.3f} | MAE: {rf_mae:.3f} | R2: {rf_r2:.3f} | RMSE: {rf_rmse:.3f}")

# Feature Importance (Top 15)
feature_names = rf_best.named_steps["preprocessor"].get_feature_names_out()
importances = rf_best.named_steps["model"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(data=fi_df, x="importance", y="feature")
plt.title("Random Forest - Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150)
plt.close()

# ============================================================
# 6. XGBoost (Verbessertes Modell)
# ============================================================
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

y_pred_xgb = xgb_best.predict(X_test)

xgb_mse = mean_squared_error(y_test, y_pred_xgb)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_r2 = r2_score(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(xgb_mse)  # <-- hier einfügen

print(f"XGBoost -> MSE: {xgb_mse:.3f} | MAE: {xgb_mae:.3f} | R2: {xgb_r2:.3f} | RMSE: {xgb_rmse:.3f}")
xgb_importances = xgb_best.named_steps["model"].feature_importances_
fi_xgb_df = pd.DataFrame({"feature": feature_names, "importance": xgb_importances})
fi_xgb_df = fi_xgb_df.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(data=fi_xgb_df, x="importance", y="feature")
plt.title("XGBoost - Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150)
plt.close()

# ============================================================
# 7. Modellvergleich
# ============================================================
results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "MSE": [rf_mse, xgb_mse],
    "MAE": [rf_mae, xgb_mae],
    "R2": [rf_r2, xgb_r2],
})
print("\n=== Modellvergleich ===")
print(results.to_string(index=False))

# Predicted vs Actual Plot für beide Modelle
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_pred, name in zip(axes, [y_pred_rf, y_pred_xgb], ["Random Forest", "XGBoost"]):
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Tatsächlich (yoy_price_change_pct)")
    ax.set_ylabel("Vorhergesagt")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
plt.close()

print("\nFertig. Plots gespeichert: rf_feature_importance.png, "
      "xgb_feature_importance.png, predicted_vs_actual.png")

