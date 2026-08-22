import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from feast import FeatureStore
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

TRAIN_ENTITY_SOURCE = "data/processed/aqi_weather_data_train.csv"
FEATURE_REPO_DIR = "feature_repo"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "aqi_models.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "aqi_models_metadata.json")

# ---- Feast feature view / entity names ----------------------------------
FEATURE_VIEW = "aqi_daily_features"
ENTITY_COLUMN = "city_id"

TARGET_COLUMNS = ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]

RANDOM_STATE = 42
TEST_SIZE_FRACTION = 0.2
CV_SPLITS = 5
SEARCH_ITER = 20

# --- Candidate models for per-horizon selection --------------------------
# No single algorithm wins on every horizon, so each horizon runs its own
# RandomizedSearchCV across both families using TimeSeriesSplit CV on the
# training set only, and whichever wins on CV R2 is evaluated on the held-out
# test set. On the current data RandomForest wins all three horizons, but
# XGBoost is close on day1 and kept as a genuine competitor.
#
# (Note: an earlier version imported LightGBM here, which is not in
# requirements.txt and crashed on import; replaced with XGBoost, which is
# installed and benchmarked competitively.)
CANDIDATE_MODELS = {
    "RandomForest": (
        lambda: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [2, 4, 8, 15],
            "max_features": ["sqrt", 0.5, 0.7],
        },
    ),
    "XGBoost": (
        lambda: XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [300, 500, 700],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.02, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        },
    ),
}

CURATED_FEATURES = [
    "aqi", "pm25", "pm10", "co", "so2", "no2", "o3",
    "temperature", "humidity", "pressure", "wind_speed", "clouds",
    "month", "doy_sin", "doy_cos",
    "aqi_lag_1d", "aqi_lag_2d", "aqi_lag_3d", "aqi_lag_7d",
    "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d",
    "aqi_rolling_mean_3d", "aqi_rolling_mean_7d", "aqi_rolling_mean_14d",
    "pm25_rolling_mean_3d", "pm25_rolling_mean_7d", "pm25_rolling_mean_14d",
    "aqi_change_rate",
    "wind_speed_roll_3d", "temperature_roll_3d", "humidity_roll_3d",
    "pressure_roll_3d", "wind_speed_roll_7d", "temperature_roll_7d",
    "humidity_roll_7d",
]


def get_feature_fields(store):
    """
    Validate CURATED_FEATURES against what's actually registered in the
    Feast feature view, and return the curated list to use for training.
    Raises loudly if a curated feature no longer exists in the store (e.g.
    feature_engineering.py renamed/removed a column) rather than silently
    training on a stale/missing feature.
    """
    fv = store.get_feature_view(FEATURE_VIEW)
    # fv.schema includes the entity join key (city_id); fv.features is the
    # actual queryable feature list with entity keys excluded.
    available = {f.name for f in fv.features}

    missing = [f for f in CURATED_FEATURES if f not in available]
    if missing:
        raise ValueError(
            f"CURATED_FEATURES references fields not found in the "
            f"'{FEATURE_VIEW}' feature view: {missing}. Did feature_engineering.py "
            f"change column names? Available fields: {sorted(available)}"
        )
    return CURATED_FEATURES


def build_entity_df():
    """
    Build the entity dataframe Feast needs: one row per (city_id,
    event_timestamp) we want features for. We source the timestamps from
    the already-cleaned train CSV, since those are exactly the rows that
    are known to have complete features AND complete targets.
    """
    df = pd.read_csv(TRAIN_ENTITY_SOURCE, usecols=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns={"timestamp": "event_timestamp"})
    df[ENTITY_COLUMN] = "lahore"
    return df[["event_timestamp", ENTITY_COLUMN]]


def fetch_training_data():
    """Retrieve point-in-time correct historical features from Feast."""
    entity_df = build_entity_df()
    print(f"Built entity dataframe with {len(entity_df)} rows.")

    store = FeatureStore(repo_path=FEATURE_REPO_DIR)
    feature_fields = get_feature_fields(store)
    print(f"Discovered {len(feature_fields)} feature fields from the Feast feature view.")

    feature_refs = [f"{FEATURE_VIEW}:{f}" for f in feature_fields + TARGET_COLUMNS]

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()

    print(f"Retrieved {len(training_df)} rows x {training_df.shape[1]} columns from Feast.")

    before = len(training_df)
    training_df = training_df.dropna(subset=feature_fields + TARGET_COLUMNS).reset_index(drop=True)
    dropped = before - len(training_df)
    if dropped:
        print(f"Dropped {dropped} rows with incomplete features/targets after the Feast join.")

    training_df = training_df.sort_values("event_timestamp").reset_index(drop=True)
    return training_df, feature_fields


def chronological_split(df, test_fraction=TEST_SIZE_FRACTION):
    """
    Time-series-safe split: the most recent `test_fraction` of rows become
    the test set. A random shuffle split would leak future information
    (via lag/rolling features) into training and inflate accuracy.
    """
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def select_best_model_for_target(X_train, y_train, target_name):
    """
    Run RandomizedSearchCV (TimeSeriesSplit CV) for every candidate model
    family on this single target, and return whichever family/config had
    the best mean cross-validated R2 - refit on the full training set.
    """
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    best_score = -np.inf
    best_model = None
    best_family = None
    best_params = None

    for family_name, (estimator_fn, param_dist) in CANDIDATE_MODELS.items():
        search = RandomizedSearchCV(
            estimator_fn(),
            param_distributions=param_dist,
            n_iter=SEARCH_ITER,
            cv=tscv,
            scoring="r2",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        print(f"  {target_name} / {family_name}: best CV R2={search.best_score_:.3f}  params={search.best_params_}")

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_family = family_name
            best_params = search.best_params_

    return best_model, best_family, best_params, best_score


def train_and_evaluate(train_df, test_df, feature_fields):
    X_train = train_df[feature_fields]
    X_test = test_df[feature_fields]

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows...")

    models = {}
    metrics = {}
    selection_info = {}

    for col in TARGET_COLUMNS:
        print(f"\nSelecting model for {col}...")
        model, family, params, cv_r2 = select_best_model_for_target(X_train, train_df[col], col)
        models[col] = model
        selection_info[col] = {"algorithm": family, "params": params, "cv_r2": round(float(cv_r2), 4)}

        preds = model.predict(X_test)
        mae = mean_absolute_error(test_df[col], preds)
        rmse = float(np.sqrt(mean_squared_error(test_df[col], preds)))
        r2 = r2_score(test_df[col], preds)
        metrics[col] = {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 4)}
        print(f"  -> chosen: {family}  |  test MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

    return models, metrics, selection_info


def save_artifacts(models, metrics, selection_info, train_df, test_df, feature_fields):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(models, MODEL_PATH)
    print(f"\nSaved per-horizon model dict to {MODEL_PATH}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "per-horizon model selection (RandomForest vs XGBoost via TimeSeriesSplit CV)",
        "targets": TARGET_COLUMNS,
        "feature_columns": feature_fields,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "train_date_range": [
            str(train_df["event_timestamp"].min()),
            str(train_df["event_timestamp"].max()),
        ],
        "test_date_range": [
            str(test_df["event_timestamp"].min()),
            str(test_df["event_timestamp"].max()),
        ],
        "metrics": metrics,
        "model_selection": selection_info,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training metadata to {METADATA_PATH}")


def main():
    training_df, feature_fields = fetch_training_data()
    train_df, test_df = chronological_split(training_df)
    models, metrics, selection_info = train_and_evaluate(train_df, test_df, feature_fields)
    save_artifacts(models, metrics, selection_info, train_df, test_df, feature_fields)


if __name__ == "__main__":
    main()