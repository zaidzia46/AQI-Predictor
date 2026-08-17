"""
Train 3-day-ahead AQI forecasting models, using historical features
retrieved from the local Feast feature store (point-in-time correct
join), rather than reading the processed CSV directly.

Flow:
    1. Build an entity dataframe (city_id, event_timestamp) from the rows
       that are known-trainable (data/processed/aqi_weather_data_train.csv
       already tells us which timestamps have complete features + targets).
    2. Ask Feast for a curated feature set (see CURATED_FEATURES) for those
       entity rows via get_historical_features().
    3. Chronological train/test split (never shuffle time series data).
    4. For EACH horizon (day1/day2/day3) independently: run
       RandomizedSearchCV with TimeSeriesSplit cross-validation across two
       model families (RandomForest, LightGBM), and keep whichever
       family/hyperparameters had the best cross-validated R2. Different
       horizons often favor different models/hyperparameters as the
       signal-to-noise ratio drops with distance, so this is a genuine
       per-horizon model-selection pipeline rather than one fixed algorithm.
    5. Evaluate the chosen model per horizon on the held-out test set.
    6. Save all three fitted models (as a dict, keyed by target column)
       plus metadata (feature list, chosen algorithm/params, metrics) to
       models/ for later use by a prediction/serving script.
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from feast import FeatureStore
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# ---- Paths -------------------------------------------------------------
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
TEST_SIZE_FRACTION = 0.2  # last 20% of the timeline held out for testing
CV_SPLITS = 5
SEARCH_ITER = 20  # RandomizedSearchCV iterations per model family per horizon

# --- Candidate models for per-horizon selection --------------------------
# Benchmarking showed no single algorithm wins on every horizon (LightGBM
# tends to win day1, tuned RandomForest tends to win day2/day3 - the
# signal-to-noise ratio drops as the horizon lengthens, which favors more
# heavily regularized trees). Rather than hardcoding a winner picked by eye
# off one 145-row test split, each horizon runs its own RandomizedSearchCV
# across both families using TimeSeriesSplit cross-validation on the
# training set only, and whichever wins on CV R2 is what actually gets
# evaluated on the held-out test set.
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
    "LightGBM": (
        lambda: LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1),
        {
            "n_estimators": [100, 200, 300],
            "num_leaves": [7, 15, 31],
            "min_child_samples": [10, 15, 25],
            "learning_rate": [0.02, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        },
    ),
}

# --- Feature selection ---------------------------------------------------
# feature_engineering.py now generates lag/rolling/change-rate features for
# EVERY pollutant + weather variable (115 features), and the Feast schema
# (future_definitions.py) tracks all of them automatically. But benchmarking
# on our current data volume (722 rows) showed the FULL feature set
# overfits badly: RandomForest/XGBoost/Ridge all saw R2 go negative on the
# 2- and 3-day-ahead targets when trained on all 115 features. This curated
# subset (aqi/pm25 lag+rolling, all raw pollutants/weather, time features)
# was the most robust across all three horizons in that comparison.
#
# The full feature set stays available in Feast for later experiments --
# once we have more years of history (see backfill_data.py), it's worth
# re-benchmarking with the wider set, since more rows should let the model
# support more features without overfitting.
CURATED_FEATURES = [
    "aqi", "pm25", "pm10", "co", "so2", "no2", "o3",
    "temperature", "humidity", "pressure", "wind_speed", "clouds",
    "day", "month", "day_of_week",
    "aqi_lag_1d", "aqi_lag_2d", "aqi_lag_3d", "aqi_lag_7d",
    "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d",
    "aqi_rolling_mean_3d", "aqi_rolling_mean_7d", "aqi_rolling_mean_14d",
    "pm25_rolling_mean_3d", "pm25_rolling_mean_7d", "pm25_rolling_mean_14d",
    "aqi_change_rate",
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

    # Point-in-time joins can leave NaNs if any requested timestamp falls
    # outside the source's coverage - drop those defensively.
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
        "model_type": "per-horizon model selection (RandomForest vs LightGBM via TimeSeriesSplit CV)",
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