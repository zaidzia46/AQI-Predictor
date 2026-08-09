"""
Feature Engineering script for AQI Prediction Project.

What this script does (matches Week 2 of the project guide):
1. Loads the raw hourly data collected by fetch_data.py.
2. Handles missing values and outliers.
3. Creates time-based features: hour, day, month.
4. Builds rolling averages and lag features for AQI and PM2.5.
5. Computes AQI change rate (how fast AQI is rising/falling).
6. Creates THREE target columns - AQI for tomorrow, the day after, and
   3 days from now (what we're trying to predict for each of the 3 days).
7. Saves the result as a new CSV, ready for model training.

How to run:
    python feature_engineering.py

Note: Since lag/rolling features need past hours, and the targets need
FUTURE hours (up to 3 days = 72 hours ahead), the first few and last few
rows of the output will have missing values in those columns. This is
normal and expected - those rows just can't be used for training yet
until more data builds up.
"""

import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
INPUT_FILE = "aqi_weather_data.csv"
OUTPUT_FILE = "aqi_weather_data_features.csv"

# We predict AQI for 3 separate future days: tomorrow, the day after,
# and 3 days from now. Since data is hourly, each "day ahead" = 24 hours.
FORECAST_DAYS_AHEAD = {
    "target_aqi_day1": 24,   # tomorrow, same hour
    "target_aqi_day2": 48,   # day after tomorrow, same hour
    "target_aqi_day3": 72,   # 3 days from now, same hour
}

# Which columns to build rolling averages / lag features for
FEATURE_COLUMNS = ["aqi", "pm25", "pm10", "co", "so2", "no2", "o3"]

# Rolling average window sizes (in hours)
ROLLING_WINDOWS = [3, 6, 24]

# Lag features (in hours) - e.g. lag_1 = value 1 hour ago
LAG_HOURS = [1, 2, 3, 24]
# -----------------------------------------


def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def handle_missing_and_outliers(df):
    """
    Handle missing values and obviously bad outlier values.
    """
    # Forward-fill small gaps (carry the last known value forward),
    # since AQI/weather don't change drastically hour to hour.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill()

    # Drop rows where AQI is still missing after filling (can't use these for training)
    df = df[df["aqi"].notna()].reset_index(drop=True)

    # Basic outlier guard: AQI must be within the real 0-500 scale.
    # Anything outside this range is not physically valid, so we treat it as missing.
    df.loc[(df["aqi"] < 0) | (df["aqi"] > 500), "aqi"] = np.nan
    df["aqi"] = df["aqi"].ffill()

    return df


def add_time_features(df):
    """Add hour, day, and month extracted from the timestamp."""
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0 = Monday
    return df


def add_lag_features(df):
    """Add lag features: value N hours ago, for each feature column."""
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        for lag in LAG_HOURS:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df):
    """Add rolling average features over different time windows."""
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        for window in ROLLING_WINDOWS:
            df[f"{col}_rolling_mean_{window}h"] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
    return df


def add_aqi_change_rate(df):
    """
    AQI change rate = how much AQI changed compared to 1 hour ago.
    Positive = getting worse, negative = getting better.
    """
    df["aqi_change_rate"] = df["aqi"].diff()
    return df


def add_target_columns(df):
    """
    Creates THREE separate target columns - one for each future day we
    want to predict: tomorrow, the day after, and 3 days from now.
    Each is just the AQI value that many hours ahead in the data.
    """
    for column_name, hours_ahead in FORECAST_DAYS_AHEAD.items():
        df[column_name] = df["aqi"].shift(-hours_ahead)
    return df


def engineer_features():
    df = load_data(INPUT_FILE)
    print(f"Loaded {len(df)} raw rows.")

    df = handle_missing_and_outliers(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_aqi_change_rate(df)
    df = add_target_columns(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows with engineered features to {OUTPUT_FILE}")

    # Quick summary so you can see how much of the data is actually
    # usable for training right now (has a real target value), per target.
    for column_name, hours_ahead in FORECAST_DAYS_AHEAD.items():
        usable_rows = df[column_name].notna().sum()
        days = hours_ahead // 24
        print(f"{column_name} (+{days} day(s)): {usable_rows} usable rows so far")
        if usable_rows == 0:
            print(
                f"  -> Need at least {hours_ahead} hours (~{days} day(s)) "
                f"of data before this target has any real values."
            )


if __name__ == "__main__":
    engineer_features()