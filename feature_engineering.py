import pandas as pd
import numpy as np
from aqi_calc import calculate_standard_aqi

INPUT_FILE = "aqi_weather_data.csv"
OUTPUT_FILE = "aqi_weather_data_features.csv"

FEATURE_COLUMNS = ["aqi", "pm25"]

ROLLING_WINDOWS = [3, 7, 14]

LAG_DAYS = [1, 2, 3, 7]


def load_and_aggregate_to_daily(path):
    """
    Load the raw hourly CSV and collapse it into one row per day, using
    the mean of each column across that day's hourly readings.
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df["date"] = df["timestamp"].dt.date

    pollutant_cols = ["pm25", "pm10", "co", "so2", "no2", "o3"]
    weather_cols = ["temperature", "humidity", "pressure", "wind_speed", "clouds"]

    agg_cols = [c for c in pollutant_cols + weather_cols if c in df.columns]
    daily = df.groupby("date")[agg_cols].mean().reset_index()

    daily = daily.rename(columns={"date": "timestamp"})
    daily["timestamp"] = pd.to_datetime(daily["timestamp"])
    daily = daily.sort_values("timestamp").reset_index(drop=True)

    return daily


def recalculate_daily_aqi(df):
    """
    Recalculate AQI from the DAILY AVERAGE pollutant concentrations,
    rather than averaging the already-calculated hourly AQI values.
    This matches the official EPA method more closely (24-hour average
    concentration -> breakpoint table), and avoids the mathematical
    problem of averaging an already-nonlinear index.
    """
    df["aqi"] = df.apply(
        lambda row: calculate_standard_aqi(
            pm25=row.get("pm25"), pm10=row.get("pm10"), co=row.get("co"),
            so2=row.get("so2"), no2=row.get("no2"), o3=row.get("o3"),
        ),
        axis=1,
    )
    return df


def handle_missing_and_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill()

    df = df[df["aqi"].notna()].reset_index(drop=True)

    df.loc[(df["aqi"] < 0) | (df["aqi"] > 500), "aqi"] = np.nan
    df["aqi"] = df["aqi"].ffill()

    return df


def add_time_features(df):
    """Add day-of-week and month. (No 'hour' anymore - each row is a full day.)"""
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0 = Monday
    return df


def add_lag_features(df):
    """Add lag features: value N DAYS ago, for each feature column."""
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        for lag in LAG_DAYS:
            df[f"{col}_lag_{lag}d"] = df[col].shift(lag)
    return df


def add_rolling_features(df):
    """Add rolling average features over different day windows."""
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        for window in ROLLING_WINDOWS:
            df[f"{col}_rolling_mean_{window}d"] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
    return df


def add_aqi_change_rate(df):
    """AQI change rate = how much AQI changed compared to yesterday."""
    df["aqi_change_rate"] = df["aqi"].diff()
    return df


def add_target_columns(df):
    """
    Three target columns - simply the AQI of the NEXT 3 ROWS, since each
    row is already exactly one day. No hour-counting needed anymore.
    """
    df["target_aqi_day1"] = df["aqi"].shift(-1)  # tomorrow
    df["target_aqi_day2"] = df["aqi"].shift(-2)  # day after tomorrow
    df["target_aqi_day3"] = df["aqi"].shift(-3)  # 3 days from now
    return df


def engineer_features():
    df = load_and_aggregate_to_daily(INPUT_FILE)
    print(f"Aggregated raw hourly data into {len(df)} daily rows.")

    df = recalculate_daily_aqi(df)
    df = handle_missing_and_outliers(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_aqi_change_rate(df)
    df = add_target_columns(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} daily rows with engineered features to {OUTPUT_FILE}")

    for col in ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]:
        usable_rows = df[col].notna().sum()
        print(f"{col}: {usable_rows} usable rows so far")


if __name__ == "__main__":
    engineer_features()