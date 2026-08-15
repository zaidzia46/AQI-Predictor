import pandas as pd
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
FEATURES_FILE = "data/processed/aqi_weather_data_features.csv"

FEATURE_GROUP_NAME = "aqi_lahore_daily_features"
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_DESCRIPTION = "Daily aggregated AQI, pollutant, and weather features for Lahore, with 3-day-ahead targets"
# -----------------------------------------


def push_to_hopsworks():
    df = pd.read_csv(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df)} rows from {FEATURES_FILE}")

    # Hopsworks feature groups need a "primary key" column that uniquely
    # identifies each row. Since each row is exactly one day, timestamp works.
    project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=FEATURE_GROUP_DESCRIPTION,
        primary_key=["timestamp"],
        event_time="timestamp",
    )

    feature_group.insert(df, write_options={"wait_for_job": True})
    print(f"Pushed {len(df)} rows to Hopsworks feature group '{FEATURE_GROUP_NAME}' (v{FEATURE_GROUP_VERSION})")


if __name__ == "__main__":
    push_to_hopsworks()