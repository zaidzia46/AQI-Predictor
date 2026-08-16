import pandas as pd
import subprocess
import os

INPUT_FILE = "data/processed/aqi_weather_data_features.csv"
FEAST_DATA_FILE = "feature_repo/data/aqi_features.parquet"
FEATURE_REPO_DIR = "feature_repo"


def prepare_feast_data():
    os.makedirs("feature_repo/data", exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)

    df["city_id"] = "lahore"

    df.to_parquet(FEAST_DATA_FILE, index=False)
    print(f"Saved {len(df)} rows to {FEAST_DATA_FILE} (Feast's data source)")


def apply_feast_definitions():
    result = subprocess.run(
        ["feast", "apply"],
        cwd=FEATURE_REPO_DIR,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("feast apply failed")
    print("Feast feature definitions registered successfully.")


if __name__ == "__main__":
    prepare_feast_data()
    apply_feast_definitions()