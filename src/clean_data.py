import pandas as pd

INPUT_FILE = "data/processed/aqi_weather_data_features.csv"
TRAIN_OUTPUT_FILE = "data/processed/aqi_weather_data_train.csv"
LATEST_OUTPUT_FILE = "data/processed/latest_features.csv"

TARGET_COLUMNS = ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]


def clean_data():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    feature_cols = [c for c in df.columns if c not in ["timestamp"] + TARGET_COLUMNS]
    has_complete_features = df[feature_cols].notna().all(axis=1)
    has_complete_targets = df[TARGET_COLUMNS].notna().all(axis=1)
    is_trainable = has_complete_features & has_complete_targets

    train_df = df[is_trainable].reset_index(drop=True)
    train_df.to_csv(TRAIN_OUTPUT_FILE, index=False)
    print(f"Saved {len(train_df)} complete rows to {TRAIN_OUTPUT_FILE} (ready for model training)")

    rows_with_features = df[has_complete_features].reset_index(drop=True)
    if len(rows_with_features) > 0:
        latest_row = rows_with_features.tail(1)
        latest_row.to_csv(LATEST_OUTPUT_FILE, index=False)
        print(f"Saved the latest row (date: {latest_row['timestamp'].values[0]}) to {LATEST_OUTPUT_FILE}")
        print("This is the row you'll feed into the trained model to get today's real 3-day forecast.")
    else:
        print("No row yet has complete features - keep collecting data.")

    dropped = len(df) - len(train_df)
    print(f"\n{dropped} rows were excluded from training (not enough history yet, or targets not available yet). This is expected and will shrink as more days of data come in.")


if __name__ == "__main__":
    clean_data()