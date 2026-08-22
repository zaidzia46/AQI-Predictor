import os
import requests
import pandas as pd
from datetime import date, timedelta
from aqi_calc import calculate_standard_aqi

CITY_NAME = "Lahore"
LAT, LON = 31.5204, 74.3587

EARLIEST_AVAILABLE = "2022-08-13"
START_DATE = EARLIEST_AVAILABLE
END_DATE = (date.today() - timedelta(days=1)).isoformat()  # yesterday

OUTPUT_FILE = "data/raw/aqi_weather_data.csv"

CHUNK_DAYS = 90


def date_chunks(start_str, end_str, chunk_days):
    """Yield (chunk_start, chunk_end) date string pairs covering the full range."""
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)


def fetch_weather_chunk(start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["hourly"]


def fetch_air_quality_chunk(start, end):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": "pm2_5,pm10,carbon_monoxide,sulphur_dioxide,nitrogen_dioxide,ozone",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["hourly"]


def backfill():
    all_rows = []

    for start, end in date_chunks(START_DATE, END_DATE, CHUNK_DAYS):
        print(f"Fetching {start} to {end} ...")

        weather = fetch_weather_chunk(start, end)
        air_quality = fetch_air_quality_chunk(start, end)

        weather_df = pd.DataFrame({
            "timestamp": weather["time"],
            "temperature": weather["temperature_2m"],
            "humidity": weather["relative_humidity_2m"],
            "pressure": weather["surface_pressure"],
            "wind_speed": weather["wind_speed_10m"],
            "wind_deg": weather["wind_direction_10m"],
            "clouds": weather["cloud_cover"],
        })

        air_df = pd.DataFrame({
            "timestamp": air_quality["time"],
            "pm25": air_quality["pm2_5"],
            "pm10": air_quality["pm10"],
            "co": air_quality["carbon_monoxide"],
            "so2": air_quality["sulphur_dioxide"],
            "no2": air_quality["nitrogen_dioxide"],
            "o3": air_quality["ozone"],
        })

        merged = pd.merge(weather_df, air_df, on="timestamp", how="inner")
        all_rows.append(merged)

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"]).dt.tz_localize("UTC")
    full_df["city"] = CITY_NAME

    full_df = full_df.dropna(subset=["pm25", "pm10", "co", "so2", "no2", "o3"], how="all")

    full_df["aqi"] = full_df.apply(
        lambda row: calculate_standard_aqi(
            pm25=row["pm25"], pm10=row["pm10"], co=row["co"],
            so2=row["so2"], no2=row["no2"], o3=row["o3"],
        ),
        axis=1,
    )

    # Match the same column order/structure as fetch_data.py's output
    full_df = full_df[[
        "timestamp", "city", "aqi", "pm25", "pm10", "co", "so2", "no2", "o3",
        "temperature", "humidity", "pressure", "wind_speed", "wind_deg", "clouds"
    ]]

    # Non-destructive merge: if a raw file already exists (e.g. hourly rows
    # appended by fetch_data.py, or a prior backfill), union on the hourly
    # timestamp and keep the already-present row on any collision. This makes
    # backfill safe to re-run without clobbering live-collected data.
    if os.path.isfile(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], format="mixed", utc=True)
        combined = pd.concat([full_df, existing], ignore_index=True)
        combined["_hour"] = combined["timestamp"].dt.floor("h")
        combined = combined.drop_duplicates(subset="_hour", keep="last").drop(columns="_hour")
        full_df = combined.sort_values("timestamp").reset_index(drop=True)

    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(full_df)} historical rows to {OUTPUT_FILE}")
    print(f"Date range: {full_df['timestamp'].min()} to {full_df['timestamp'].max()}")


if __name__ == "__main__":
    backfill()