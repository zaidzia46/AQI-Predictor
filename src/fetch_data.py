import requests
import csv
import os
from datetime import datetime, timezone
from aqi_calc import calculate_standard_aqi

CITY_NAME = "Lahore"
LAT, LON = 31.5204, 74.3587

CSV_FILE = "data/raw/aqi_weather_data.csv"


def get_weather_data():
    """Get current weather data from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["current"]


def get_air_quality_data():
    """Get current pollutant data from Open-Meteo's Air Quality API."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "pm2_5,pm10,carbon_monoxide,sulphur_dioxide,nitrogen_dioxide,ozone",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["current"]


def collect_data():
    weather = get_weather_data()
    air_quality = get_air_quality_data()

    standard_aqi = calculate_standard_aqi(
        pm25=air_quality.get("pm2_5"),
        pm10=air_quality.get("pm10"),
        co=air_quality.get("carbon_monoxide"),
        so2=air_quality.get("sulphur_dioxide"),
        no2=air_quality.get("nitrogen_dioxide"),
        o3=air_quality.get("ozone"),
    )

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": CITY_NAME,
        "aqi": standard_aqi,
        "pm25": air_quality.get("pm2_5"),
        "pm10": air_quality.get("pm10"),
        "co": air_quality.get("carbon_monoxide"),
        "so2": air_quality.get("sulphur_dioxide"),
        "no2": air_quality.get("nitrogen_dioxide"),
        "o3": air_quality.get("ozone"),
        "temperature": weather.get("temperature_2m"),
        "humidity": weather.get("relative_humidity_2m"),
        "pressure": weather.get("surface_pressure"),
        "wind_speed": weather.get("wind_speed_10m"),
        "wind_deg": weather.get("wind_direction_10m"),
        "clouds": weather.get("cloud_cover"),
    }

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved data for {CITY_NAME} at {row['timestamp']} - AQI: {standard_aqi}")


if __name__ == "__main__":
    collect_data()