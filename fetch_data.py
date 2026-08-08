import requests
import csv
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

OPENAQ_KEY = os.environ.get("OPENAQ_KEY")
OPENWEATHER_KEY = os.environ.get("OPENWEATHER_KEY")

CITY_NAME = "Lahore"
LAT, LON = 31.5204, 74.3587
OPENAQ_LOCATION_ID = 1894641
OPENAQ_PM25_SENSOR_ID = 7466365

MAX_READING_AGE_HOURS = 6

CSV_FILE = "aqi_weather_data.csv"

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def calculate_sub_aqi(concentration, breakpoints):
    """
    Convert a pollutant concentration into an AQI value (0-500) using
    the EPA breakpoint table + linear interpolation formula:

        AQI = ((AQI_high - AQI_low) / (Conc_high - Conc_low)) * (Conc - Conc_low) + AQI_low
    """
    if concentration is None:
        return None

    for conc_low, conc_high, aqi_low, aqi_high in breakpoints:
        if conc_low <= concentration <= conc_high:
            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (concentration - conc_low) + aqi_low
            return round(aqi)

    if concentration > breakpoints[-1][1]:
        return 500
    return None


def calculate_standard_aqi(pm25):
    """
    Calculate the standard 0-500 AQI from PM2.5 using the EPA formula.
    """
    return calculate_sub_aqi(pm25, PM25_BREAKPOINTS)


def get_openaq_pm25():
    """
    Get the latest PM2.5 reading from our chosen OpenAQ station, but only
    if the reading is actually fresh (not a stale/frozen sensor value).
    Returns the PM2.5 value, or None if missing or too old to trust.
    """
    url = f"https://api.openaq.org/v3/locations/{OPENAQ_LOCATION_ID}/latest"
    headers = {"X-API-Key": OPENAQ_KEY}
    response = requests.get(url, headers=headers)
    data = response.json()

    now = datetime.now(timezone.utc)

    for entry in data.get("results", []):
        if entry.get("sensorsId") != OPENAQ_PM25_SENSOR_ID:
            continue

        datetime_str = entry.get("datetime", {}).get("utc")
        if not datetime_str:
            continue
        reading_time = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
        age_hours = (now - reading_time).total_seconds() / 3600

        if age_hours > MAX_READING_AGE_HOURS:
            print(f"PM2.5 reading is {age_hours:.1f} hours old (stale). Skipping.")
            return None

        return entry.get("value")

    return None


def get_weather_data(lat, lon):
    """Get current weather data (temperature, humidity, wind, etc.) from OpenWeather."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    )
    response = requests.get(url)
    return response.json()


def collect_data():
    pm25 = get_openaq_pm25()
    weather_data = get_weather_data(LAT, LON)

    if pm25 is None:
        print("Could not get fresh PM2.5 data from OpenAQ right now. Stopping this run.")
        return

    standard_aqi = calculate_standard_aqi(pm25)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": CITY_NAME,
        "aqi": standard_aqi,  # standard 0-500 AQI, calculated from PM2.5 with EPA formula
        "pm25": pm25,
        "temperature": weather_data.get("main", {}).get("temp"),
        "humidity": weather_data.get("main", {}).get("humidity"),
        "pressure": weather_data.get("main", {}).get("pressure"),
        "wind_speed": weather_data.get("wind", {}).get("speed"),
        "wind_deg": weather_data.get("wind", {}).get("deg"),
        "clouds": weather_data.get("clouds", {}).get("all"),
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