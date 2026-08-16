from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64
from feast.value_type import ValueType
from datetime import timedelta

city = Entity(name="city_id", join_keys=["city_id"], value_type=ValueType.STRING)

aqi_source = FileSource(
    name="aqi_features_source",
    path="data/aqi_features.parquet",
    timestamp_field="timestamp",
)

aqi_feature_view = FeatureView(
    name="aqi_daily_features",
    entities=[city],
    ttl=timedelta(days=3650),  # how far back features are considered valid
    schema=[
        Field(name="aqi", dtype=Float64),
        Field(name="pm25", dtype=Float64),
        Field(name="pm10", dtype=Float64),
        Field(name="co", dtype=Float64),
        Field(name="so2", dtype=Float64),
        Field(name="no2", dtype=Float64),
        Field(name="o3", dtype=Float64),
        Field(name="temperature", dtype=Float64),
        Field(name="humidity", dtype=Float64),
        Field(name="pressure", dtype=Float64),
        Field(name="wind_speed", dtype=Float64),
        Field(name="clouds", dtype=Float64),
        Field(name="day", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="aqi_change_rate", dtype=Float64),
        # Lag features - value N days ago (created by feature_engineering.py)
        Field(name="aqi_lag_1d", dtype=Float64),
        Field(name="aqi_lag_2d", dtype=Float64),
        Field(name="aqi_lag_3d", dtype=Float64),
        Field(name="aqi_lag_7d", dtype=Float64),
        Field(name="pm25_lag_1d", dtype=Float64),
        Field(name="pm25_lag_2d", dtype=Float64),
        Field(name="pm25_lag_3d", dtype=Float64),
        Field(name="pm25_lag_7d", dtype=Float64),
        # Rolling average features
        Field(name="aqi_rolling_mean_3d", dtype=Float64),
        Field(name="aqi_rolling_mean_7d", dtype=Float64),
        Field(name="aqi_rolling_mean_14d", dtype=Float64),
        Field(name="pm25_rolling_mean_3d", dtype=Float64),
        Field(name="pm25_rolling_mean_7d", dtype=Float64),
        Field(name="pm25_rolling_mean_14d", dtype=Float64),
        Field(name="target_aqi_day1", dtype=Float64),
        Field(name="target_aqi_day2", dtype=Float64),
        Field(name="target_aqi_day3", dtype=Float64),
    ],
    source=aqi_source,
)