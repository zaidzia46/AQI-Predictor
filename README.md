# Lahore AQI Forecast

A machine-learning app that predicts the **US Air Quality Index (AQI)** for
Lahore for the **next 3 days**. It collects live weather and pollution data,
retrains itself every day, and shows the forecast on a public dashboard.

## What it does

- Fetches live weather + pollutant data every hour (Open-Meteo).
- Builds daily features and computes the US EPA AQI (mostly driven by PM2.5).
- Trains one model per horizon (Day +1, +2, +3) and picks the best one.
- Shows the current air quality and 3-day forecast on a Streamlit dashboard,
  with a health alert when dangerous AQI is expected.

## Live dashboard

The app is deployed on **Streamlit Community Cloud**. See
[dashboard/README.md](dashboard/README.md) for the deploy steps and public URL.

Run it locally:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## How it works

Everything is automated with two GitHub Actions workflows:

```
Hourly:  fetch data -> build features -> clean -> push to Feast -> commit
Daily:   train models -> commit model + metrics -> dashboard redeploys
```

- **Hourly** ([`fetch_data.yml`](.github/workflows/fetch_data.yml)) — adds the
  latest reading and rebuilds the feature files.
- **Daily** ([`train_model.yml`](.github/workflows/train_model.yml)) — retrains
  the models and commits them to the repo.
- The dashboard loads the committed model with a plain file path — no separate
  server, no manual upload.

## The model

- **Task:** forecast the daily AQI 1, 2 and 3 days ahead (one model per day).
- **Algorithm:** RandomForest vs XGBoost, chosen per horizon by time-series
  cross-validation.
- **Features (37):** current pollutants & weather, season (day-of-year), plus
  AQI/PM2.5 lags and rolling averages.
- **Split:** the most recent 20% of days are held out for testing (kept in time
  order, so the model never sees the future during training).
- **Explainability:** SHAP shows which features drive each forecast.

### Performance (held-out test set)

| Horizon | R²   | MAE (AQI points) |
|---------|------|------------------|
| Day +1  | 0.67 | 17.6             |
| Day +2  | 0.54 | 22.2             |
| Day +3  | 0.51 | 23.9             |

These are honest 3-day-ahead numbers on a time-ordered test set. Accuracy is
highest for Day +1 and drops further out, which is expected.

## Project structure

```
src/                 Data + model pipeline
  fetch_data.py        Get live weather + pollutants
  feature_engineering.py  Build daily features
  clean_data.py        Clean the data
  aqi_calc.py          US EPA AQI formula
  push_to_feast.py     Save features to the Feast store
  train_model.py       Train and evaluate the models
dashboard/app.py     Streamlit dashboard (the deliverable)
feature_repo/        Feast feature store config
models/              Trained model + metadata (committed)
data/processed/      Feature and training CSVs
.github/workflows/   Hourly + daily automation
```

## Setup

```bash
git clone https://github.com/zaidzia46/AQI-Predictor.git
cd AQI-Predictor
python -m venv .venv
.venv/Scripts/activate      # Windows  (use: source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
```

## Data source

Weather and air-quality data from [Open-Meteo](https://open-meteo.com)
(ERA5 weather archive + CAMS air quality). AQI is calculated with the US EPA
formula. Forecasts are model estimates, not official measurements.
