# Lahore AQI Forecast — Streamlit dashboard

Predicts the US AQI for Lahore for the next 3 days, using the models that the
GitHub Actions pipeline trains and commits to this repo.

- **App:** [`dashboard/app.py`](app.py)
- **Reads:** `models/aqi_models.joblib`, `models/aqi_models_metadata.json`,
  `data/processed/latest_features.csv`, `data/processed/aqi_weather_data_features.csv`
- **Does not** fetch data or train — it only loads what the workflows commit.

## How the model reaches the app

The trained model is a **file committed to this GitHub repo**
(`models/aqi_models.joblib`, ~3 MB). Streamlit Community Cloud runs the app
directly from the repo, so it loads that file with a plain path — no upload,
no separate server. The daily "Train AQI Model" workflow commits a fresh model;
Streamlit redeploys on each push, so the dashboard always serves the latest.

```
GitHub Actions (hourly data + daily retrain) --commit--> repo --pull--> Streamlit Cloud --> app
```

## Run locally

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Then open http://localhost:8501.

## Deploy to Streamlit Community Cloud (public URL)

1. Push this repo to GitHub (make sure `dashboard/app.py`, `dashboard/requirements.txt`,
   `.streamlit/config.toml`, and the `models/` + `data/processed/` files are committed).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. **Create app → Deploy a public app from GitHub** and set:
   - **Repository:** `zaidzia46/AQI-Predictor`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
4. Open **Advanced settings** and set **Python version 3.11** (matches the training
   environment, so the saved model unpickles cleanly).
5. Click **Deploy**. The first build takes a few minutes. Your public URL will look
   like `https://<your-app-name>.streamlit.app` — that's the URL to submit.

Streamlit installs from `dashboard/requirements.txt` (the file nearest the app),
which is intentionally slim — it does **not** pull in `feast`/`xgboost`, so the
build is fast and reliable.

### Keeping it fresh
The app reboots and re-reads the committed files whenever the pipeline pushes
(hourly data, daily model). No manual step is needed. You can also click
**"Reboot app"** from the Streamlit dashboard to force a refresh.
