# Lahore AQI Forecasting System

A machine learning-based system for forecasting **Lahore's Air Quality Index (AQI) for the next three days** using historical air quality and weather data.

The project combines automated data collection, data processing, feature engineering, a feature store, machine learning, model evaluation, explainability, and an interactive Streamlit dashboard into a complete end-to-end forecasting pipeline.

## 🚀 Live Dashboard

**Live Streamlit Dashboard:**
[Streamlit](https://aqi-predictor-lahore.streamlit.app/)

The dashboard provides:

* Current AQI
* AQI category and health information
* Current pollutant levels
* Weather information
* Three-day AQI forecast
* Historical AQI trends
* Model feature importance
* Prediction-related information

---

## 📌 Project Overview

Air pollution is a major environmental problem in Lahore, particularly during periods of unfavorable weather and increased pollutant concentration.

This project aims to predict future AQI values instead of only displaying current air quality. The system predicts:

* **Day +1:** Tomorrow's AQI
* **Day +2:** AQI after two days
* **Day +3:** AQI after three days

The system uses historical pollutant concentrations, weather conditions, recent AQI behavior, seasonal patterns, and rolling historical statistics as model inputs.

The complete workflow is automated so that new data can continuously enter the system and the forecasting models can be updated with recent information.

---

# 🏗️ System Architecture

The overall workflow can be summarized as:

```text
Open-Meteo
    │
    ▼
Raw Hourly Air Quality + Weather Data
    │
    ▼
Data Processing
    │
    ├── Daily Aggregation
    ├── AQI Calculation
    ├── Missing Value Handling
    └── Outlier Handling
    │
    ▼
Feature Engineering
    │
    ├── Lag Features
    ├── Rolling Features
    ├── Seasonal Features
    ├── Weather Trends
    └── AQI Change Rate
    │
    ▼
Feast Feature Store
    │
    ▼
Time-Series Model Training
    │
    ├── Random Forest
    └── XGBoost
    │
    ▼
Model Selection + Evaluation
    │
    ├── MAE
    ├── RMSE
    └── R²
    │
    ▼
Trained Models + SHAP Explanations
    │
    ▼
Streamlit Dashboard
    │
    ▼
3-Day AQI Forecast
```

---

# 📊 Data

The project uses air quality and weather data from **Open-Meteo** for Lahore.

The raw data is collected at an hourly frequency and contains major air pollutants and weather variables.

### Air Quality Variables

* PM2.5
* PM10
* CO
* SO₂
* NO₂
* O₃

### Weather Variables

* Temperature
* Humidity
* Atmospheric pressure
* Wind speed
* Cloud cover

The hourly observations are converted into daily observations before being used for forecasting.

Daily pollutant and weather values are calculated from the hourly observations, and AQI is recalculated using the daily pollutant concentrations rather than simply averaging hourly AQI values.

---

# 🧹 Data Cleaning

The data-cleaning stage prepares the processed dataset for machine learning.

The pipeline:

1. Loads the engineered feature dataset.
2. Identifies rows with complete feature values.
3. Checks whether all three future AQI targets are available.
4. Removes rows that cannot be used for model training.
5. Saves the complete training dataset.
6. Separately saves the latest complete feature row for making the current forecast.

Rows near the end of the dataset may not have future targets yet because future AQI values are naturally unavailable. These rows are therefore excluded from training but can still be used when enough features are available for prediction.

---

# ⚙️ Feature Engineering

Feature engineering is one of the most important parts of the project.

The raw daily data is transformed into features that allow the models to learn recent pollution behavior and seasonal patterns.

### AQI Lag Features

Previous AQI values are used as predictors:

* AQI from 1 day ago
* AQI from 2 days ago
* AQI from 3 days ago
* AQI from 7 days ago

### PM2.5 Lag Features

Historical PM2.5 values are also included:

* PM2.5 from 1 day ago
* PM2.5 from 2 days ago
* PM2.5 from 3 days ago
* PM2.5 from 7 days ago

### Rolling Averages

Rolling averages help the model understand recent pollution trends.

The project uses:

* 3-day AQI average
* 7-day AQI average
* 14-day AQI average
* 3-day PM2.5 average
* 7-day PM2.5 average
* 14-day PM2.5 average

### Seasonal Features

The project uses cyclical day-of-year encoding with:

* `doy_sin`
* `doy_cos`

This allows the model to represent the yearly seasonal cycle continuously rather than treating December and January as unrelated months.

### Weather Trend Features

Short-term weather trends are also included using rolling averages for:

* Wind speed
* Temperature
* Humidity
* Pressure

### AQI Change Rate

The daily change in AQI is calculated as:

```text
AQI Change Rate = Today's AQI - Yesterday's AQI
```

### Prediction Targets

Three target variables are created:

```text
target_aqi_day1 → Tomorrow's AQI
target_aqi_day2 → AQI after 2 days
target_aqi_day3 → AQI after 3 days
```

The feature engineering pipeline is designed so that the model only receives information that would have been available at prediction time, preventing future-data leakage.

---

# 🗄️ Feature Store

The project uses **Feast** as the feature store.

The processed feature dataset is stored as a Parquet file inside the Feast repository. A `city_id` field identifies the data as belonging to Lahore.

Feast is then used to retrieve historical features in a point-in-time-correct way for model training.

This provides a structured connection between:

```text
Feature Engineering
        ↓
Feast
        ↓
Model Training
```

The training pipeline validates the required feature list against the registered Feast feature view before training.

---

# 🤖 Modeling

The forecasting problem is treated as a **multi-horizon time-series regression problem**.

Instead of using one model to predict all future values, the system maintains separate prediction targets for:

```text
Day +1
Day +2
Day +3
```

The main candidate model families in the production training pipeline are:

### Random Forest

Random Forest is used as one of the main tree-based regression models. It can capture nonlinear relationships between AQI, pollutants, weather variables, and historical features.

### XGBoost

XGBoost is also evaluated as a candidate gradient-boosting model.

---

# 🔍 Model Training

The model-training pipeline retrieves historical features from Feast and sorts the data chronologically.

The dataset is split using a **time-based split**:

```text
Earlier 80% → Training
Recent 20% → Testing
```

A random shuffle is intentionally avoided because the dataset contains lagged and rolling features. Randomly mixing future observations into training could result in unrealistic evaluation.

For model selection, the pipeline uses:

* `RandomizedSearchCV`
* `TimeSeriesSplit`
* 5 cross-validation splits
* R² as the model-selection metric

Hyperparameter search is performed separately for each prediction target.

The model with the best cross-validated R² is selected for that forecasting horizon.

---

# 📏 Model Evaluation

The final models are evaluated on the unseen chronological test set using:

### R²

Measures how well the model explains the variation in future AQI values.

### MAE

Measures the average absolute difference between predicted and actual AQI.

### RMSE

Measures prediction error while giving greater weight to larger errors.

### Final Test Results

| Forecast Horizon |    R² |   MAE |  RMSE |
| ---------------- | ----: | ----: | ----: |
| Day +1           | 0.671 | 17.61 | 26.94 |
| Day +2           | 0.541 | 22.25 | 31.84 |
| Day +3           | 0.506 | 23.94 | 33.03 |

The model performs best for the next-day forecast. As the forecasting horizon increases, prediction becomes more difficult and performance decreases.

---

# 🧪 Model Improvement Experiments

After establishing the production baseline, additional experiments were performed in an isolated experimentation environment.

The purpose was to determine whether changing the model, target formulation, feature set, or ensemble strategy could produce a reliable improvement.

## Experiments Performed

### Model Families

The following approaches were investigated:

* Random Forest
* XGBoost
* Extra Trees
* HistGradientBoosting
* Ridge Regression

### Target Formulations

The experiments also tested:

* Direct AQI prediction
* Delta modeling
* Log transformation

### Ensemble Approaches

Different model combinations were investigated to determine whether combining predictions could improve performance.

### Additional Feature Engineering

Several additional feature groups were tested:

* PM10 historical features
* Weather-history features
* Volatility and momentum features
* Long-term historical features
* Feature interactions
* Feature pruning

---

# 🔬 Experiment Findings

Some approaches produced small improvements on the recent 20% holdout set.

For example, HistGradientBoosting with additional PM10 and volatility/momentum features achieved a slightly higher average R² and lower MAE on the recent holdout.

However, a single holdout improvement was not considered sufficient.

The promising approaches were therefore evaluated using an **expanding-window walk-forward backtest**.

The results showed that the differences between the approaches were small compared with the variation across different time periods.

| Approach                                   | Recent Holdout Avg. R² | Walk-Forward Avg. R² | Walk-Forward MAE |
| ------------------------------------------ | ---------------------: | -------------------: | ---------------: |
| Random Forest Baseline                     |                  0.574 |         0.438 ± 0.27 |             22.1 |
| HistGradientBoosting                       |                  0.578 |         0.423 ± 0.28 |             22.1 |
| HistGradientBoosting + Additional Features |                  0.585 |         0.425 ± 0.29 |             21.8 |

### Final Decision

The experiments did **not provide enough evidence to replace the production Random Forest model**.

Although some approaches performed slightly better on a recent test period, the improvement was not stable across multiple historical periods.

Therefore:

```text
Production Model
        ↓
Random Forest
        ↓
Retained
```

This experiment was important because it prevented the project from adopting a model based only on a small improvement from one test split.

The experiments also indicated that the existing feature set is already well-curated and that simply adding more features does not guarantee better forecasting performance.

The most promising long-term improvement is expected to come from **collecting more historical data**, particularly more observations from Lahore's winter and smog seasons.

---

# 🧠 Model Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to understand which features influence the predictions.

SHAP values are calculated for the trained tree-based models and ranked according to their mean absolute contribution.

The system stores the top feature-importance results for each forecasting horizon.

This makes the forecasting system more interpretable and helps identify which historical AQI, pollution, and weather variables have the greatest influence on predictions.

---

# 📈 Exploratory Data Analysis

Before model training, Exploratory Data Analysis is performed to understand the structure and behavior of the AQI data.

The EDA pipeline generates:

1. AQI trend over time
2. AQI distribution by month
3. AQI distribution by day of week
4. AQI and feature correlation heatmap
5. Overall AQI distribution

The analysis helps identify:

* Long-term AQI variation
* Seasonal behavior
* Monthly pollution differences
* Possible day-of-week patterns
* Relationships between AQI and pollutants
* Distribution and range of AQI values

---

# 🖥️ Streamlit Dashboard

The trained models are integrated into a Streamlit web application.

The dashboard provides an accessible interface for viewing:

* Current AQI
* AQI category
* Pollutant concentrations
* Weather conditions
* Three-day AQI forecast
* Historical AQI trends
* Model explanation information
* Health-related information

### Live Application

[Open the Lahore AQI Forecasting Dashboard](https://aqi-predictor-lahore.streamlit.app/)

---

# 🔄 Automation & Deployment

The project is designed as an automated machine learning pipeline.

The overall automation process is:

```text
New Data
   ↓
Data Processing
   ↓
Feature Engineering
   ↓
Feast Feature Store
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Artifacts
   ↓
Streamlit Dashboard
```

GitHub Actions is used to automate recurring tasks.

The system is designed to:

* Collect updated data
* Process new observations
* Update engineered features
* Update the feature store
* Retrain models
* Save updated model artifacts
* Make updated predictions available to the dashboard

This reduces manual intervention and allows the system to continuously work with newer data.

---

# 🛠️ Tools & Technologies

| Technology     | Purpose                                  |
| -------------- | ---------------------------------------- |
| Python         | Main programming language                |
| Pandas         | Data processing and manipulation         |
| NumPy          | Numerical operations                     |
| Scikit-learn   | Machine learning and evaluation          |
| Random Forest  | Main production forecasting model        |
| XGBoost        | Alternative model during model selection |
| Feast          | Feature store                            |
| SHAP           | Model explainability                     |
| Joblib         | Model serialization                      |
| Matplotlib     | Data visualization                       |
| Seaborn        | Exploratory data analysis                |
| Streamlit      | Web dashboard                            |
| GitHub Actions | Pipeline automation                      |
| Open-Meteo     | Air quality and weather data source      |

---

# 📁 Project Structure

A simplified view of the project structure is:

```text
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data collection
│   ├── feature engineering
│   ├── data cleaning
│   └── model training
│
├── feature_repo/
│   └── data/
│
├── models/
│   ├── trained models
│   └── model metadata
│
├── dashboard/
│   └── Streamlit application
│
├── experiments/
│   └── model improvement experiments
│
├── eda_report/
│   └── EDA visualizations
│
└── README.md
```

The exact contents may change as the project evolves.

---

# 🚀 Running the Project

## 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-folder>
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Data Pipeline

Run the project pipeline according to the scripts provided in the repository.

The general sequence is:

```text
Data Collection
      ↓
Feature Engineering
      ↓
Data Cleaning
      ↓
Feast Update
      ↓
Model Training
```

## 5. Run the Dashboard

From the dashboard directory:

```bash
streamlit run app.py
```

The application will then be available locally through the Streamlit URL shown in the terminal.

---

# 📌 Limitations

The main limitation of the current system is the amount of historical data available for training.

AQI forecasting becomes increasingly difficult as the prediction horizon increases. This is reflected in the difference between the Day +1, Day +2, and Day +3 evaluation results.

The model improvement experiments also showed that simply changing algorithms or adding more features does not guarantee better performance.

---

# 🔮 Future Improvements

Potential future improvements include:

* Collecting more historical AQI and weather data
* Increasing winter/smog-season coverage
* Exploring additional environmental variables
* Testing advanced time-series architectures
* Improving uncertainty estimation
* Adding prediction confidence intervals
* Adding more cities
* Adding automated model monitoring
* Tracking model performance over time
* Improving dashboard visualizations

---

# 🎯 Project Goals

The project demonstrates an end-to-end machine learning workflow:

```text
Real-World Data
      ↓
Data Engineering
      ↓
Feature Engineering
      ↓
Feature Store
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Model Explainability
      ↓
Automation
      ↓
Deployment
      ↓
Real-World Predictions
```

The primary goal is not only to train a machine learning model, but to build a complete and maintainable forecasting system that can continuously process new data and provide useful AQI predictions.

---

# 👨‍💻 Internship Project

This project was developed as part of the **10Pearls SHINE Internship Program**.

The project provided practical experience in:

* Data engineering
* Machine learning
* Time-series forecasting
* Feature engineering
* Feature stores
* Model evaluation
* Experimentation
* Model explainability
* MLOps automation
* Dashboard development
* Deployment

---

# ⭐ Acknowledgements

* **10Pearls SHINE Internship Program**
* **Open-Meteo** for air quality and weather data
* Open-source Python machine learning and data science libraries
