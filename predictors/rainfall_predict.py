import requests
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import os

# ----------------------
# SETTINGS
# ----------------------
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

LOCATION = "Kolkata"
UNIT_GROUP = "metric"
INCLUDE = "days"

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_FILE = os.getenv(
    "RAINFALL_MODEL_FILE",
    "rainfall_model.pkl"
)

MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_FILE)

model = joblib.load(MODEL_PATH)

# ----------------------
# FEATURE COLUMNS
# ----------------------
feature_cols = [
    'temp_max',
    'temp_min',
    'temp_median',
    'pressure_mean',
    'humidity_mean',
    'wind_mean',
    'cloud_mean',
    'day_of_year',
    'day_sin',
    'day_cos',

    'temp_max_lag1','temp_max_lag2','temp_max_lag3','temp_max_lag4','temp_max_lag5','temp_max_lag6','temp_max_lag7',

    'temp_min_lag1','temp_min_lag2','temp_min_lag3','temp_min_lag4','temp_min_lag5','temp_min_lag6','temp_min_lag7',

    'temp_median_lag1','temp_median_lag2','temp_median_lag3','temp_median_lag4','temp_median_lag5','temp_median_lag6','temp_median_lag7',

    'humidity_mean_lag1','humidity_mean_lag2','humidity_mean_lag3','humidity_mean_lag4','humidity_mean_lag5','humidity_mean_lag6','humidity_mean_lag7',

    'pressure_mean_lag1','pressure_mean_lag2','pressure_mean_lag3','pressure_mean_lag4','pressure_mean_lag5','pressure_mean_lag6','pressure_mean_lag7',

    'rain_total_lag1','rain_total_lag2','rain_total_lag3','rain_total_lag4','rain_total_lag5','rain_total_lag6','rain_total_lag7',

    'wind_mean_lag1','wind_mean_lag2','wind_mean_lag3','wind_mean_lag4','wind_mean_lag5','wind_mean_lag6','wind_mean_lag7',

    'cloud_mean_lag1','cloud_mean_lag2','cloud_mean_lag3','cloud_mean_lag4','cloud_mean_lag5','cloud_mean_lag6','cloud_mean_lag7',

    'temp_max_roll7_mean',
    'temp_max_roll7_std',

    'temp_min_roll7_mean',
    'temp_min_roll7_std',

    'temp_median_roll7_mean',
    'temp_median_roll7_std',

    'humidity_roll7_mean',
    'pressure_roll7_mean',

    'rain_roll3_sum',
    'rain_roll7_sum',

    'cloud_roll7_mean'
]

# ----------------------
# FETCH WEATHER DATA
# ----------------------
def fetch_weather():

    url = (
        f"https://weather.visualcrossing.com/"
        f"VisualCrossingWebServices/rest/services/"
        f"timeline/{LOCATION}/last8days?"
    )

    url += (
        f"key={API_KEY}"
        f"&unitGroup={UNIT_GROUP}"
        f"&contentType=json"
        f"&include={INCLUDE}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    daily_list = []

    for day in data['days']:

        daily_list.append({
            'date': day['datetime'],
            'temp_max': day['tempmax'],
            'temp_min': day['tempmin'],
            'temp_median': day['temp'],
            'humidity_mean': day['humidity'],
            'pressure_mean': day['pressure'],
            'wind_mean': day['windspeed'],
            'cloud_mean': day['cloudcover'],
            'rain_total': day['precip']
        })

    df = pd.DataFrame(daily_list)

    df['date'] = pd.to_datetime(df['date'])

    return df


# ----------------------
# FEATURE ENGINEERING
# ----------------------
def create_features(df):

    df['day_of_year'] = df['date'].dt.dayofyear

    df['day_sin'] = np.sin(
        2 * np.pi * df['day_of_year'] / 365
    )

    df['day_cos'] = np.cos(
        2 * np.pi * df['day_of_year'] / 365
    )

    lag_cols = [
        'temp_max',
        'temp_min',
        'temp_median',
        'humidity_mean',
        'pressure_mean',
        'rain_total',
        'wind_mean',
        'cloud_mean'
    ]

    for col in lag_cols:
        for lag in range(1, 8):

            df[f'{col}_lag{lag}'] = (
                df[col].shift(lag)
            )

    # Rolling statistics
    df['temp_max_roll7_mean'] = (
        df['temp_max'].rolling(7).mean()
    )

    df['temp_max_roll7_std'] = (
        df['temp_max'].rolling(7).std()
    )

    df['temp_min_roll7_mean'] = (
        df['temp_min'].rolling(7).mean()
    )

    df['temp_min_roll7_std'] = (
        df['temp_min'].rolling(7).std()
    )

    df['temp_median_roll7_mean'] = (
        df['temp_median'].rolling(7).mean()
    )

    df['temp_median_roll7_std'] = (
        df['temp_median'].rolling(7).std()
    )

    df['humidity_roll7_mean'] = (
        df['humidity_mean'].rolling(7).mean()
    )

    df['pressure_roll7_mean'] = (
        df['pressure_mean'].rolling(7).mean()
    )

    df['rain_roll3_sum'] = (
        df['rain_total'].rolling(3).sum()
    )

    df['rain_roll7_sum'] = (
        df['rain_total'].rolling(7).sum()
    )

    df['cloud_roll7_mean'] = (
        df['cloud_mean'].rolling(7).mean()
    )

    df = df.dropna().reset_index(drop=True)

    return df


# ----------------------
# RAINFALL PREDICTION
# ----------------------
def predict_rainfall():

    df = fetch_weather()

    if df.empty:
        raise ValueError(
            "Not enough data for prediction"
        )

    df = create_features(df)

    X_input = df[feature_cols].iloc[-1:].values

    rain_probability = (
        model.predict_proba(X_input)[0][1]
    )

    rain_probability_percent = (
        rain_probability * 100
    )

    threshold = 0.36

    if rain_probability >= threshold:
        weather_class = "Rain Likely"
    else:
        weather_class = "No Significant Rain"

    last_date = df['date'].iloc[-1]

    prediction_date = (
        last_date + timedelta(days=1)
    )

    return {
        "prediction_date": prediction_date.strftime("%d-%m-%Y"),
        "prediction": {
            "rain_probability": round(
                float(rain_probability_percent),
                2
            ),
            "weather_class": weather_class
        },
        "unit": "Percent",
        "location": "Kolkata"
    }