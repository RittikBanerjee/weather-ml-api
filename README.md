# Weather Prediction ML API 🌦️

## 📌 Overview

This project is an end-to-end Machine Learning system that predicts:

* Maximum Temperature
* Minimum Temperature
* Median Temperature

using historical weather data of Kolkata.

It integrates **data collection, feature engineering, model training, and deployment** into a single pipeline.

---

## ⚙️ Tech Stack

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* XGBoost

---

## 🧠 Model Details

* Model: XGBoost Regressor
* Input: Past 7 days weather data
* Output: Next day temperature predictions

---

## 🔧 Feature Engineering

The model uses:

* Lag features (last 7 days)
* Rolling statistics (mean, standard deviation)
* Time-based features:

  * Day of year
  * Sin/Cos transformation (seasonality)

---

## 📊 Dataset

* Location: Kolkata
* File: `data/kolkata_daily_data_new.csv`
* Contains daily weather parameters like:

  * Temperature
  * Humidity
  * Pressure
  * Wind speed
  * Cloud cover
  * Rainfall

---

## 🚀 API Endpoint

### GET /predict

Returns:

```json
{
  "temp_max": float,
  "temp_min": float,
  "temp_median": float
}
```

---

## 🛠️ Setup Instructions

```bash
git clone https://github.com/RittikBanerjee/weather-ml-api.git
cd weather-ml-api
pip install -r requirements.txt
python app.py
```

---

## 🌐 Deployment

Deployed using Render: https://weather-ml-api-1.onrender.com/predict

---

## 📁 Project Structure

```
weather-ml-api/
│
├── app.py
├── temp_model.pkl
├── requirements.txt
├── Procfile
│
├── data/
│   └── kolkata_daily_data_new.csv
│
├── notebooks/
│   └── model_training.ipynb
```
