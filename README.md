# Weather Prediction ML API 🌦️

## 📌 Overview

This project is an end-to-end Machine Learning weather forecasting API built using Flask and XGBoost.

The system predicts:

- 🌡️ Next-day Maximum Temperature
- 🌡️ Next-day Minimum Temperature
- 🌡️ Next-day Median Temperature
- 🌧️ Next-day Rainfall Probability

using historical weather data of Kolkata.

The project integrates:

- Weather data processing
- Feature engineering
- Machine learning model training
- Model evaluation and visualization
- REST API deployment

into a complete scalable ML pipeline.

---

## ⚙️ Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Render (Deployment)

---

## 🧠 Machine Learning Models

### 🌡️ Temperature Prediction Model

- Model: XGBoost Regressor
- Input: Previous 7 days weather data
- Output:
  - Maximum Temperature
  - Minimum Temperature
  - Median Temperature

---

### 🌧️ Rainfall Prediction Model

- Model: XGBoost Classifier
- Input: Previous 7 days weather data
- Output:
  - Rainfall Probability
  - Rainfall Classification

Classification Labels:

- `Rain Likely`
- `No Significant Rain`

---

## 🔧 Feature Engineering

The models use advanced time-series feature engineering including:

### Lag Features

Previous 7 days values of:

- Temperature
- Humidity
- Pressure
- Rainfall
- Wind speed
- Cloud cover

---

### Rolling Statistics

Rolling window statistics such as:

- Mean
- Standard Deviation
- Rainfall accumulation

---

### Seasonal Features

- Day of Year
- Sin/Cos cyclical encoding for seasonality

---

## 📊 Dataset

### Location

Kolkata, India

### Dataset File

`data/kolkata_daily_data_new.csv`

### Weather Parameters

The dataset contains daily values of:

- Maximum Temperature
- Minimum Temperature
- Median Temperature
- Humidity
- Atmospheric Pressure
- Wind Speed
- Cloud Cover
- Rainfall

---

## 🚀 API Endpoints

### 🌡️ Temperature Prediction

```http
GET /predict_temperature
```

Backward-compatible endpoint:

```http
GET /predict
```

### Example Response

```json
{
  "location": "Kolkata",
  "prediction": {
    "temp_max": 34.87,
    "temp_min": 28.60,
    "temp_median": 31.07
  },
  "prediction_date": "24-05-2026",
  "unit": "Celsius"
}
```

---

### 🌧️ Rainfall Prediction

```http
GET /predict_rainfall
```

### Example Response

```json
{
  "location": "Kolkata",
  "prediction": {
    "rain_probability": 30.97,
    "weather_class": "No Significant Rain"
  },
  "prediction_date": "24-05-2026",
  "unit": "Percent"
}
```

---

## 🛠️ Setup Instructions

Clone the repository:

```bash
git clone https://github.com/RittikBanerjee/weather-ml-api.git
cd weather-ml-api
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set API key:

### Windows (CMD)

```bash
set API_KEY=YOUR_API_KEY
```

### Linux / macOS

```bash
export API_KEY=YOUR_API_KEY
```

Run the Flask server:

```bash
python app.py
```

---

## 🌐 Deployment

Deployed using Render

### Live API

```text
https://weather-ml-api-1.onrender.com/predict_temperature
```

```text
https://weather-ml-api-1.onrender.com/predict_rainfall
```

---

## 📁 Project Structure

```text
weather-ml-api/
│
├── data/
│   └── kolkata_daily_data_new.csv
│
├── models/
│   ├── temp_model.pkl
│   └── rainfall_model.pkl
│
├── notebooks/
│   ├── temp_training.ipynb
│   └── rainfall_training.ipynb
│
├── predictors/
│   ├── __init__.py
│   ├── temp_predict.py
│   └── rainfall_predict.py
│
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
└── .gitignore
```

---

## 📈 Future Improvements

- Real-time weather dashboard
- IoT sensor integration
- Multi-city forecasting
- Advanced rainfall forecasting
- Weather visualization dashboard
- Docker containerization
- CI/CD pipeline
- Historical prediction logging

---

## 👨‍💻 Author

Developed by Rittik Banerjee

GitHub:
https://github.com/RittikBanerjee