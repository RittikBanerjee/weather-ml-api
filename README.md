# Weather ML API 🌦️

A Flask-based machine learning API that predicts:
- Maximum temperature
- Minimum temperature
- Median temperature

## Tech Stack
- Flask
- Scikit-learn
- Pandas
- NumPy

## Setup

1. Clone the repository:
   git clone https://github.com/RittikBanerjee/weather-ml-api.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run:
   python app.py

## Endpoint

GET /predict

Returns:
{
  "temp_max": float,
  "temp_min": float,
  "temp_median": float
}
