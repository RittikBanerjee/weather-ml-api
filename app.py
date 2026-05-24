from flask import Flask, jsonify
import os

from predictors.temp_predict import predict_temperature
from predictors.rainfall_predict import predict_rainfall

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({
        "message": "Weather ML API Running"
    })


# Old endpoint for temperature prediction, kept for backward compatibility
@app.route("/predict", methods=["GET"])

# New scalable endpoint for temperature prediction
@app.route("/predict_temperature", methods=["GET"])
def temperature_prediction():

    try:
        result = predict_temperature()
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# New scalable endpoint for rainfall prediction
@app.route("/predict_rainfall", methods=["GET"])
def rainfall_prediction():

    try:
        result = predict_rainfall()
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
    
@app.route("/healthz")
def health_check():
    return "OK", 200


debug_mode = os.getenv("DEBUG", "False") == "True"

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=debug_mode
    )
