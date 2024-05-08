from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle

app = Flask(__name__)

# Load the ARIMA model
with open('arima_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the function to make predictions
def predict_market_value(steps):
    predictions = model.forecast(steps=steps)
    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    steps = data['steps']
    predictions = predict_market_value(steps)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
