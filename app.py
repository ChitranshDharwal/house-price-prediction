from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load(r"C:\Users\ASUS\Desktop\ml A6 project\decision_tree.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "House Price Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from request
    features = np.array(data['features']).reshape(1, -1)  # Convert to array
    prediction = model.predict(features)[0]  # Predict house price

    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)
