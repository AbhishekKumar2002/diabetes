from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from Next.js frontend

# Load the model
with open("diabetes.sav", "rb") as f:
    diabetes_model = pickle.load(f)

@app.route("/")
def home():
    return "Diabetes Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodpressure']),
            float(data['skin']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetespedigree']),
            float(data['age']),
        ]
        prediction = diabetes_model.predict([features])[0]
        return jsonify({"result": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
