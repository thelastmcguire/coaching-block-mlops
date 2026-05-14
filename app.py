import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "model/coaching_block_model.pkl"
model = joblib.load(MODEL_PATH)

strength_trend_map = {
    "Declining": 0,
    "Plateau": 1,
    "Improving": 2
}

comp_in_5_weeks_map = {
    "N": 0,
    "Y": 1
}

@app.route("/")
def home():
    return jsonify({
        "message": "Coaching Block Recommendation API is running"
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_data = pd.DataFrame([{
        "Experience": data["Experience"],
        "Fatigue": data["Fatigue"],
        "StrengthTrend": strength_trend_map[data["StrengthTrend"]],
        "WeeksInBlock": data["WeeksInBlock"],
        "CompIn5Weeks": comp_in_5_weeks_map[data["CompIn5Weeks"]],
        "Stress": data["Stress"]
    }])

    prediction = model.predict(input_data)

    return jsonify({
        "recommended_block": prediction[0]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
