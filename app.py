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

#standard page welcome
@app.route("/")
def home():
    return jsonify({
        "message": "Coaching Block Recommendation API is running. Please enter your details to predict your next block."
    })

##health check used by the deployment workflow
@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })

#model prediction
@app.route("/predict")
def predict():
    experience = int(request.args.get("Experience"))
    fatigue = int(request.args.get("Fatigue"))
    strength_trend = request.args.get("StrengthTrend")
    weeks_in_block = int(request.args.get("WeeksInBlock"))
    comp_in_5_weeks = request.args.get("CompIn5Weeks")
    stress = int(request.args.get("Stress"))

    input_data = pd.DataFrame([{
        "Experience": experience,
        "Fatigue": fatigue,
        "StrengthTrend": strength_trend_map[strength_trend],
        "WeeksInBlock": weeks_in_block,
        "CompIn5Weeks": comp_in_5_weeks_map[comp_in_5_weeks],
        "Stress": stress
    }])

    prediction = model.predict(input_data)[0]

    message = f"Based on the input data, your recommended next coaching block is {prediction}. Best of luck with your training."

    return jsonify({
        "recommended_block": prediction,
        "message": message,
        "input_received": {
            "Experience": experience,
            "Fatigue": fatigue,
            "StrengthTrend": strength_trend,
            "WeeksInBlock": weeks_in_block,
            "CompIn5Weeks": comp_in_5_weeks,
            "Stress": stress
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
