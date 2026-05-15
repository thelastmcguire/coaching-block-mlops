import os
import json
import joblib
import pandas as pd
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "data/coaching_block_type_synthetic.csv"
MODEL_PATH = "model/coaching_block_model.pkl"
METRICS_PATH = "model/metrics.json"

os.makedirs("model", exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)

    df["CompIn5Weeks"] = df["CompIn5Weeks"].map({"Y": 1, "N": 0})
    df["StrengthTrend"] = df["StrengthTrend"].map({
        "Declining": 0,
        "Plateau": 1,
        "Improving": 2
    })

    X = df[[
        "Experience",
        "Fatigue",
        "StrengthTrend",
        "WeeksInBlock",
        "CompIn5Weeks",
        "Stress"
    ]]

    y = df["Block"]

    return X, y

def train_model():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "accuracy": accuracy,
        "this model was trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Model trained successfully. Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    train_model()
