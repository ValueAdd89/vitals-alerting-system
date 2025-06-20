import json
import os
import time
import joblib
import pandas as pd
import requests

model = joblib.load("processor/anomaly_model.pkl")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")

def send_slack_alert(alert):
    if SLACK_WEBHOOK:
        text = f"🚨 *ALERT for {alert['user_id']}* at {pd.to_datetime(alert['timestamp'], unit='s')}:\n - {', '.join(alert['alerts'])}"
        requests.post(SLACK_WEBHOOK, json={"text": text})

def detect_anomaly(vitals):
    features = [[
        vitals["heart_rate"],
        vitals["systolic_bp"],
        vitals["diastolic_bp"],
        vitals["oxygen_saturation"]
    ]]
    is_outlier = model.predict(features)[0] == -1
    return ["Anomaly Detected"] if is_outlier else []

if __name__ == "__main__":
    data_path = "data"
    while True:
        for file in os.listdir(data_path):
            if file.startswith("stream_") and file.endswith(".json"):
                with open(os.path.join(data_path, file)) as f:
                    vitals = json.load(f)
                alerts = detect_anomaly(vitals)
                if alerts:
                    alert = {"user_id": vitals["user_id"], "timestamp": vitals["timestamp"], "alerts": alerts}
                    print("ML ALERT:", alert)
                    send_slack_alert(alert)
        time.sleep(5)