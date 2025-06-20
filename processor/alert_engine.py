import json
import os
import time

def detect_anomalies(v):
    alerts = []
    if v['heart_rate'] > 110 or v['heart_rate'] < 50:
        alerts.append('Heart Rate Abnormal')
    if v['systolic_bp'] > 140 or v['diastolic_bp'] > 90:
        alerts.append('High Blood Pressure')
    if v['oxygen_saturation'] < 94:
        alerts.append('Low Oxygen Saturation')
    return alerts

if __name__ == "__main__":
    data_path = "data"
    alert_log = []

    while True:
        for file in os.listdir(data_path):
            if file.startswith("stream_") and file.endswith(".json"):
                with open(os.path.join(data_path, file)) as f:
                    vitals = json.load(f)
                alerts = detect_anomalies(vitals)
                if alerts:
                    alert_event = {"user_id": vitals['user_id'], "timestamp": vitals['timestamp'], "alerts": alerts}
                    print("ALERT:", alert_event)
                    alert_log.append(alert_event)
        time.sleep(5)