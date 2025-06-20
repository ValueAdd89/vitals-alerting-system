import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv('data/sample_patient_metadata.csv')
users = df['user_id'].tolist()

# Simulate training data
train_data = []
for user in users:
    for _ in range(100):
        train_data.append({
            "user_id": user,
            "heart_rate": 70 + 10 * (user[-1] == '1'),
            "systolic_bp": 120 + 5 * (user[-1] == '2'),
            "diastolic_bp": 80 + 5 * (user[-1] == '3'),
            "oxygen_saturation": 98
        })
train_df = pd.DataFrame(train_data)

X = train_df[["heart_rate", "systolic_bp", "diastolic_bp", "oxygen_saturation"]]
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

joblib.dump(model, "processor/anomaly_model.pkl")