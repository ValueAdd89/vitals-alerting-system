import random
import time
import json

def generate_vitals(user_id):
    return {
        "user_id": user_id,
        "timestamp": time.time(),
        "heart_rate": random.randint(50, 120),
        "systolic_bp": random.randint(90, 160),
        "diastolic_bp": random.randint(60, 100),
        "oxygen_saturation": round(random.uniform(90, 100), 1)
    }

if __name__ == "__main__":
    users = ['user_1', 'user_2', 'user_3']
    while True:
        for user in users:
            vitals = generate_vitals(user)
            with open(f"data/stream_{user}.json", "w") as f:
                json.dump(vitals, f)
        time.sleep(5)