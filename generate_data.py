import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker for realistic data generation
fake = Faker('en_US')

# --- Configuration for Data Generation ---
NUM_PATIENTS = 100 # Number of unique patients to generate
VITAL_READINGS_PER_DAY = 5 # How many vital readings per patient per active day
DAYS_OF_VITALS_HISTORY = 90 # How many days of vital history to generate up to today
DATA_DIR = 'data' # Directory to save CSVs

# Define normal ranges for vitals for realistic generation and anomaly testing
VITAL_PROFILES = {
    'heart_rate': {'mean': 75, 'std': 10, 'min_val': 40, 'max_val': 120, 'anomaly_range': (110, 150)},
    'blood_pressure_systolic': {'mean': 115, 'std': 15, 'min_val': 80, 'max_val': 180, 'anomaly_range': (150, 180)},
    'blood_pressure_diastolic': {'mean': 75, 'std': 10, 'min_val': 50, 'max_val': 120, 'anomaly_range': (95, 120)},
    'temperature': {'mean': 98.6, 'std': 0.7, 'min_val': 95.0, 'max_val': 105.0, 'anomaly_range': (100.5, 105.0)},
    'oxygen_saturation': {'mean': 97, 'std': 2, 'min_val': 85, 'max_val': 100, 'anomaly_range': (88, 93)},
    'respiration_rate': {'mean': 16, 'std': 3, 'min_val': 8, 'max_val': 30, 'anomaly_range': (25, 30)},
    'glucose_level': {'mean': 100, 'std': 20, 'min_val': 50, 'max_val': 250, 'anomaly_range': (180, 250)},
}

# Ensure data directory exists
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# --- 1. Generate Patient Metadata ---
print(f"Generating {NUM_PATIENTS} patient metadata records...")
patient_data = []
diagnosis_options = ["Hypertension", "Diabetes", "Asthma", "Heart Disease", "Pneumonia", "Flu", "Migraine", "Arthritis", "Cancer (Stage I)", "No Specific Condition"]
medication_options = ["Aspirin", "Insulin", "Albuterol", "Lisinopril", "Amoxicillin", "Ibuprofen", "Paracetamol", "Metformin"]
allergy_options = ["Pollen", "Peanuts", "Penicillin", "Dust", "Latex", "None"]
doctor_names = [fake.name() for _ in range(10)]
insurance_providers = ["Blue Cross Blue Shield", "Kaiser Permanente", "Aetna", "UnitedHealthcare", "Cigna", "Medicare", "Medicaid"]
medical_history_options = ["Seasonal Allergies", "Childhood Asthma", "Broken Bone (past)", "No significant history", "Family history of diabetes"]


for i in range(1, NUM_PATIENTS + 1):
    patient_id = f'PAT{i:05d}'
    gender = random.choice(['Male', 'Female', 'Other'])
    age = random.randint(18, 90)
    condition = random.choice(diagnosis_options)
    room_number = f'{random.randint(100, 500)}{random.choice(["A", "B"])}'
    
    admission_date = fake.date_between(start_date='-2y', end_date='today')
    discharge_date = None
    if random.random() < 0.7: # 70% chance of being discharged
        discharge_date = fake.date_between(start_date=admission_date, end_date='today')
        if discharge_date < admission_date: # Ensure discharge is after admission
            discharge_date = admission_date + timedelta(days=random.randint(1, 30))

    medications = random.sample(medication_options, k=random.randint(0, 3))
    allergies = random.sample(allergy_options, k=random.randint(0, 2))
    doctor_assigned = random.choice(doctor_names)
    insurance_provider = random.choice(insurance_providers)
    medical_history = random.sample(medical_history_options, k=random.randint(0, 2))

    patient_data.append({
        'patient_id': patient_id,
        'gender': gender,
        'age': age,
        'condition': condition,
        'room_number': room_number,
        'admission_date': admission_date,
        'discharge_date': discharge_date,
        'medications': ", ".join(medications),
        'allergies': ", ".join(allergies),
        'doctor_assigned': doctor_assigned,
        'insurance_provider': insurance_provider,
        'medical_history': ", ".join(medical_history)
    })

df_patient_metadata = pd.DataFrame(patient_data)
df_patient_metadata.to_csv(Path(DATA_DIR) / 'sample_patient_metadata.csv', index=False)
print("Generated sample_patient_metadata.csv")

# --- 2. Generate Archived Vitals Data ---
print(f"Generating vital signs data for {NUM_PATIENTS} patients over {DAYS_OF_VITALS_HISTORY} days...")
vitals_data = []
today = datetime.now()

for patient in patient_data:
    patient_id = patient['patient_id']
    
    # Define an active period for vitals, typically around admission/discharge dates
    # If discharged, vitals stop around discharge date. If not, continue to today.
    admission_dt = pd.to_datetime(patient['admission_date'])
    discharge_dt = pd.to_datetime(patient['discharge_date']) if patient['discharge_date'] else None

    # Calculate the range of dates for which to generate vitals
    # Start generating vitals from 1 day before admission or max of 90 days ago
    vitals_start_date = max(admission_dt - timedelta(days=1), today - timedelta(days=DAYS_OF_VITALS_HISTORY))
    vitals_end_date = min(discharge_dt + timedelta(days=1) if discharge_dt else today, today) # Vitals end around discharge or today

    current_date = vitals_start_date
    while current_date <= vitals_end_date:
        for _ in range(VITAL_READINGS_PER_DAY):
            timestamp = current_date + timedelta(seconds=random.randint(0, 86399)) # Random time within the day
            
            vitals_reading = {'patient_id': patient_id, 'timestamp': timestamp}
            
            # Introduce occasional anomalies (e.g., 5% chance per reading)
            is_anomaly = random.random() < 0.05

            for vital_name, profile in VITAL_PROFILES.items():
                if is_anomaly and vital_name != 'blood_pressure_diastolic': # Avoid always making BP sys/dias inconsistent
                    # Generate anomaly within a specific range
                    value = random.uniform(profile['anomaly_range'][0], profile['anomaly_range'][1])
                else:
                    # Generate normal value
                    value = random.gauss(profile['mean'], profile['std'])
                
                # Ensure value is within overall min/max
                value = max(profile['min_val'], min(profile['max_val'], value))
                
                vitals_reading[vital_name] = round(value, 1) if vital_name == 'temperature' else int(value) # Round temp, int for others

            vitals_data.append(vitals_reading)
        current_date += timedelta(days=1)

df_vitals = pd.DataFrame(vitals_data)
# Ensure filename matches what's in app.py
df_vitals.to_csv(Path(DATA_DIR) / 'archived_vitals_20250620_212622.csv', index=False) 
print("Generated archived_vitals_20250620_212622.csv")

print("\nData generation complete! Please run your Streamlit app.")
