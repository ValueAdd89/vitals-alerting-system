import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="Patient Vitals Monitoring",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants for Vitals Normal Ranges (for display purposes) ---
VITAL_RANGES = {
    'heart_rate': {'min': 60, 'max': 100, 'unit': 'bpm'},
    'blood_pressure_systolic': {'min': 90, 'max': 120, 'unit': 'mmHg'},
    'blood_pressure_diastolic': {'min': 60, 'max': 80, 'unit': 'mmHg'},
    'temperature': {'min': 97.0, 'max': 99.0, 'unit': '°F'}, # Assuming Fahrenheit
    'oxygen_saturation': {'min': 95, 'max': 100, 'unit': '%'},
    'respiration_rate': {'min': 12, 'max': 20, 'unit': 'breaths/min'},
    'glucose_level': {'min': 70, 'max': 140, 'unit': 'mg/dL'}
}

# --- Data Loading (Cached) ---
@st.cache_data
def load_patient_data():
    data_dir = Path(__file__).parent.parent / "data" # Assumes data is in project_root/data/

    metadata_path = data_dir / "sample_patient_metadata.csv"
    vitals_path = data_dir / "archived_vitals_20250620_212622.csv"

    try:
        df_metadata = pd.read_csv(metadata_path)
        df_vitals = pd.read_csv(vitals_path)

        # --- DEBUGGING LINES START HERE ---
        st.write("--- Debugging df_metadata ---")
        st.write("df_metadata head:", df_metadata.head())
        st.write("df_metadata columns:", df_metadata.columns.tolist())
        st.write("df_metadata info:")
        buffer = io.StringIO()
        df_metadata.info(buf=buffer)
        st.text(buffer.getvalue())
        st.write("--- Debugging df_metadata End ---")
        # --- DEBUGGING LINES END HERE ---

        # Convert timestamps to datetime
        df_vitals['timestamp'] = pd.to_datetime(df_vitals['timestamp'])
        
        # Check if 'admission_date' exists before trying to convert
        if 'admission_date' in df_metadata.columns:
            df_metadata['admission_date'] = pd.to_datetime(df_metadata['admission_date'])
        else:
            st.error("Error: 'admission_date' column not found in sample_patient_metadata.csv. Please check the file content.")
            return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames to prevent further errors

        if 'discharge_date' in df_metadata.columns:
            df_metadata['discharge_date'] = pd.to_datetime(df_metadata['discharge_date'])

        # Merge dataframes
        df_merged = pd.merge(df_vitals, df_metadata, on='patient_id', how='left')

        # Basic data quality checks/fill missing values (example)
        for col in ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                    'temperature', 'oxygen_saturation', 'respiration_rate', 'glucose_level']:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
            df_merged = df_merged.dropna(subset=[col])

        return df_merged, df_metadata

    except FileNotFoundError:
        st.error(f"Error: Data CSV files not found at expected path: {metadata_path} or {vitals_path}. Please ensure your data files are correctly placed.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}. This might indicate a problem with column names or data types AFTER file loading. Check the debugging info above for clues.")
        st.stop()

# It is important to import io for the debugging info function
import io
df_all_data, df_patient_metadata = load_patient_data()

if df_all_data.empty:
    st.warning("No data available after loading and initial processing.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Patient Filters")

# Patient ID selection
all_patient_ids = ['All'] + sorted(df_all_data['patient_id'].unique().tolist())
selected_patient_id = st.sidebar.selectbox("Select Patient ID", all_patient_ids)

# Date Range Filter
min_date = df_all_data['timestamp'].min().date()
max_date = df_all_data['timestamp'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) # Include the end day fully

# Demographic Filters
all_genders = ['All'] + sorted(df_patient_metadata['gender'].unique().tolist())
selected_gender = st.sidebar.selectbox("Filter by Gender", all_genders)

all_diagnoses = ['All'] + sorted(df_patient_metadata['diagnosis'].unique().tolist())
selected_diagnosis = st.sidebar.selectbox("Filter by Diagnosis", all_diagnoses)

all_doctors = ['All'] + sorted(df_patient_metadata['doctor_assigned'].unique().tolist())
selected_doctor = st.sidebar.selectbox("Filter by Doctor", all_doctors)


# --- Apply Filters ---
df_filtered_data = df_all_data.copy()

if selected_patient_id != 'All':
    df_filtered_data = df_filtered_data[df_filtered_data['patient_id'] == selected_patient_id]

df_filtered_data = df_filtered_data[
    (df_filtered_data['timestamp'] >= start_date) &
    (df_filtered_data['timestamp'] < end_date)
]

if selected_gender != 'All':
    df_filtered_data = df_filtered_data[df_filtered_data['gender'] == selected_gender]

if selected_diagnosis != 'All':
    df_filtered_data = df_filtered_data[df_filtered_data['diagnosis'] == selected_diagnosis]

if selected_doctor != 'All':
    df_filtered_data = df_filtered_data[df_filtered_data['doctor_assigned'] == selected_doctor]


# --- Check if data exists after filtering ---
if df_filtered_data.empty:
    st.warning("No data available for the selected filters. Please adjust your selections.")
    st.stop()

# --- Dashboard Content ---
st.title("🏥 Patient Vitals Monitoring Dashboard")
st.markdown("Monitor patient vital signs and demographic trends.")

tabs = st.tabs([
    "📊 Overall Summary",
    "📈 Vitals Trends",
    "📉 Vitals Distribution",
    "🚨 Anomaly Detection"
])

# --- TAB 1: Overall Summary ---
with tabs[0]:
    st.header("Overall Patient Summary")

    total_patients_filtered = df_filtered_data['patient_id'].nunique()
    total_readings = len(df_filtered_data)
    avg_hr = df_filtered_data['heart_rate'].mean()
    avg_temp = df_filtered_data['temperature'].mean()
    avg_o2 = df_filtered_data['oxygen_saturation'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Patients (Filtered)", total_patients_filtered)
    col2.metric("Total Vital Readings", total_readings)
    col3.metric("Avg Heart Rate", f"{avg_hr:.1f} bpm")
    col4.metric("Avg Temperature", f"{avg_temp:.1f} °F")
    col5.metric("Avg O2 Saturation", f"{avg_o2:.1f} %")

    st.markdown("---")
    st.subheader("Patient Demographics (Filtered Population)")

    col_demog1, col_demog2 = st.columns(2)

    with col_demog1:
        # Diagnosis Distribution
        diagnosis_counts = df_filtered_data['diagnosis'].value_counts().reset_index()
        diagnosis_counts.columns = ['Diagnosis', 'Count']
        fig_diagnosis = px.pie(
            diagnosis_counts,
            values='Count',
            names='Diagnosis',
            title='Diagnosis Distribution',
            hole=0.3
        )
        st.plotly_chart(fig_diagnosis, use_container_width=True)

    with col_demog2:
        # Age Distribution (using age groups if available, or just histogram)
        fig_age = px.histogram(
            df_filtered_data,
            x='age',
            nbins=10,
            title='Age Distribution',
            labels={'age': 'Age (Years)', 'count': 'Number of Patients'}
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Patient Metadata Details (Filtered Population)")
    # Show metadata for unique patients in filtered data
    unique_patients_metadata = df_filtered_data[['patient_id', 'gender', 'age', 'diagnosis', 'doctor_assigned', 'room_number', 'admission_date']].drop_duplicates().set_index('patient_id')
    st.dataframe(unique_patients_metadata)


# --- TAB 2: Vitals Trends ---
with tabs[1]:
    st.header("Vital Signs Trends")

    if selected_patient_id == 'All':
        st.warning("Please select a specific patient from the sidebar to view individual vital sign trends.")
    else:
        st.info(f"Showing vital signs for Patient ID: {selected_patient_id} from {date_range[0]} to {date_range[1]}.")
        patient_data = df_filtered_data[df_filtered_data['patient_id'] == selected_patient_id].sort_values('timestamp')

        if patient_data.empty:
            st.warning("No vital sign readings for this patient within the selected date range and filters.")
        else:
            # Heart Rate
            fig_hr = px.line(patient_data, x='timestamp', y='heart_rate', title='Heart Rate Trend')
            if 'heart_rate' in VITAL_RANGES:
                fig_hr.add_hrect(y0=VITAL_RANGES['heart_rate']['min'], y1=VITAL_RANGES['heart_rate']['max'], line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal Range", annotation_position="top left")
            st.plotly_chart(fig_hr, use_container_width=True)

            # Blood Pressure
            fig_bp = go.Figure()
            fig_bp.add_trace(go.Scatter(x=patient_data['timestamp'], y=patient_data['blood_pressure_systolic'], mode='lines+markers', name='Systolic'))
            fig_bp.add_trace(go.Scatter(x=patient_data['timestamp'], y=patient_data['blood_pressure_diastolic'], mode='lines+markers', name='Diastolic'))
            fig_bp.update_layout(title='Blood Pressure Trend', yaxis_title='BP (mmHg)')
            if 'blood_pressure_systolic' in VITAL_RANGES:
                fig_bp.add_hrect(y0=VITAL_RANGES['blood_pressure_systolic']['min'], y1=VITAL_RANGES['blood_pressure_systolic']['max'], line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal Systolic", annotation_position="top left")
            if 'blood_pressure_diastolic' in VITAL_RANGES:
                fig_bp.add_hrect(y0=VITAL_RANGES['blood_pressure_diastolic']['min'], y1=VITAL_RANGES['blood_pressure_diastolic']['max'], line_width=0, fillcolor="blue", opacity=0.1, annotation_text="Normal Diastolic", annotation_position="top right")
            st.plotly_chart(fig_bp, use_container_width=True)
            
            # Temperature
            fig_temp = px.line(patient_data, x='timestamp', y='temperature', title='Temperature Trend')
            if 'temperature' in VITAL_RANGES:
                fig_temp.add_hrect(y0=VITAL_RANGES['temperature']['min'], y1=VITAL_RANGES['temperature']['max'], line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal Range", annotation_position="top left")
            st.plotly_chart(fig_temp, use_container_width=True)

            # Oxygen Saturation & Respiration Rate (combined for space)
            col_vitals_extra1, col_vitals_extra2 = st.columns(2)
            with col_vitals_extra1:
                fig_o2 = px.line(patient_data, x='timestamp', y='oxygen_saturation', title='Oxygen Saturation Trend')
                if 'oxygen_saturation' in VITAL_RANGES:
                    fig_o2.add_hrect(y0=VITAL_RANGES['oxygen_saturation']['min'], y1=VITAL_RANGES['oxygen_saturation']['max'], line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal Range", annotation_position="top left")
                st.plotly_chart(fig_o2, use_container_width=True)
            with col_vitals_extra2:
                fig_resp = px.line(patient_data, x='timestamp', y='respiration_rate', title='Respiration Rate Trend')
                if 'respiration_rate' in VITAL_RANGES:
                    fig_resp.add_hrect(y0=VITAL_RANGES['respiration_rate']['min'], y1=VITAL_RANGES['respiration_rate']['max'], line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal Range", annotation_position="top left")
                st.plotly_chart(fig_resp, use_container_width=True)

            # Glucose Level
            fig_glucose = px.line(patient_data, x='timestamp', y='glucose_level', title='Glucose Level Trend')
            if 'glucose_level' in VITAL_RANGES:
                fig_glucose.add_hrect(y0=VITAL_RANGES['glucose_level']['min'], y1=VITAL_RANGES['glucose_level']['max'], line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal Range", annotation_position="top left")
            st.plotly_chart(fig_glucose, use_container_width=True)


# --- TAB 3: Vitals Distribution ---
with tabs[2]:
    st.header("Vital Signs Distribution (Filtered Population)")

    vital_signs_cols = [
        'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
        'temperature', 'oxygen_saturation', 'respiration_rate', 'glucose_level'
    ]
    
    # Ensure only available vital signs are in the list
    available_vitals = [col for col in vital_signs_cols if col in df_filtered_data.columns]

    if not available_vitals:
        st.warning("No vital sign columns found in the filtered data to display distribution.")
    else:
        selected_vital_dist = st.selectbox("Select Vital Sign to View Distribution", available_vitals)

        if selected_vital_dist:
            fig_dist = px.histogram(
                df_filtered_data,
                x=selected_vital_dist,
                nbins=20,
                title=f'Distribution of {selected_vital_dist.replace("_", " ").title()}',
                labels={selected_vital_dist: f'{selected_vital_dist.replace("_", " ").title()} ({VITAL_RANGES.get(selected_vital_dist, {}).get("unit", "")})', 'count': 'Frequency'}
            )
            if selected_vital_dist in VITAL_RANGES:
                fig_dist.add_vrect(
                    x0=VITAL_RANGES[selected_vital_dist]['min'], x1=VITAL_RANGES[selected_vital_dist]['max'],
                    fillcolor="green", opacity=0.1, line_width=0,
                    annotation_text="Normal Range", annotation_position="top left"
                )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("---")
            st.subheader(f"Summary Statistics for {selected_vital_dist.replace('_', ' ').title()}")
            st.dataframe(df_filtered_data[selected_vital_dist].describe().to_frame().T)
        else:
            st.info("No vital sign selected for distribution.")

# --- TAB 4: Anomaly Detection ---
with tabs[3]:
    st.header("Potential Vital Sign Anomalies")
    st.markdown("Highlights recent vital sign readings that fall outside typical 'normal' ranges.")

    if df_filtered_data.empty:
        st.info("No data available to check for anomalies.")
    else:
        anomalies_found = []
        for vital, ranges in VITAL_RANGES.items():
            if vital in df_filtered_data.columns:
                out_of_range_low = df_filtered_data[df_filtered_data[vital] < ranges['min']]
                out_of_range_high = df_filtered_data[df_filtered_data[vital] > ranges['max']]

                if not out_of_range_low.empty:
                    for _, row in out_of_range_low.iterrows():
                        anomalies_found.append({
                            'Patient ID': row['patient_id'],
                            'Timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M'),
                            'Vital Sign': vital.replace('_', ' ').title(),
                            'Reading': f"{row[vital]:.1f} {ranges['unit']}",
                            'Normal Range': f"{ranges['min']}-{ranges['max']} {ranges['unit']}",
                            'Deviation': 'Below Normal'
                        })
                if not out_of_range_high.empty:
                    for _, row in out_of_range_high.iterrows():
                        anomalies_found.append({
                            'Patient ID': row['patient_id'],
                            'Timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M'),
                            'Vital Sign': vital.replace('_', ' ').title(),
                            'Reading': f"{row[vital]:.1f} {ranges['unit']}",
                            'Normal Range': f"{ranges['min']}-{ranges['max']} {ranges['unit']}",
                            'Deviation': 'Above Normal'
                        })
        
        if anomalies_found:
            df_anomalies = pd.DataFrame(anomalies_found).sort_values(by=['Timestamp', 'Patient ID'], ascending=False)
            st.warning(f"Found {len(df_anomalies)} potential vital sign anomalies!")
            st.dataframe(df_anomalies, use_container_width=True)
            
            # Allow user to select a vital sign to see anomaly trend
            st.subheader("Anomaly Trends by Vital Sign")
            anomaly_vitals = sorted(df_anomalies['Vital Sign'].unique().tolist())
            selected_anomaly_vital = st.selectbox("Select Vital Sign to see its anomaly trend", anomaly_vitals)

            if selected_anomaly_vital:
                vital_col_name = selected_anomaly_vital.replace(' ', '_').lower() # Convert back to column name
                if vital_col_name in df_filtered_data.columns:
                    fig_anomaly_trend = px.line(
                        df_filtered_data[df_filtered_data['patient_id'].isin(df_anomalies[df_anomalies['Vital Sign'] == selected_anomaly_vital]['Patient ID'])],
                        x='timestamp',
                        y=vital_col_name,
                        color='patient_id',
                        title=f'Trend for {selected_anomaly_vital} with Anomalies Highlighted'
                    )
                    # Add normal range to this trend chart as well
                    if vital_col_name in VITAL_RANGES:
                         fig_anomaly_trend.add_hrect(y0=VITAL_RANGES[vital_col_name]['min'], y1=VITAL_RANGES[vital_col_name]['max'], line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal Range", annotation_position="top left")
                    st.plotly_chart(fig_anomaly_trend, use_container_width=True)
                else:
                    st.info(f"Cannot plot trend for {selected_anomaly_vital} as column is not directly available.")
        else:
            st.success("No vital sign anomalies detected within the selected filters and defined normal ranges.")
