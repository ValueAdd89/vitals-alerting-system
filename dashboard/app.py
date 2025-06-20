import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import io # Needed for debugging output buffer, will remove later

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

        # --- FIX: Rename user_id to patient_id in metadata for merging ---
        if 'user_id' in df_metadata.columns:
            df_metadata = df_metadata.rename(columns={'user_id': 'patient_id'})
        else:
            st.error("Error: 'user_id' column not found in sample_patient_metadata.csv. Cannot merge without a common ID.")
            return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames

        # --- FIX: Remove 'admission_date'/'discharge_date' processing as they are not present ---
        # df_metadata['admission_date'] = pd.to_datetime(df_metadata['admission_date']) # Removed
        # if 'discharge_date' in df_metadata.columns: # Removed
        #     df_metadata['discharge_date'] = pd.to_datetime(df_metadata['discharge_date']) # Removed

        # Convert vitals timestamp
        df_vitals['timestamp'] = pd.to_datetime(df_vitals['timestamp'])

        # Merge dataframes
        df_merged = pd.merge(df_vitals, df_metadata, on='patient_id', how='left')

        # Basic data quality checks/fill missing values (example)
        for col in ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                    'temperature', 'oxygen_saturation', 'respiration_rate', 'glucose_level']:
            if col in df_merged.columns: # Ensure column exists before processing
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce') # Coerce non-numeric to NaN
                df_merged = df_merged.dropna(subset=[col]) # Drop rows where vitals are NaN
            else:
                st.warning(f"Vital sign column '{col}' not found in vital signs data. Some dashboard elements may be incomplete.")

        return df_merged, df_metadata

    except FileNotFoundError:
        st.error(f"Error: Data CSV files not found at expected path: {metadata_path} or {vitals_path}. Please ensure your data files are correctly placed.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}. Please check your CSV file contents and column names.")
        st.stop()

# --- Load Data ---
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
# Ensure min/max date are valid before using them
if not df_all_data['timestamp'].empty:
    min_date_val = df_all_data['timestamp'].min().date()
    max_date_val = df_all_data['timestamp'].max().date()
else: # Fallback for empty data if it somehow reaches here
    min_date_val = pd.to_datetime('2023-01-01').date()
    max_date_val = pd.to_datetime('2023-01-01').date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date_val, max_date_val),
    min_value=min_date_val,
    max_value=max_date_val
)

# Ensure date_range has two elements before unpacking
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) # Include the end day fully
else: # Handle case where only one date is selected (e.g., initial state)
    start_date = pd.to_datetime(min_date_val)
    end_date = pd.to_datetime(max_date_val) + pd.Timedelta(days=1)


# Demographic Filters
all_genders = ['All'] + sorted(df_patient_metadata['gender'].unique().tolist())
selected_gender = st.sidebar.selectbox("Filter by Gender", all_genders)

# --- FIX: Changed Diagnosis to Condition ---
if 'condition' in df_patient_metadata.columns:
    all_conditions = ['All'] + sorted(df_patient_metadata['condition'].unique().tolist())
    selected_condition = st.sidebar.selectbox("Filter by Condition", all_conditions)
else:
    selected_condition = 'All'
    st.sidebar.info("Condition column not found in metadata for filtering.")

# --- FIX: Removed Doctor filter as 'doctor_assigned' is not present ---
# all_doctors = ['All'] + sorted(df_patient_metadata['doctor_assigned'].unique().tolist()) # Removed
# selected_doctor = st.sidebar.selectbox("Filter by Doctor", all_doctors) # Removed


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

if selected_condition != 'All': # --- FIX: Apply Condition filter ---
    df_filtered_data = df_filtered_data[df_filtered_data['condition'] == selected_condition]

# --- FIX: Removed Doctor filter application ---
# if selected_doctor != 'All':
#     df_filtered_data = df_filtered_data[df_filtered_data['doctor_assigned'] == selected_doctor]


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
    
    # Ensure vital sign columns exist before calculating mean
    avg_hr = df_filtered_data['heart_rate'].mean() if 'heart_rate' in df_filtered_data.columns else np.nan
    avg_temp = df_filtered_data['temperature'].mean() if 'temperature' in df_filtered_data.columns else np.nan
    avg_o2 = df_filtered_data['oxygen_saturation'].mean() if 'oxygen_saturation' in df_filtered_data.columns else np.nan

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Patients (Filtered)", total_patients_filtered)
    col2.metric("Total Vital Readings", total_readings)
    col3.metric("Avg Heart Rate", f"{avg_hr:.1f} bpm" if not np.isnan(avg_hr) else "N/A")
    col4.metric("Avg Temperature", f"{avg_temp:.1f} °F" if not np.isnan(avg_temp) else "N/A")
    col5.metric("Avg O2 Saturation", f"{avg_o2:.1f} %" if not np.isnan(avg_o2) else "N/A")

    st.markdown("---")
    st.subheader("Patient Demographics (Filtered Population)")

    col_demog1, col_demog2 = st.columns(2)

    with col_demog1:
        # Condition Distribution (formerly Diagnosis)
        if 'condition' in df_filtered_data.columns and not df_filtered_data['condition'].empty:
            condition_counts = df_filtered_data['condition'].value_counts().reset_index()
            condition_counts.columns = ['Condition', 'Count']
            fig_condition = px.pie(
                condition_counts,
                values='Count',
                names='Condition',
                title='Condition Distribution',
                hole=0.3
            )
            st.plotly_chart(fig_condition, use_container_width=True)
        else:
            st.info("No condition data available for distribution.")


    with col_demog2:
        # Age Distribution
        if 'age' in df_filtered_data.columns and not df_filtered_data['age'].empty:
            fig_age = px.histogram(
                df_filtered_data,
                x='age',
                nbins=10,
                title='Age Distribution',
                labels={'age': 'Age (Years)', 'count': 'Number of Patients'}
            )
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("No age data available for distribution.")
    
    st.markdown("---")
    st.subheader("Patient Metadata Details (Filtered Population)")
    # Show metadata for unique patients in filtered data
    # --- FIX: Display only available columns ---
    display_cols = [col for col in ['patient_id', 'gender', 'age', 'condition'] if col in df_filtered_data.columns]
    if display_cols:
        unique_patients_metadata = df_filtered_data[display_cols].drop_duplicates().set_index('patient_id')
        st.dataframe(unique_patients_metadata)
    else:
        st.info("No relevant metadata columns available for display.")


# --- TAB 2: Vitals Trends ---
with tabs[1]:
    st.header("Vital Signs Trends")

    if selected_patient_id == 'All':
        st.warning("Please select a specific patient from the sidebar to view individual vital sign trends.")
    else:
        st.info(f"Showing vital signs for Patient ID: {selected_patient_id} from {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}.")
        patient_data = df_filtered_data[df_filtered_data['patient_id'] == selected_patient_id].sort_values('timestamp')

        if patient_data.empty:
            st.warning("No vital sign readings for this patient within the selected date range and filters.")
        else:
            # Common function to plot vital trends
            def plot_vital_trend(data, vital_col, title, unit):
                if vital_col in data.columns:
                    fig = px.line(data, x='timestamp', y=vital_col, title=title)
                    if vital_col in VITAL_RANGES:
                        fig.add_hrect(y0=VITAL_RANGES[vital_col]['min'], y1=VITAL_RANGES[vital_col]['max'], line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal Range", annotation_position="top left")
                    fig.update_layout(yaxis_title=f"{vital_col.replace('_', ' ').title()} ({unit})")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"'{title}' data not available for plotting.")

            plot_vital_trend(patient_data, 'heart_rate', 'Heart Rate Trend', VITAL_RANGES['heart_rate']['unit'])
            
            # Blood Pressure
            if 'blood_pressure_systolic' in patient_data.columns and 'blood_pressure_diastolic' in patient_data.columns:
                fig_bp = go.Figure()
                fig_bp.add_trace(go.Scatter(x=patient_data['timestamp'], y=patient_data['blood_pressure_systolic'], mode='lines+markers', name='Systolic'))
                fig_bp.add_trace(go.Scatter(x=patient_data['timestamp'], y=patient_data['blood_pressure_diastolic'], mode='lines+markers', name='Diastolic'))
                fig_bp.update_layout(title='Blood Pressure Trend', yaxis_title='BP (mmHg)')
                if 'blood_pressure_systolic' in VITAL_RANGES:
                    fig_bp.add_hrect(y0=VITAL_RANGES['blood_pressure_systolic']['min'], y1=VITAL_RANGES['blood_pressure_systolic']['max'], line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal Systolic", annotation_position="top left")
                if 'blood_pressure_diastolic' in VITAL_RANGES:
                    fig_bp.add_hrect(y0=VITAL_RANGES['blood_pressure_diastolic']['min'], y1=VITAL_RANGES['blood_pressure_diastolic']['max'], line_width=0, fillcolor="blue", opacity=0.1, annotation_text="Normal Diastolic", annotation_position="top right")
                st.plotly_chart(fig_bp, use_container_width=True)
            else:
                st.info("Blood Pressure data (systolic/diastolic) not available for plotting.")
            
            plot_vital_trend(patient_data, 'temperature', 'Temperature Trend', VITAL_RANGES['temperature']['unit'])

            col_vitals_extra1, col_vitals_extra2 = st.columns(2)
            with col_vitals_extra1:
                plot_vital_trend(patient_data, 'oxygen_saturation', 'Oxygen Saturation Trend', VITAL_RANGES['oxygen_saturation']['unit'])
            with col_vitals_extra2:
                plot_vital_trend(patient_data, 'respiration_rate', 'Respiration Rate Trend', VITAL_RANGES['respiration_rate']['unit'])

            plot_vital_trend(patient_data, 'glucose_level', 'Glucose Level Trend', VITAL_RANGES['glucose_level']['unit'])


# --- TAB 3: Vitals Distribution ---
with tabs[2]:
    st.header("Vital Signs Distribution (Filtered Population)")

    vital_signs_cols = [
        'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
        'temperature', 'oxygen_saturation', 'respiration_rate', 'glucose_level'
    ]
    
    available_vitals = [col for col in vital_signs_cols if col in df_filtered_data.columns and not df_filtered_data[col].empty]

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
            if vital in df_filtered_data.columns: # Ensure vital column exists
                # Filter out NaN values before comparison
                vital_data_filtered = df_filtered_data[df_filtered_data[vital].notna()]
                
                out_of_range_low = vital_data_filtered[vital_data_filtered[vital] < ranges['min']]
                out_of_range_high = vital_data_filtered[vital_data_filtered[vital] > ranges['max']]

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
                    # Filter patient data to only show patients with anomalies for the selected vital
                    patients_with_anomaly = df_anomalies[df_anomalies['Vital Sign'] == selected_anomaly_vital]['Patient ID'].unique()
                    data_for_anomaly_trend = df_filtered_data[df_filtered_data['patient_id'].isin(patients_with_anomaly)].sort_values('timestamp')
                    
                    fig_anomaly_trend = px.line(
                        data_for_anomaly_trend,
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
                    st.info(f"Cannot plot trend for {selected_anomaly_vital} as column is not directly available in data.")
        else:
            st.success("No vital sign anomalies detected within the selected filters and defined normal ranges.")
