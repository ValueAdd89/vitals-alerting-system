import streamlit as st
import os
import json
import pandas as pd
import time

st.set_page_config(layout="wide")
st.title("🩺 Real-Time Vitals Monitoring")

def load_data():
    vitals_data = []
    for file in os.listdir("data"):
        if file.startswith("stream_") and file.endswith(".json"):
            with open(os.path.join("data", file)) as f:
                data = json.load(f)
                vitals_data.append(data)
    return pd.DataFrame(vitals_data)

if "last_update" not in st.session_state:
    st.session_state.last_update = 0

df = load_data()
if not df.empty:
    st.write("## Live Vitals")
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

    # Simulated alerts
    st.write("## Simulated Alert Rules")
    st.markdown("- HR > 110 or < 50")
    st.markdown("- BP > 140/90")
    st.markdown("- O2 < 94%")
else:
    st.warning("No vitals found. Run the simulator.")