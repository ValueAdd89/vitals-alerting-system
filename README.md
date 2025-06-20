# 🩺 Real-Time Vitals Alerting System

A real-time health monitoring platform that simulates wearable device vitals, detects anomalies using ML, and routes alerts to a live dashboard and Slack.

## 📦 Features

- Real-time vitals simulator (heart rate, blood pressure, oxygen)
- Anomaly detection with Isolation Forest
- Slack alert routing
- Streamlit dashboard for live monitoring
- Batch ingestion pipeline (Snowflake-ready archive)
- Modular orchestration with Luigi
- GitHub Actions CI setup

## 🛠 Stack

- Python (Streamlit, Scikit-Learn, Slack SDK)
- Luigi (orchestration)
- Snowflake-ready CSV loader
- Streamlit (UI)
- GitHub Actions (CI)

## 📂 Project Structure

```
simulator/                # Vitals generator (heart rate, BP, etc.)
processor/                # ML-based anomaly detection + Slack alerts
dashboard/                # Streamlit UI to view live vitals
orchestration/            # Batch archiving to warehouse
data/                     # Real-time + archived data
.github/workflows/        # CI pipeline
```

## 🔔 Slack Alerts

Set your Slack Webhook in the environment:

```bash
export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'
```

## 🧠 Run ML Detection

Train the model:

```bash
python processor/train_model.py
```

Run alert engine:

```bash
python processor/alert_engine_ml.py
```

## 💾 Archive for Warehouse

```bash
python orchestration/archive_to_snowflake.py
```

## 📊 Dashboard

```bash
streamlit run dashboard/app.py
```