# Cloud-Based Health Monitoring System

Real-time healthcare prediction system using Google Cloud Platform for monitoring patient vitals and predicting health risks.

## Overview

This system generates synthetic patient data, processes sensor readings, and uses machine learning models to predict various health risks including:
- Hypoglycemia
- Fall risk
- Cardiac anomalies
- Severe hypotension
- Autonomic dysregulation

## Architecture

- **Data Generator**: Cloud Run service that generates synthetic patient sensor data
- **Sensor Aggregator**: Cloud Function that aggregates 5-minute windows of sensor data
- **Health Predictor**: Cloud Run service that runs ML predictions on aggregated data
- **Predictions Writer**: Cloud Function that writes predictions to BigQuery
- **Vitals Writer**: Cloud Function that writes vitals and predictions to BigQuery for dashboards[https://raghuveer9303.github.io/mediguard-dashboard/#/]

## Components

### Files

- `data-generator.py` - Generates synthetic patient data and publishes to Pub/Sub
- `sensor-aggregator.py` - Aggregates sensor data into 5-minute windows
- `health-predictor.py` - ML predictor module for loading models and making predictions
- `cloud_run_for_health_predictor.py` - Flask app for prediction service
- `predictions-writer.py` - Writes prediction results to BigQuery
- `vitals-predictions-writer.py` - Writes vitals and predictions to BigQuery
- `requirements.txt` - Python dependencies

### Google Cloud Services Used

- Cloud Run
- Cloud Functions (Gen 2)
- Pub/Sub
- BigQuery
- Cloud Storage
- Cloud Scheduler

## Usage

Once deployed, the system runs automatically:
1. Cloud Scheduler triggers data generation every minute
2. Sensor data flows through Pub/Sub topics
3. Data is aggregated and predictions are made
4. Results are stored in BigQuery for analysis

## License

MIT
