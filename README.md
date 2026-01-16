# 🩺 Cloud-Based Health Monitoring System

Real-time healthcare prediction system using Google Cloud Platform for monitoring patient vitals and predicting health risks.

## Tools
- **Python** - Data generation, ML models, Flask APIs, data transformation  
- **Google Cloud Platform** - Cloud Run, Cloud Functions (Gen 2), Pub/Sub, BigQuery, Cloud Storage, Cloud Scheduler  
- **SQL** - BigQuery queries, data aggregation, relationship modeling  
- **Flask** - RESTful API for prediction service  
- **Machine Learning** - Health risk prediction models (scikit-learn, TensorFlow)  
- **Visualization Tools** - Dashboard creation for stakeholder insights  

## Goal
Identify patient health risks in real-time by monitoring vital signs and predicting adverse events to support proactive clinical interventions.  

**Health Risks Predicted:** Hypoglycemia, Fall risk, Cardiac anomalies, Severe hypotension, Autonomic dysregulation  

## Process Overview
1. **Extract:** Generated synthetic patient sensor data (heart rate, blood pressure, temperature, glucose) via Cloud Run service triggered by Cloud Scheduler  
2. **Transform:** Aggregated streaming sensor readings into 5-minute windows using Cloud Functions, cleaned and normalized data for ML input  
3. **Load:** Published aggregated data to Pub/Sub topics, ingested predictions into BigQuery tables for analysis  
4. **Predict:** Deployed ML models via Cloud Run Flask API to generate real-time health risk scores  
5. **Visualize:** Created BigQuery-powered dashboards displaying patient vitals and risk predictions  
6. **Decide:** Delivered actionable health insights to clinical stakeholders through interactive dashboards  

## Data Architecture / Pipeline
1. **Cloud Scheduler** → Triggers data generation every minute  
2. **Data Generator (Cloud Run)** → Publishes sensor data to Pub/Sub  
3. **Sensor Aggregator (Cloud Function)** → Aggregates 5-min windows  
4. **Health Predictor (Cloud Run)** → ML prediction service  
5. **Predictions Writer (Cloud Function)** → Stores results in BigQuery  
6. **Vitals Writer (Cloud Function)** → Dashboard data preparation  


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

## Key Tables
- **sensor_data** → Raw patient vitals (`patient_id`, `timestamp`, `vitals`)  
- **aggregated_vitals** → 5-minute aggregated windows  
- **predictions** → Health risk scores (`patient_id`, `risk_type`, `probability`)  

## Relationships
- `sensor_data` → `aggregated_vitals` (via `patient_id`, `time_window`)  
- `aggregated_vitals` → `predictions` (via `patient_id`, `timestamp`)  
- `predictions` → `dashboard_vitals` (via `patient_id`)  

## Key Queries
- Aggregating 5-minute sensor windows  
- Joining predictions with patient vitals  
- Calculating risk score trends  
- Identifying high-risk patients  

## Dashboard Insights - [Dashboard](https://raghuveer9303.github.io/mediguard-dashboard/)
- Real-time patient vital signs monitoring  
- Health risk probability scores  
- Trend analysis across patient populations  
- Alert triggers for critical thresholds  

## The Impact

**Key Findings:**  
- Successfully processed 1,440+ data points per patient daily (1 RPM)  
- Achieved sub-second prediction latency through Cloud Run autoscaling  
- Identified high-risk patients requiring immediate clinical attention  
- Enabled proactive intervention by predicting adverse events priorly  

**Technical Achievements:**  
- Built fully automated, serverless pipeline requiring zero manual effort  
- Implemented scalable architecture handling multiple patients  
- Integrated 5 GCP services into cohesive data workflow  


## Usage

Once deployed, the system runs automatically:
1. Cloud Scheduler triggers data generation every minute
2. Sensor data flows through Pub/Sub topics
3. Data is aggregated and predictions are made
4. Results are stored in BigQuery for analysis
