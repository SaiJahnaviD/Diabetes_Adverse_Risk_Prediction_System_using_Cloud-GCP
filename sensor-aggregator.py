"""
Cloud Function: 5-Minute Aggregator & Feature Engineer

Triggered by raw-sensor-data Pub/Sub topic.
Aggregates 5-minute windows and publishes to predictions-input topic.

Deploy with:
    gcloud functions deploy sensor-aggregator \
        --gen2 \
        --runtime=python311 \
        --region=us-central1 \
        --source=. \
        --entry-point=aggregate_sensor_data \
        --trigger-topic=raw-sensor-data \
        --set-env-vars PROJECT_ID=your-project,OUTPUT_TOPIC=predictions-input \
        --memory=512MB \
        --timeout=540s
"""

import base64
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from google.cloud import pubsub_v1
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.environ.get("PROJECT_ID")
OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "predictions-input")
WINDOW_MINUTES = 5  # Aggregate 5-minute windows

# Global state to store sensor data per patient
# In production, use Cloud Datastore or Redis for persistent storage
patient_windows = defaultdict(list)


def extract_features(sensor_data_list: list) -> dict:
    """
    Extract features from 5-minute window of sensor data.
    Creates features matching the training data format.
    
    Args:
        sensor_data_list: List of sensor readings (should be ~5 readings for 5 minutes)
        
    Returns:
        Dictionary with feature_t0 through feature_t4 format
    """
    # Sort by timestamp
    sensor_data_list.sort(key=lambda x: x['timestamp'])
    
    # Take up to 5 time points (t0, t1, t2, t3, t4)
    # If we have more, sample evenly; if less, interpolate/pad
    n_points = min(len(sensor_data_list), 5)
    
    if n_points == 0:
        return None
    
    # Sample evenly across the window
    indices = [int(i * (len(sensor_data_list) - 1) / 4) for i in range(5)] if n_points >= 5 else list(range(n_points))
    sampled_data = [sensor_data_list[i] if i < len(sensor_data_list) else sensor_data_list[-1] for i in indices]
    
    # Extract features for each time point
    features = {}
    
    for t, data in enumerate(sampled_data):
        # Extract features matching the model's expected format (14 features per timestep)
        features[f"feature_t{t}_glucose_mgdl"] = data.get("glucose_mgdl", 100.0)
        features[f"feature_t{t}_heart_rate_bpm"] = data.get("heart_rate_bpm", 70.0)
        features[f"feature_t{t}_hrv_sdnn"] = data.get("hrv_sdnn", 50.0)
        features[f"feature_t{t}_respiratory_rate_rpm"] = data.get("respiration_rate", 16.0)  # Map respiration_rate to respiratory_rate_rpm
        features[f"feature_t{t}_spo2_pct"] = data.get("spo2_pct", 98.0)
        features[f"feature_t{t}_steps_per_minute"] = data.get("steps_per_minute", 0)
        features[f"feature_t{t}_vertical_acceleration_g"] = data.get("vertical_acceleration_g", 0.0)
        features[f"feature_t{t}_skin_temperature_c"] = data.get("skin_temperature_c", 37.0)
        features[f"feature_t{t}_eda_microsiemens"] = data.get("eda_microsiemens", 0.0)
        features[f"feature_t{t}_qt_interval_ms"] = data.get("qt_interval_ms", 400)
        features[f"feature_t{t}_insulin_on_board"] = data.get("insulin_on_board", 0.5)  # From metabolic state
        features[f"feature_t{t}_carbs_in_stomach"] = data.get("carbs_in_stomach", 0.0)  # From metabolic state
        features[f"feature_t{t}_activity_intensity"] = data.get("activity_intensity", 0.0)
        # hour_of_day will be added below from timestamp
    
    # Add metadata and hour_of_day feature (extracted from timestamp)
    latest_data = sampled_data[-1]
    features["patient_id"] = latest_data.get("patient_id", "UNKNOWN")
    timestamp_str = latest_data.get("timestamp", datetime.now().isoformat())
    features["timestamp"] = timestamp_str
    
    # Extract hour_of_day from timestamp for all timesteps
    try:
        if isinstance(timestamp_str, str):
            ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            ts = timestamp_str
        hour_of_day = ts.hour
    except:
        hour_of_day = 12  # Default to midday
    
    # Add hour_of_day for each timestep (model expects this as a feature)
    for t in range(5):
        features[f"feature_t{t}_hour_of_day"] = hour_of_day
    
    features["activity_mode"] = latest_data.get("activity_mode", "SEDENTARY")
    features["glucose_trend"] = latest_data.get("glucose_trend", "STABLE")
    
    return features


def aggregate_sensor_data(event, context):
    """
    Cloud Function triggered by Pub/Sub.
    Aggregates sensor data into 5-minute windows and publishes features.
    
    Args:
        event: The dictionary with data specific to this type of event.
        context: The Cloud Functions event metadata.
    """
    # Decode Pub/Sub message
    if "data" not in event:
        logger.error("No data in event")
        return
    
    try:
        message_data = base64.b64decode(event["data"]).decode("utf-8")
        sensor_data = json.loads(message_data)
        
        # Handle both nested and flattened message formats
        if "meta" in sensor_data and "vitals" in sensor_data:
            # Nested format - flatten it
            logger.info("Converting nested message format to flat format")
            sensor_data = {
                "patient_id": sensor_data.get("meta", {}).get("device_id", "UNKNOWN"),
                "timestamp": sensor_data.get("meta", {}).get("timestamp"),
                "glucose_mgdl": sensor_data.get("metabolics", {}).get("glucose_mgdl", 100.0),
                "heart_rate_bpm": sensor_data.get("vitals", {}).get("heart_rate_bpm", 70.0),
                "respiration_rate": sensor_data.get("vitals", {}).get("resp_rate_rpm", 16.0),
                "hrv_sdnn": sensor_data.get("vitals", {}).get("hrv_sdnn", 50.0),
                "spo2_pct": sensor_data.get("vitals", {}).get("spo2_pct", 98.0),
                "qt_interval_ms": sensor_data.get("vitals", {}).get("qt_interval_ms", 400),
                "glucose_trend": sensor_data.get("metabolics", {}).get("trend_arrow", "STABLE"),
                "steps_per_minute": sensor_data.get("wearable", {}).get("steps_per_minute", 0),
                "skin_temperature_c": sensor_data.get("wearable", {}).get("skin_temp_c", 37.0),
                "eda_microsiemens": sensor_data.get("wearable", {}).get("eda_microsiemens", 0.0),
                # Default values for missing fields
                "systolic_bp": 120.0,
                "diastolic_bp": 80.0,
                "temperature_celsius": sensor_data.get("wearable", {}).get("skin_temp_c", 37.0),
                "hrv_rmssd": 40.0,
                "activity_intensity": 0.0,
                "stress_level": 0.0,
                "sleep_quality": 0.5,
                "steps_count": sensor_data.get("wearable", {}).get("steps_per_minute", 0),
                "calories_burned": 0.0,
                "activity_mode": "SEDENTARY",
                "vertical_acceleration_g": sensor_data.get("wearable", {}).get("accel_y_g", 0.0),
                # Add missing metabolic fields (not in nested format, use defaults)
                "insulin_on_board": 0.5,  # Default residual insulin
                "carbs_in_stomach": 0.0,  # Default no carbs
            }
        
        patient_id = sensor_data.get("patient_id", "UNKNOWN")
        timestamp_str = sensor_data.get("timestamp")
        
        logger.info(f"Received data for {patient_id} at {timestamp_str}")
        
        # Add to patient's window
        patient_windows[patient_id].append(sensor_data)
        
        # Check if we have enough data for a 5-minute window
        if len(patient_windows[patient_id]) >= 5:
            # Extract features
            features = extract_features(patient_windows[patient_id])
            
            if features:
                # Publish to predictions-input topic
                publisher = pubsub_v1.PublisherClient()
                topic_path = publisher.topic_path(PROJECT_ID, OUTPUT_TOPIC)
                
                message_bytes = json.dumps(features).encode("utf-8")
                future = publisher.publish(topic_path, data=message_bytes)
                message_id = future.result()
                
                logger.info(f"Published features for {patient_id}: {message_id}")
                logger.info(f"   Glucose: {features['feature_t0_glucose_mgdl']:.1f} -> {features['feature_t4_glucose_mgdl']:.1f}")
                
                # Clear window (keep last reading for continuity)
                patient_windows[patient_id] = [patient_windows[patient_id][-1]]
            else:
                logger.warning(f"Could not extract features for {patient_id}")
        else:
            logger.info(f"Collecting data for {patient_id}: {len(patient_windows[patient_id])}/5")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}", exc_info=True)
        raise


# For local testing
if __name__ == "__main__":
    # Mock event and context
    class Context:
        event_id = "test-event-123"
        timestamp = "2024-12-14T12:00:00Z"
    
    test_data = {
        "patient_id": "PATIENT_001",
        "timestamp": "2024-12-14T12:00:00",
        "glucose_mgdl": 95.0,
        "heart_rate_bpm": 72,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "respiration_rate": 16.0,
        "temperature_celsius": 37.0,
        "hrv_sdnn": 45.0,
        "hrv_rmssd": 40.0,
        "activity_mode": "WALKING",
        "activity_intensity": 0.3,
        "stress_level": 0.2,
        "sleep_quality": 0.8,
        "steps_count": 100,
        "calories_burned": 5.0,
        "glucose_trend": "STABLE",
    }
    
    event = {
        "data": base64.b64encode(json.dumps(test_data).encode("utf-8"))
    }
    
    aggregate_sensor_data(event, Context())
