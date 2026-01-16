"""
Cloud Function: Vitals and Predictions Writer

Writes both vitals data and predictions to a new BigQuery table for dashboard use.
Triggered by predictions-output Pub/Sub topic.
"""

import base64
import json
import os
from datetime import datetime
from google.cloud import bigquery
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET_ID = os.environ.get("DATASET_ID", "healthcare_predictions")
TABLE_ID = os.environ.get("TABLE_ID", "vitals_and_predictions")

# Initialize BigQuery client
bigquery_client = bigquery.Client()


def extract_vitals_from_features(input_data: dict) -> dict:
    """
    Extract raw vitals from feature data (using t0 as primary timestamp).
    
    Args:
        input_data: Dictionary containing feature_* keys from prediction input
        
    Returns:
        Dictionary with vitals data
    """
    # Extract vitals from t0 (latest/most recent reading)
    vitals = {
        # Glucose (metabolic)
        "glucose_mgdl": input_data.get("feature_t0_glucose_mgdl"),
        
        # Heart metrics (vitals)
        "heart_rate_bpm": input_data.get("feature_t0_heart_rate_bpm"),
        "hrv_sdnn": input_data.get("feature_t0_hrv_sdnn"),
        "qt_interval_ms": input_data.get("feature_t0_qt_interval_ms"),
        
        # Respiratory (vitals)
        "respiratory_rate_rpm": input_data.get("feature_t0_respiratory_rate_rpm"),
        "spo2_pct": input_data.get("feature_t0_spo2_pct"),
        
        # Wearable metrics
        "steps_per_minute": input_data.get("feature_t0_steps_per_minute"),
        "vertical_acceleration_g": input_data.get("feature_t0_vertical_acceleration_g"),
        "skin_temperature_c": input_data.get("feature_t0_skin_temperature_c"),
        "eda_microsiemens": input_data.get("feature_t0_eda_microsiemens"),
        
        # Metabolic
        "insulin_on_board": input_data.get("feature_t0_insulin_on_board"),
        "carbs_in_stomach": input_data.get("feature_t0_carbs_in_stomach"),
        "activity_intensity": input_data.get("feature_t0_activity_intensity"),
    }
    
    return vitals


def write_vitals_and_predictions(event, context):
    """
    Cloud Function triggered by Pub/Sub.
    Writes vitals and predictions data to BigQuery.
    
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
        prediction_result = json.loads(message_data)
        
        logger.info(f"Received prediction for patient: {prediction_result.get('patient_id', 'UNKNOWN')}")
        
        # Check if prediction was successful
        if prediction_result.get("status") != "success":
            logger.warning(f"Skipping failed prediction: {prediction_result.get('error')}")
            return
        
        # Prepare row for BigQuery
        row = prepare_vitals_and_predictions_row(prediction_result)
        
        # Insert into BigQuery
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
        errors = bigquery_client.insert_rows_json(table_ref, [row])
        
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
            raise Exception(f"Failed to insert row: {errors}")
        else:
            logger.info(f"Inserted vitals and predictions for {row['patient_id']} into BigQuery")
            logger.info(f"   Glucose: {row.get('glucose_mgdl', 'N/A')}, HR: {row.get('heart_rate_bpm', 'N/A')}")
            logger.info(f"   Risks: Hypo={row['hypoglycemia_risk']}, Fall={row['fall_risk']}, Cardiac={row['cardiac_risk']}")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        logger.error(f"Error writing to BigQuery: {e}", exc_info=True)
        raise


def prepare_vitals_and_predictions_row(prediction_result: dict) -> dict:
    """
    Prepare vitals and predictions data for BigQuery insertion.
    
    Args:
        prediction_result: Prediction result from ML model (should include input_data)
        
    Returns:
        Dictionary formatted for BigQuery
    """
    predictions = prediction_result.get("predictions", {})
    
    # Extract prediction values
    hypoglycemia = predictions.get("label_hypoglycemia_risk", {})
    fall = predictions.get("label_fall_risk", {})
    cardiac = predictions.get("label_cardiac_anomaly", {})
    hypotension = predictions.get("label_severe_hypotension_risk", {})
    autonomic = predictions.get("label_autonomic_dysregulation", {})
    
    # Extract raw vitals from input_data
    input_data = prediction_result.get("input_data", {})
    vitals = extract_vitals_from_features(input_data)
    
    # Create BigQuery row with both vitals and predictions
    row = {
        "prediction_id": f"{prediction_result.get('patient_id', 'UNKNOWN')}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "patient_id": prediction_result.get("patient_id", "UNKNOWN"),
        "timestamp": prediction_result.get("timestamp", datetime.utcnow().isoformat()),
        "inserted_at": datetime.utcnow().isoformat(),
        
        # Vitals data (from t0 timestamp - most recent reading)
        "glucose_mgdl": vitals.get("glucose_mgdl"),
        "heart_rate_bpm": vitals.get("heart_rate_bpm"),
        "hrv_sdnn": vitals.get("hrv_sdnn"),
        "qt_interval_ms": vitals.get("qt_interval_ms"),
        "respiratory_rate_rpm": vitals.get("respiratory_rate_rpm"),
        "spo2_pct": vitals.get("spo2_pct"),
        "steps_per_minute": vitals.get("steps_per_minute"),
        "vertical_acceleration_g": vitals.get("vertical_acceleration_g"),
        "skin_temperature_c": vitals.get("skin_temperature_c"),
        "eda_microsiemens": vitals.get("eda_microsiemens"),
        "insulin_on_board": vitals.get("insulin_on_board"),
        "carbs_in_stomach": vitals.get("carbs_in_stomach"),
        "activity_intensity": vitals.get("activity_intensity"),
        
        # Hypoglycemia risk predictions
        "hypoglycemia_risk": hypoglycemia.get("prediction", 0),
        "hypoglycemia_probability": hypoglycemia.get("probability", 0.0),
        "hypoglycemia_confidence": hypoglycemia.get("confidence", 0.0),
        "hypoglycemia_risk_level": hypoglycemia.get("risk_level", "LOW"),
        
        # Fall risk predictions
        "fall_risk": fall.get("prediction", 0),
        "fall_probability": fall.get("probability", 0.0),
        "fall_confidence": fall.get("confidence", 0.0),
        "fall_risk_level": fall.get("risk_level", "LOW"),
        
        # Cardiac anomaly predictions
        "cardiac_risk": cardiac.get("prediction", 0),
        "cardiac_probability": cardiac.get("probability", 0.0),
        "cardiac_confidence": cardiac.get("confidence", 0.0),
        "cardiac_risk_level": cardiac.get("risk_level", "LOW"),
        
        # Hypotension risk predictions
        "hypotension_risk": hypotension.get("prediction", 0),
        "hypotension_probability": hypotension.get("probability", 0.0),
        "hypotension_confidence": hypotension.get("confidence", 0.0),
        "hypotension_risk_level": hypotension.get("risk_level", "LOW"),
        
        # Autonomic dysregulation predictions
        "autonomic_risk": autonomic.get("prediction", 0),
        "autonomic_probability": autonomic.get("probability", 0.0),
        "autonomic_confidence": autonomic.get("confidence", 0.0),
        "autonomic_risk_level": autonomic.get("risk_level", "LOW"),
        
        # Metadata
        "models_used": prediction_result.get("metadata", {}).get("models_used", 0),
        "num_features": prediction_result.get("metadata", {}).get("num_features", 0),
    }
    
    return row
