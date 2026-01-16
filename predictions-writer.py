"""
Cloud Function: BigQuery Writer

Triggered by predictions-output Pub/Sub topic.
Writes prediction results to BigQuery for visualization in Looker Studio.

Deploy with:
    gcloud functions deploy predictions-writer \
        --gen2 \
        --runtime=python311 \
        --region=us-central1 \
        --source=. \
        --entry-point=write_to_bigquery \
        --trigger-topic=predictions-output \
        --set-env-vars PROJECT_ID=your-project,DATASET_ID=healthcare_predictions,TABLE_ID=predictions \
        --memory=256MB \
        --timeout=60s
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
TABLE_ID = os.environ.get("TABLE_ID", "predictions")

# Initialize BigQuery client
bigquery_client = bigquery.Client()


def write_to_bigquery(event, context):
    """
    Cloud Function triggered by Pub/Sub.
    Writes prediction results to BigQuery.
    
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
        row = prepare_bigquery_row(prediction_result)
        
        # Insert into BigQuery
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
        errors = bigquery_client.insert_rows_json(table_ref, [row])
        
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
            raise Exception(f"Failed to insert row: {errors}")
        else:
            logger.info(f"Inserted prediction for {row['patient_id']} into BigQuery")
            logger.info(f"   Risks: Hypo={row['hypoglycemia_risk']}, Fall={row['fall_risk']}, Cardiac={row['cardiac_risk']}")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        logger.error(f"Error writing to BigQuery: {e}", exc_info=True)
        raise


def prepare_bigquery_row(prediction_result: dict) -> dict:
    """
    Prepare prediction result for BigQuery insertion.
    
    Args:
        prediction_result: Prediction result from ML model
        
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
    
    # Create BigQuery row
    row = {
        "prediction_id": f"{prediction_result.get('patient_id', 'UNKNOWN')}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "patient_id": prediction_result.get("patient_id", "UNKNOWN"),
        "timestamp": prediction_result.get("timestamp", datetime.utcnow().isoformat()),
        "inserted_at": datetime.utcnow().isoformat(),
        
        # Hypoglycemia risk
        "hypoglycemia_risk": hypoglycemia.get("prediction", 0),
        "hypoglycemia_probability": hypoglycemia.get("probability", 0.0),
        "hypoglycemia_confidence": hypoglycemia.get("confidence", 0.0),
        "hypoglycemia_risk_level": hypoglycemia.get("risk_level", "LOW"),
        
        # Fall risk
        "fall_risk": fall.get("prediction", 0),
        "fall_probability": fall.get("probability", 0.0),
        "fall_confidence": fall.get("confidence", 0.0),
        "fall_risk_level": fall.get("risk_level", "LOW"),
        
        # Cardiac anomaly
        "cardiac_risk": cardiac.get("prediction", 0),
        "cardiac_probability": cardiac.get("probability", 0.0),
        "cardiac_confidence": cardiac.get("confidence", 0.0),
        "cardiac_risk_level": cardiac.get("risk_level", "LOW"),
        
        # Hypotension risk
        "hypotension_risk": hypotension.get("prediction", 0),
        "hypotension_probability": hypotension.get("probability", 0.0),
        "hypotension_confidence": hypotension.get("confidence", 0.0),
        "hypotension_risk_level": hypotension.get("risk_level", "LOW"),
        
        # Autonomic dysregulation
        "autonomic_risk": autonomic.get("prediction", 0),
        "autonomic_probability": autonomic.get("probability", 0.0),
        "autonomic_confidence": autonomic.get("confidence", 0.0),
        "autonomic_risk_level": autonomic.get("risk_level", "LOW"),
        
        # Metadata
        "models_used": prediction_result.get("metadata", {}).get("models_used", 0),
        "num_features": prediction_result.get("metadata", {}).get("num_features", 0),
    }
    
    return row


# For local testing
if __name__ == "__main__":
    # Mock event and context
    class Context:
        event_id = "test-event-123"
        timestamp = "2024-12-14T12:00:00Z"
    
    test_result = {
        "status": "success",
        "patient_id": "PATIENT_001",
        "timestamp": "2024-12-14T12:00:00",
        "predictions": {
            "label_hypoglycemia_risk": {
                "prediction": 0,
                "probability": 0.15,
                "confidence": 0.85,
                "risk_level": "LOW"
            },
            "label_fall_risk": {
                "prediction": 0,
                "probability": 0.08,
                "confidence": 0.92,
                "risk_level": "LOW"
            },
            "label_cardiac_anomaly": {
                "prediction": 0,
                "probability": 0.12,
                "confidence": 0.88,
                "risk_level": "LOW"
            },
            "label_severe_hypotension_risk": {
                "prediction": 0,
                "probability": 0.10,
                "confidence": 0.90,
                "risk_level": "LOW"
            },
            "label_autonomic_dysregulation": {
                "prediction": 0,
                "probability": 0.20,
                "confidence": 0.80,
                "risk_level": "LOW"
            }
        },
        "metadata": {
            "num_features": 79,
            "models_used": 5
        }
    }
    
    event = {
        "data": base64.b64encode(json.dumps(test_result).encode("utf-8"))
    }
    
    write_to_bigquery(event, Context())
