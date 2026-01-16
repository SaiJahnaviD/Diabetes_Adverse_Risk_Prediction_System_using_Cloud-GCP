"""
Cloud Run Application for Healthcare Predictions

This Flask application handles prediction requests and publishes results to Pub/Sub.
Alternative to Cloud Functions with more control and flexibility.

Deploy with:
    gcloud run deploy healthcare-predictor \
        --source . \
        --region=us-central1 \
        --platform=managed \
        --allow-unauthenticated \
        --set-env-vars PROJECT_ID=your-project,BUCKET_NAME=your-bucket \
        --memory=2Gi \
        --cpu=2 \
        --min-instances=0 \
        --max-instances=10
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import pubsub_v1
from google.cloud import bigquery
from predictor import HealthcarePredictor
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Environment variables
PROJECT_ID = os.environ.get("PROJECT_ID")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "predictions-output")
DATASET_ID = os.environ.get("DATASET_ID", "healthcare_predictions")
VITALS_TABLE_ID = os.environ.get("VITALS_TABLE_ID", "vitals_and_predictions")

# Global instances (initialized once)
predictor = None
publisher = None
bigquery_client = None


def get_predictor():
    """Lazy load predictor."""
    global predictor
    if predictor is None:
        logger.info("Initializing predictor...")
        predictor = HealthcarePredictor(bucket_name=BUCKET_NAME)
        logger.info("Predictor initialized")
    return predictor


def get_publisher():
    """Lazy load Pub/Sub publisher."""
    global publisher
    if publisher is None:
        publisher = pubsub_v1.PublisherClient()
    return publisher


def get_bigquery_client():
    """Lazy load BigQuery client."""
    global bigquery_client
    if bigquery_client is None:
        bigquery_client = bigquery.Client()
    return bigquery_client


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction request.
    
    Can handle both:
    1. Direct JSON: {"patient_id": "...", "feature_t0_glucose_mgdl": 95.0, ...}
    2. Pub/Sub push: {"message": {"data": "base64encoded", "attributes": {...}}}
    
    Returns:
        JSON response with predictions
    """
    try:
        # Get request data
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({
                "status": "error",
                "error": "No data provided"
            }), 400
        
        # Handle Pub/Sub push format
        if "message" in request_data:
            # Pub/Sub push subscription format
            import base64
            message = request_data["message"]
            if "data" in message:
                decoded_data = base64.b64decode(message["data"]).decode("utf-8")
                data = json.loads(decoded_data)
            else:
                return jsonify({
                    "status": "error",
                    "error": "Invalid Pub/Sub message format"
                }), 400
        else:
            # Direct JSON format
            data = request_data
        
        logger.info(f"Received prediction request for patient: {data.get('patient_id', 'UNKNOWN')}")
        
        # Get predictor
        pred = get_predictor()
        
        # Make prediction
        result = pred.predict(data)
        
        # Include input_data in result so BigQuery can extract vitals
        if result.get("status") == "success":
            result["input_data"] = data
        
        # Publish to output topic if configured
        if OUTPUT_TOPIC:
            try:
                pub = get_publisher()
                topic_path = pub.topic_path(PROJECT_ID, OUTPUT_TOPIC)
                
                message_bytes = json.dumps(result).encode("utf-8")
                future = pub.publish(topic_path, data=message_bytes)
                message_id = future.result()
                
                logger.info(f"Published to {OUTPUT_TOPIC}: {message_id}")
                result["published_message_id"] = message_id
            except Exception as e:
                logger.warning(f"Failed to publish to Pub/Sub: {e}")
        
        # Return result
        status_code = 200 if result.get("status") == "success" else 500
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Handle batch prediction request.
    
    Expected JSON body:
    {
        "predictions": [
            {patient_data_1},
            {patient_data_2},
            ...
        ]
    }
    
    Returns:
        JSON response with all predictions
    """
    try:
        data = request.get_json()
        
        if not data or "predictions" not in data:
            return jsonify({
                "status": "error",
                "error": "No predictions array provided"
            }), 400
        
        predictions_input = data["predictions"]
        logger.info(f"Received batch prediction request for {len(predictions_input)} patients")
        
        # Get predictor
        pred = get_predictor()
        
        # Make predictions
        results = []
        for input_data in predictions_input:
            result = pred.predict(input_data)
            results.append(result)
        
        # Count successes and failures
        successes = sum(1 for r in results if r.get("status") == "success")
        failures = len(results) - successes
        
        response = {
            "status": "success",
            "total": len(results),
            "successes": successes,
            "failures": failures,
            "results": results
        }
        
        logger.info(f"Batch prediction complete: {successes} successes, {failures} failures")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        # Check if predictor is initialized
        pred = get_predictor()
        model_info = pred.get_model_info()
        
        return jsonify({
            "status": "healthy",
            "service": "healthcare-predictor",
            "models_loaded": model_info["num_models"],
            "bucket": model_info["bucket"]
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route("/ready", methods=["GET"])
def ready():
    """Readiness check endpoint."""
    try:
        if predictor is None:
            return jsonify({
                "status": "not_ready",
                "message": "Predictor not initialized"
            }), 503
        
        return jsonify({
            "status": "ready"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "not_ready",
            "error": str(e)
        }), 503


@app.route("/models", methods=["GET"])
def models():
    """Get information about loaded models."""
    try:
        pred = get_predictor()
        model_info = pred.get_model_info()
        
        return jsonify({
            "status": "success",
            "model_info": model_info
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/dashboard/data", methods=["GET"])
def dashboard_data():
    """
    Get joined vitals and predictions data for dashboard.
    
    Query parameters:
        patient_id: (optional) Filter by patient ID
        start_time: (optional) Start timestamp (ISO format)
        end_time: (optional) End timestamp (ISO format)
        limit: (optional) Maximum number of records (default: 1000)
        
    Returns:
        JSON response with vitals and predictions data
    """
    try:
        # Get query parameters
        patient_id = request.args.get("patient_id")
        start_time = request.args.get("start_time")
        end_time = request.args.get("end_time")
        limit = int(request.args.get("limit", 1000))
        
        # Build BigQuery query
        bq_client = get_bigquery_client()
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{VITALS_TABLE_ID}"
        
        query_parts = [
            "SELECT",
            "  prediction_id,",
            "  patient_id,",
            "  timestamp,",
            "  inserted_at,",
            "  -- Vitals",
            "  glucose_mgdl,",
            "  heart_rate_bpm,",
            "  hrv_sdnn,",
            "  qt_interval_ms,",
            "  respiratory_rate_rpm,",
            "  spo2_pct,",
            "  steps_per_minute,",
            "  vertical_acceleration_g,",
            "  skin_temperature_c,",
            "  eda_microsiemens,",
            "  insulin_on_board,",
            "  carbs_in_stomach,",
            "  activity_intensity,",
            "  -- Predictions",
            "  hypoglycemia_risk,",
            "  hypoglycemia_probability,",
            "  hypoglycemia_confidence,",
            "  hypoglycemia_risk_level,",
            "  fall_risk,",
            "  fall_probability,",
            "  fall_confidence,",
            "  fall_risk_level,",
            "  cardiac_risk,",
            "  cardiac_probability,",
            "  cardiac_confidence,",
            "  cardiac_risk_level,",
            "  hypotension_risk,",
            "  hypotension_probability,",
            "  hypotension_confidence,",
            "  hypotension_risk_level,",
            "  autonomic_risk,",
            "  autonomic_probability,",
            "  autonomic_confidence,",
            "  autonomic_risk_level,",
            "  models_used,",
            "  num_features",
            f"FROM `{table_ref}`",
            "WHERE 1=1",
        ]
        
        query_params = []
        
        if patient_id:
            query_parts.append("AND patient_id = @patient_id")
            query_params.append(bigquery.ScalarQueryParameter("patient_id", "STRING", patient_id))
        
        if start_time:
            query_parts.append("AND timestamp >= @start_time")
            query_params.append(bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", start_time))
        
        if end_time:
            query_parts.append("AND timestamp <= @end_time")
            query_params.append(bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", end_time))
        
        query_parts.append("ORDER BY timestamp DESC")
        query_parts.append(f"LIMIT {limit}")
        
        query = "\n".join(query_parts)
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        query_job = bq_client.query(query, job_config=job_config)
        
        # Execute query and get results
        results = query_job.result()
        
        # Convert to list of dictionaries
        data = []
        for row in results:
            data.append({
                "prediction_id": row.prediction_id,
                "patient_id": row.patient_id,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "inserted_at": row.inserted_at.isoformat() if row.inserted_at else None,
                "vitals": {
                    "glucose_mgdl": row.glucose_mgdl,
                    "heart_rate_bpm": row.heart_rate_bpm,
                    "hrv_sdnn": row.hrv_sdnn,
                    "qt_interval_ms": row.qt_interval_ms,
                    "respiratory_rate_rpm": row.respiratory_rate_rpm,
                    "spo2_pct": row.spo2_pct,
                    "steps_per_minute": row.steps_per_minute,
                    "vertical_acceleration_g": row.vertical_acceleration_g,
                    "skin_temperature_c": row.skin_temperature_c,
                    "eda_microsiemens": row.eda_microsiemens,
                    "insulin_on_board": row.insulin_on_board,
                    "carbs_in_stomach": row.carbs_in_stomach,
                    "activity_intensity": row.activity_intensity,
                },
                "predictions": {
                    "hypoglycemia": {
                        "risk": row.hypoglycemia_risk,
                        "probability": row.hypoglycemia_probability,
                        "confidence": row.hypoglycemia_confidence,
                        "risk_level": row.hypoglycemia_risk_level,
                    },
                    "fall": {
                        "risk": row.fall_risk,
                        "probability": row.fall_probability,
                        "confidence": row.fall_confidence,
                        "risk_level": row.fall_risk_level,
                    },
                    "cardiac": {
                        "risk": row.cardiac_risk,
                        "probability": row.cardiac_probability,
                        "confidence": row.cardiac_confidence,
                        "risk_level": row.cardiac_risk_level,
                    },
                    "hypotension": {
                        "risk": row.hypotension_risk,
                        "probability": row.hypotension_probability,
                        "confidence": row.hypotension_confidence,
                        "risk_level": row.hypotension_risk_level,
                    },
                    "autonomic": {
                        "risk": row.autonomic_risk,
                        "probability": row.autonomic_probability,
                        "confidence": row.autonomic_confidence,
                        "risk_level": row.autonomic_risk_level,
                    },
                },
                "metadata": {
                    "models_used": row.models_used,
                    "num_features": row.num_features,
                }
            })
        
        return jsonify({
            "status": "success",
            "count": len(data),
            "data": data
        }), 200
        
    except Exception as e:
        logger.error(f"Dashboard data query error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }), 500


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API documentation."""
    return jsonify({
        "service": "Healthcare Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Make a single prediction",
                "body": "JSON with patient features"
            },
            "/batch_predict": {
                "method": "POST",
                "description": "Make batch predictions",
                "body": '{"predictions": [patient_data_array]}'
            },
            "/health": {
                "method": "GET",
                "description": "Health check"
            },
            "/ready": {
                "method": "GET",
                "description": "Readiness check"
            },
            "/models": {
                "method": "GET",
                "description": "Get model information"
            },
            "/dashboard/data": {
                "method": "GET",
                "description": "Get joined vitals and predictions data for dashboard",
                "query_params": {
                    "patient_id": "(optional) Filter by patient ID",
                    "start_time": "(optional) Start timestamp (ISO format)",
                    "end_time": "(optional) End timestamp (ISO format)",
                    "limit": "(optional) Maximum records (default: 1000)"
                }
            }
        },
        "example_request": {
            "patient_id": "PATIENT_123",
            "timestamp": "2024-12-14T12:00:00",
            "feature_t0_glucose_mgdl": 95.0,
            "feature_t1_glucose_mgdl": 98.0,
            "feature_t2_glucose_mgdl": 102.0,
            "feature_t3_glucose_mgdl": 100.0,
            "feature_t4_glucose_mgdl": 97.0,
            "... (other features) ...": "..."
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "status": "error",
        "error": "Endpoint not found",
        "message": "Please check the API documentation at /"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "status": "error",
        "error": "Internal server error"
    }), 500


if __name__ == "__main__":
    # For local testing
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
