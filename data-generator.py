"""
Cloud Run Data Generator Service

Generates synthetic patient data using simulation_engine and publishes to Pub/Sub.
Triggered by Cloud Scheduler every 1 minute.

Deploy with:
    gcloud run deploy data-generator \
        --source . \
        --region=us-central1 \
        --set-env-vars PROJECT_ID=your-project,OUTPUT_TOPIC=raw-sensor-data \
        --memory=1Gi
"""

import json
import os
import sys
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from google.cloud import pubsub_v1
import logging

# Add parent directories to path to import simulation modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from data_models import PatientStaticProfile, PatientDynamicState, Sex, ActivityMode
from simulation_engine import process_single_patient
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Environment variables
PROJECT_ID = os.environ.get("PROJECT_ID")
OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "raw-sensor-data")
NUM_PATIENTS = int(os.environ.get("NUM_PATIENTS", "578"))  # Simulate N patients

# Global state to maintain patient states across invocations
patient_states = {}
patient_profiles = {}


def initialize_patient(patient_id: str) -> tuple:
    """Initialize a new patient with random profile."""
    rng = np.random.default_rng()
    
    # Use the generate_random method which handles all the cascading demographics
    profile = PatientStaticProfile.generate_random(rng=rng)
    
    # Override patient_id to use our specific ID
    from dataclasses import replace
    profile = replace(profile, patient_id=patient_id)
    
    # Create initial state with default values
    state = PatientDynamicState(
        timestamp_utc=datetime.now(timezone.utc),
        simulation_tick=0,
        current_activity_mode=ActivityMode.SEDENTARY,
        activity_intensity=0.1,
        cumulative_fatigue=0.0,
    )
    
    return profile, state


def sensor_payload_to_dict(payload) -> dict:
    """Convert SensorPayload to dictionary for JSON serialization.
    
    Flatten the nested structure for backward compatibility with the pipeline.
    """
    nested = payload.to_dict(include_waveforms=False)
    
    # Flatten the nested structure
    flat_data = {
        # Meta fields
        "message_id": nested["meta"]["message_id"],
        "timestamp": nested["meta"]["timestamp"],
        "patient_id": nested["meta"]["device_id"],  # device_id is the patient_id
        
        # Vitals
        "heart_rate_bpm": nested["vitals"]["heart_rate_bpm"],
        "hrv_sdnn": nested["vitals"]["hrv_sdnn"],
        "qt_interval_ms": nested["vitals"]["qt_interval_ms"],
        "spo2_pct": nested["vitals"]["spo2_pct"],
        "respiration_rate": nested["vitals"]["resp_rate_rpm"],
        
        # Metabolics
        "glucose_mgdl": nested["metabolics"]["glucose_mgdl"],
        "glucose_trend": nested["metabolics"]["trend_arrow"],
        
        # Wearable
        "steps_per_minute": nested["wearable"]["steps_per_minute"],
        "vertical_acceleration_g": nested["wearable"]["accel_y_g"],
        "skin_temperature_c": nested["wearable"]["skin_temp_c"],
        "eda_microsiemens": nested["wearable"]["eda_microsiemens"],
    }
    
    return flat_data


@app.route("/generate", methods=["POST", "GET"])
def generate_data():
    """
    Generate sensor data for all patients and publish to Pub/Sub.
    Triggered by Cloud Scheduler.
    """
    try:
        # Initialize publisher
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, OUTPUT_TOPIC)
        
        current_time = datetime.now(timezone.utc)
        messages_published = 0
        
        logger.info(f"Generating data for {NUM_PATIENTS} patients at {current_time}")
        
        # Generate data for each patient
        for i in range(NUM_PATIENTS):
            patient_id = f"PATIENT_{i+1:03d}"
            
            # Initialize patient if not exists
            if patient_id not in patient_states:
                profile, state = initialize_patient(patient_id)
                patient_profiles[patient_id] = profile
                patient_states[patient_id] = state
                logger.info(f"Initialized new patient: {patient_id}")
            
            # Get current state
            profile = patient_profiles[patient_id]
            state = patient_states[patient_id]
            
            # Update timestamp
            from dataclasses import replace
            state = replace(state, timestamp_utc=current_time)
            
            # Process patient (advance 1 minute)
            new_state, sensor_payload = process_single_patient(profile, state)
            
            # Update state
            patient_states[patient_id] = new_state
            
            # Convert to dict
            data = sensor_payload_to_dict(sensor_payload)
            
            # Add metabolic state fields that aren't in the payload
            data["insulin_on_board"] = new_state.metabolic_state.get("insulin_on_board_units", 0.5)
            data["carbs_in_stomach"] = new_state.metabolic_state.get("carbs_in_stomach_grams", 0.0)
            
            # Debug: Log structure for first patient
            if i == 0:
                logger.info(f"DEBUG: Data keys = {list(data.keys())}")
                logger.info(f"DEBUG: patient_id = {data.get('patient_id')}")
            
            # Publish to Pub/Sub
            message_bytes = json.dumps(data).encode("utf-8")
            future = publisher.publish(topic_path, data=message_bytes)
            message_id = future.result()
            
            messages_published += 1
            
            if i < 3:  # Log first few for debugging
                logger.info(f"Published data for {patient_id}: glucose={data['glucose_mgdl']}, HR={data['heart_rate_bpm']}")
        
        response = {
            "status": "success",
            "timestamp": current_time.isoformat(),
            "patients": NUM_PATIENTS,
            "messages_published": messages_published,
            "topic": OUTPUT_TOPIC,
        }
        
        logger.info(f"Published {messages_published} messages to {OUTPUT_TOPIC}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error generating data: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "data-generator",
        "patients_tracked": len(patient_states),
        "num_patients": NUM_PATIENTS,
    }), 200


@app.route("/reset", methods=["POST"])
def reset():
    """Reset all patient states (for testing)."""
    global patient_states, patient_profiles
    patient_states = {}
    patient_profiles = {}
    logger.info("Reset all patient states")
    return jsonify({"status": "success", "message": "All patient states reset"}), 200


@app.route("/", methods=["GET"])
def index():
    """Service information."""
    return jsonify({
        "service": "Healthcare Data Generator",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "Generate and publish sensor data",
            "/health": "Health check",
            "/reset": "Reset patient states"
        },
        "config": {
            "project_id": PROJECT_ID,
            "output_topic": OUTPUT_TOPIC,
            "num_patients": NUM_PATIENTS,
            "patients_tracked": len(patient_states)
        }
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
