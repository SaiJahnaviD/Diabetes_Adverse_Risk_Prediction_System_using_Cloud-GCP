"""
Healthcare Predictor Module

Loads trained models from GCS and makes predictions on input data.
"""

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcarePredictor:
    """Healthcare risk predictor using trained XGBoost models."""
    
    def __init__(self, bucket_name: str, models_path: str = "models"):
        """
        Initialize predictor with models from GCS.
        
        Args:
            bucket_name: GCS bucket name (without gs:// prefix)
            models_path: Path to models within bucket (default: 'models')
        """
        self.bucket_name = bucket_name
        self.models_path = models_path
        self.scaler = None
        self.models = {}
        self.feature_names = None
        self._load_models()
    
    def _download_from_gcs(self, blob_path: str, local_path: str):
        """Download file from GCS."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded {blob_path} to {local_path}")
    
    def _load_models(self):
        """Load scaler and models from GCS."""
        logger.info(f"Loading models from gs://{self.bucket_name}/{self.models_path}")
        
        # Load scaler
        try:
            scaler_path = f"{self.models_path}/scaler.pkl"
            local_scaler = "/tmp/scaler.pkl"
            self._download_from_gcs(scaler_path, local_scaler)
            self.scaler = joblib.load(local_scaler)
            logger.info("Scaler loaded")
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            raise
        
        # Load models
        model_names = [
            "label_hypoglycemia_risk",
            "label_fall_risk",
            "label_cardiac_anomaly",
            "label_severe_hypotension_risk",
            "label_autonomic_dysregulation",
        ]
        
        for model_name in model_names:
            try:
                model_path = f"{self.models_path}/xgboost_{model_name}.pkl"
                local_model = f"/tmp/xgboost_{model_name}.pkl"
                self._download_from_gcs(model_path, local_model)
                self.models[model_name] = joblib.load(local_model)
                logger.info(f"Loaded {model_name}")
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")
        
        if not self.models:
            raise RuntimeError("No models loaded successfully")
        
        logger.info(f"Total models loaded: {len(self.models)}")
    
    def _prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features from input data.
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            Scaled feature array ready for prediction
        """
        # Expected feature structure
        feature_cols = [col for col in data.keys() if col.startswith("feature_")]
        
        if not feature_cols:
            raise ValueError("No features found in input data (expecting 'feature_*' keys)")
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Add temporal features if timestamp is provided
        if "timestamp" in data:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["hour"] = df["timestamp"].dt.hour
                df["day_of_week"] = df["timestamp"].dt.dayofweek
                df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
            except Exception as e:
                logger.warning(f"Could not parse timestamp: {e}")
                df["hour"] = 12
                df["day_of_week"] = 3
                df["is_weekend"] = 0
        else:
            df["hour"] = 12  # default midday
            df["day_of_week"] = 3  # default Wednesday
            df["is_weekend"] = 0  # default weekday
        
        # Calculate change features
        if all(f"feature_t{i}_glucose_mgdl" in data for i in [0, 4]):
            df["glucose_change"] = (
                df["feature_t4_glucose_mgdl"] - df["feature_t0_glucose_mgdl"]
            )
        else:
            df["glucose_change"] = 0
        
        if all(f"feature_t{i}_heart_rate_bpm" in data for i in [0, 4]):
            df["heart_rate_change"] = (
                df["feature_t4_heart_rate_bpm"] - df["feature_t0_heart_rate_bpm"]
            )
        else:
            df["heart_rate_change"] = 0
        
        if all(f"feature_t{i}_hrv_sdnn" in data for i in [0, 4]):
            df["hrv_change"] = df["feature_t4_hrv_sdnn"] - df["feature_t0_hrv_sdnn"]
        else:
            df["hrv_change"] = 0
        
        if all(f"feature_t{i}_activity_intensity" in data for i in [0, 4]):
            df["activity_change"] = (
                df["feature_t4_activity_intensity"] - df["feature_t0_activity_intensity"]
            )
        else:
            df["activity_change"] = 0
        
        # Calculate glucose statistics
        glucose_cols = [f"feature_t{i}_glucose_mgdl" for i in range(5)]
        available_glucose = [col for col in glucose_cols if col in data]
        if available_glucose:
            df["glucose_mean"] = df[available_glucose].mean(axis=1)
            df["glucose_std"] = df[available_glucose].std(axis=1)
        else:
            df["glucose_mean"] = 0
            df["glucose_std"] = 0
        
        # Define all features (same order as training)
        additional_features = [
            "hour",
            "day_of_week",
            "is_weekend",
            "glucose_change",
            "heart_rate_change",
            "hrv_change",
            "activity_change",
            "glucose_mean",
            "glucose_std",
        ]
        all_features = feature_cols + additional_features
        
        # Get feature values
        X = df[all_features].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions on input data.
        
        Args:
            data: Dictionary containing:
                - feature_*: Feature values
                - patient_id: (optional) Patient identifier
                - timestamp: (optional) Timestamp string
                
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            # Prepare features
            X = self._prepare_features(data)
            
            # Make predictions with all models
            predictions = {}
            for model_name, model in self.models.items():
                # Get prediction and probability
                pred = model.predict(X)[0]
                pred_proba = model.predict_proba(X)[0]
                
                predictions[model_name] = {
                    "prediction": int(pred),
                    "probability": float(pred_proba[1]),
                    "confidence": float(max(pred_proba)),
                    "risk_level": self._get_risk_level(pred_proba[1]),
                }
            
            # Create response
            response = {
                "status": "success",
                "predictions": predictions,
                "metadata": {
                    "num_features": X.shape[1],
                    "models_used": len(self.models),
                },
            }
            
            # Add patient metadata
            if "patient_id" in data:
                response["patient_id"] = data["patient_id"]
            if "timestamp" in data:
                response["timestamp"] = data["timestamp"]
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }
    
    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """Convert probability to risk level."""
        if probability < 0.3:
            return "LOW"
        elif probability < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "bucket": self.bucket_name,
            "models_path": self.models_path,
            "num_models": len(self.models),
            "model_names": list(self.models.keys()),
            "scaler_loaded": self.scaler is not None,
        }
