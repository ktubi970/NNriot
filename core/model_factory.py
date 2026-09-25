"""
model_factory.py - Factory pattern for Keras ML Models

Centralizes the instantiation, loading, and validation of TensorFlow/Keras models 
across the NNriot project.
"""

import os
import logging
import tensorflow as tf

from core.generate_graph import build_multi_output_model, build_keras_model
from core.feature_labels import VECTOR_DIM
from core.calibration import load_calibration_artifact

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating and loading NNriot models."""
    
    @staticmethod
    def get_model(version: str = "v2", input_dim: int = VECTOR_DIM, force_rebuild: bool = False, checkpoint_dir: str = ".", checkpoint_path: str = None):
        """
        Retrieves a trained model or builds a fresh architecture if none exists.
        
        Args:
            version: "v1" (single-output) or "v2" (multi-output)
            input_dim: The feature vector size
            force_rebuild: If True, ignores existing checkpoints
            checkpoint_dir: Directory where .keras files are stored
            checkpoint_path: Explicit file path to load/save model checkpoint
            
        Returns:
            Tuple of (keras.Model, str) representing the model and its checkpoint path
        """
        if version == "v2":
            path = checkpoint_path or os.path.join(checkpoint_dir, "model_v2.keras")
            builder = build_multi_output_model
        elif version == "v1":
            path = checkpoint_path or os.path.join(checkpoint_dir, "model.keras")
            builder = build_keras_model
        else:
            raise ValueError(f"Unknown model version: {version}")
            
        model = None
        if not force_rebuild and os.path.exists(path):
            candidate = None
            try:
                candidate = tf.keras.models.load_model(path)
                
                # Validation
                actual_input_dim = candidate.input_shape[-1]
                if actual_input_dim != input_dim:
                    raise ValueError(f"Input dim {actual_input_dim} != expected {input_dim}")
                    
                if version == "v2":
                    from core.feature_labels import ALL_LABEL_KEYS
                    actual_heads = set(candidate.output_names)
                    expected_heads = set(ALL_LABEL_KEYS)
                    if actual_heads != expected_heads:
                        raise ValueError(f"Heads mismatch. Missing: {expected_heads - actual_heads}")
            except Exception as e:
                logger.warning("Failed to load %s from %s: %s. Rebuilding...", version, path, e)
                candidate = None

            if candidate is not None:
                candidate.is_calibrated = False
                candidate.calibration_temperature = None
                candidate.calibration_metadata = None
                try:
                    metadata = load_calibration_artifact(path, input_dim=input_dim) if version == "v2" else None
                    if metadata is not None:
                        candidate.is_calibrated = True
                        candidate.calibration_temperature = metadata["temperature"]
                        candidate.calibration_metadata = metadata
                except Exception as cal_err:
                    logger.warning("Calibration loading encountered an error: %s", cal_err)
                logger.info("Model %s restored from %s.", version, path)
                return candidate, path

        # Rebuild
        logger.info("Building fresh %s architecture...", version)
        model = builder(input_dim=input_dim)
        model.is_calibrated = False
        model.calibration_temperature = None
        model.calibration_metadata = None
        return model, path
