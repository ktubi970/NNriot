"""
Test script to verify the initialization of the TensorFlow continuous trainer.
Ensures that the environment is correctly configured for model loading.
"""

import logging
import traceback
from continuous_trainer import ContinuousTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_init_trainer():
    """
    Test initialization of the ContinuousTrainer class.

    Verifies that the trainer can load the model and its configurations
    correctly within the current environment.
    """
    try:
        _ = ContinuousTrainer()
        print("SUCCESS: ContinuousTrainer initialized successfully.")
        return True
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"FAILURE: Failed to load TF trainer: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_init_trainer()
