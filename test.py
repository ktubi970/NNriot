import nnriot
import sys
import numpy as np

print(f"Python executable: {sys.executable}")
print(f"nnriot file: {getattr(nnriot, '__file__', 'unknown')}")

try:
    # Test NeuralNetwork initialization with dummy data
    dummy_graph = b"dummy"
    print(
        "Testing nnriot.NeuralNetwork initialization (should fail with invalid bytes but prove attribute exists)..."
    )
    net = nnriot.NeuralNetwork(
        dummy_graph,
        input_op_name="Input",
        label_op_name="Label",
        train_op_name="Train",
        loss_op_name="Loss",
        output_op_name="Output",
    )
    print("Initialization successful (unexpected with dummy bytes).")
except Exception as e:
    print(f"Expected failure or error during initialization: {e}")

print("Checking available attributes in nnriot:")
print([attr for attr in dir(nnriot) if not attr.startswith("__")])
