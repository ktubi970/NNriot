import numpy as np
import json_utils
import plotly.graph_objects as go
from generate_graph import build_keras_model

# 1. Define arbitrary JSON data (different shapes)
json_inputs = [
    {"type": "login", "details": {"success": True, "attempts": 1}},
    {"type": "purchase", "amount": 50.0, "items": ["book", "pen"]},
    {"type": "login", "details": {"success": False, "attempts": 3}},
    {"type": "refund", "amount": 20.0, "reason": "damaged"},
]
# Define labels (e.g., [1, 0] for "safe", [0, 1] for "alert")
y_data = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

# 2. Vectorize JSON inputs
VECTOR_DIM = 100000
x_data_sparse = json_utils.json_to_vector(json_inputs, dim=VECTOR_DIM)
x_data = x_data_sparse.toarray()

print(f"Vectorized JSON data shape: {x_data.shape}")

# 3. Initialize network
print("Initializing Neural Network...")
model = build_keras_model(VECTOR_DIM)

# 5. Train
print("Starting training on JSON features...")
try:
    history = model.fit(x_data, y_data, epochs=1000, verbose=0)
    losses = history.history["loss"]
    print(f"Training complete. Final loss: {losses[-1]:.6f}")

    # Plotting
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(losses))),
            y=losses,
            mode="lines",
            line=dict(color="#00ff88"),
        )
    )
    fig.update_layout(title="JSON Feature Training Loss", template="plotly_dark")
    fig.write_html("loss_plot.html")
    print("Plot saved to loss_plot.html")

except Exception as e:
    print(f"Training failed: {e}")
    exit(1)

# 6. Predict on a NEW JSON shape
print("\nPredicting on unseen JSON structure...")
test_json = {
    "type": "login",
    "details": {"success": False, "attempts": 10},
    "ip": "192.168.1.1",
}
test_vec = json_utils.json_to_vector(test_json, dim=VECTOR_DIM).toarray()

try:
    prediction = model.predict(test_vec, verbose=0)
    print(f"Input: {test_json}")
    print(f"Risk Prediction: {prediction[0][0]:.4f}")
    print(f"Full prediction: {prediction[0]}")
except Exception as e:
    print(f"Prediction failed: {e}")
