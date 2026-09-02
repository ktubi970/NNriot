# NNriot

[![CI](https://github.com/ktubi970/NNriot/actions/workflows/ci.yml/badge.svg)](https://github.com/ktubi970/NNriot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework: TensorFlow 2.x](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange.svg)](https://www.tensorflow.org/)

A pure Python machine learning system for processing JSON data and predicting outcomes. NNriot demonstrates a sophisticated approach to handling variable-structure JSON inputs for machine learning tasks using feature hashing and deep neural networks.

## Overview

NNriot is a pure Python system designed to handle arbitrary JSON structures and learn from them to make predictions on new, potentially different JSON schemas. The system uses feature hashing for efficient vectorization and TensorFlow for neural network operations.

## Architecture

### Core Components & Design Patterns

- **Unified CLI (Command Pattern)**: Single entrypoint (`cli.py`) for web server, training loop, and data backfills.
- **Repository Pattern (`core/repository.py`)**: Clean abstraction decoupling database operations (SQLite/PostgreSQL) from business logic.
- **Factory Pattern (`core/model_factory.py`)**: Centralized model architecture creation and checkpoint lifecycle management.
- **Service Layer (`app/services/`)**: Encapsulates player search, alias resolution, and system health services.
- **Neural Engine**: Multi-task learning deep neural network predicting 22 concurrent game markets (`core/generate_graph.py`, `scripts/continuous_trainer.py`).
- **Modern Cyber Frontend**: TypeScript + Vite + Tailwind CSS Cockpit with Playwright E2E testing suite (`frontend/`, `e2e/`).
- **Data & ETL Pipelines**: Real-time Riot API collector, timeline backfilling, and Liquipedia import (`scripts/`, `core/riot_api.py`).

### Data Processing Pipeline

1. **JSON Vectorization** (`core/json_utils.py`): Converts arbitrary JSON structures into fixed-size vectors using feature hashing.
2. **Multi-Output Neural Network** (`core/generate_graph.py`): Deep multi-task learning architecture with shared representations and task-specific heads.
3. **Web Application & Cockpit** (`app/`, `frontend/`): High-performance Flask REST API and responsive React SPA interface.

## Features

- **Standalone Mode**: Efficient predictions without heavy dependencies like TensorFlow
- **Residual Networks**: Deep learning architecture with skip connections (available in Neural Engine mode)
- **High Performance**: Optimized for low-latency match outcome analysis
- **Production-ready**: Includes dependency management, error handling, and comprehensive testing

### Explainable Analytics

NNriot includes a separate, rebuildable `analytics.db` mart and a French chat
interface at `/analytics`. It reports retrospective gameplay associations with
sample sizes, confidence intervals, corrected q-values, robustness checks and
traceable evidence IDs. Regional comparisons are adjusted by patch; causal,
timeline and unsupported team claims are explicitly refused.

```bash
python -m analytics --source training_data.db --analytics analytics.db build
python -m analytics --source training_data.db --analytics analytics.db health
python run.py
```

See [the Analytics operating guide](docs/analytics.md) for supported questions,
thresholds, model evaluation, benchmark commands and limitations.

## Use Cases

- **JSON Risk Assessment**: Classifying JSON payloads as "safe" or "alert"
- **Gaming Analytics**: League of Legends match outcome prediction via Riot API integration
- **General JSON Classification**: Any scenario requiring ML on variable-structure JSON data

## Installation

### Prerequisites

- Python 3.8+
- Riot API key (for gaming features)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd NNriot
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your Riot API key
   # RIOT_API_KEY=your_api_key_here
   ```

## Usage

### Basic JSON Classification

```python
import tensorflow.compat.v1 as tf
import numpy as np
import json_utils
import plotly.graph_objects as go

# Disable eager execution for TF 1.x graph mode
tf.disable_eager_execution()

# Define training data
json_inputs = [
    {"type": "login", "details": {"success": True, "attempts": 1}},
    {"type": "purchase", "amount": 50.0, "items": ["book", "pen"]},
    {"type": "login", "details": {"success": False, "attempts": 3}},
    {"type": "refund", "amount": 20.0, "reason": "damaged"}
]

# Define labels (binary classification)
y_data = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

# Vectorize JSON inputs
VECTOR_DIM = 20000
x_data = json_utils.json_to_vector(json_inputs, dim=VECTOR_DIM)

# Create TensorFlow session and define network
sess = tf.Session()

# Define neural network architecture
x = tf.placeholder(tf.float32, shape=[None, 10000], name="Input")
y = tf.placeholder(tf.float32, shape=[None, 2], name="Label")

# [Network definition code here...]

# Initialize variables
sess.run(tf.global_variables_initializer())

# Train the network
losses = []
for epoch in range(1000):
    _, loss_val = sess.run([train_op, loss], feed_dict={x: x_data, y: y_data})
    losses.append(loss_val)

# Predict on new JSON
test_json = {"type": "login", "details": {"success": False, "attempts": 10}, "ip": "192.168.1.1"}
test_vec = json_utils.json_to_vector(test_json, dim=VECTOR_DIM)
prediction = sess.run(pred, feed_dict={x: test_vec})
print(f"Risk Prediction: {prediction[0][0]:.4f}")
```

### Quick Start

For a complete working example, simply run:
```bash
python main.py
```

This will demonstrate:
- JSON feature training
- Loss visualization (saved to `loss_plot.html`)
- Prediction on unseen JSON structures

### League of Legends Analytics

```python
from data_collector import collect_training_data
from riot_api import get_player_summary, get_matchup_data

# Collect training data from Riot API
seed_players = [
    ("Hide on bush", "KR1"),
    ("Chovy", "KR1"),
    ("T1 Gumayusi", "KR1")
]
collect_training_data(seed_players, matches_per_player=10)

# Get player statistics
faker_stats = get_player_summary("Hide on bush", "KR1", match_count=5)
print(faker_stats)

# Compare team matchups
team_a = [("Player1", "Tag1"), ("Player2", "Tag2")]
team_b = [("Player3", "Tag3"), ("Player4", "Tag4")]
matchup = get_matchup_data(team_a, team_b)
```

## Project Structure

```
NNriot/
├── cli.py                  # Unified CLI (Command pattern: web, train, backfill)
├── core/                   # Domain logic & Design Patterns
│   ├── config.py           # Configuration management
│   ├── database.py         # DB connection & low-level migrations
│   ├── feature_labels.py   # Label registries & multi-task targets
│   ├── generate_graph.py   # Keras multi-output neural network
│   ├── json_utils.py       # JSON vectorization & hashing
│   ├── model_factory.py    # Model Factory pattern
│   ├── repository.py       # Repository pattern (DAO layer)
│   └── riot_api.py         # Riot Games API client
├── app/                    # Web Application layer
│   ├── core.py             # Global state & async model runner
│   ├── services/           # Service layer (Player & System services)
│   └── routes/             # Modular Flask blueprints (API, views, lolpros)
├── scripts/                # ETL & Maintenance Scripts
│   ├── continuous_trainer.py # Continuous learning loop
│   ├── data_collector.py     # Riot API data pipeline
│   ├── backfill_labels.py    # Label backfilling
│   └── import_liquipedia.py  # Pro player scraping & import
├── frontend/               # Modern TypeScript + Vite + Tailwind UI
├── e2e/                    # Playwright End-to-End test suite
├── templates/              # Server-rendered fallback templates
├── tests/                  # Complete Pytest test suite (224 tests)
├── start.bat               # Interactive Cyber Launcher
└── requirements.txt        # Python dependencies
```

## Neural Network Architecture

The network uses a TensorFlow 1.x graph-based architecture with the following structure:

1. **Input Layer**: 20,000 features from JSON vectorization
2. **Projection**: Dense layer (20,000 → 1,024) with ReLU activation
3. **Residual Blocks**: Two residual blocks (1,024 → 1,024) with skip connections
4. **Bottleneck**: Dense layers (1,024 → 512 → 128) with ReLU activation
5. **Output**: Dense layer (128 → 2) with softmax activation for binary classification

## JSON Vectorization

The system uses enhanced feature engineering to convert arbitrary JSON structures into fixed-size vectors:

### Enhanced Features

1. **Smart Flattening**: Recursive flattening with depth control and path preservation
2. **Advanced Feature Encoding**: Multiple encodings per key-value pair:
   - Path-based features (`path:key.subkey`)
   - Type-based features (`type:int`, `type:str`)
   - Hash-based features for categorical values
3. **Value Normalization**: 
   - Log transformation for large numbers
   - String length normalization
   - Boolean to numeric conversion
4. **Feature Combinations**: Interaction features between related attributes
5. **N-gram Features**: Multi-level key path combinations
6. **L2 Normalization**: Unit vector normalization for better ML performance

### Key Improvements

- **Richer Representation**: Multiple feature encodings capture different aspects
- **Better Generalization**: Feature combinations improve model learning
- **Sparsity Control**: Configurable sparsity for memory optimization
- **Data Type Handling**: Advanced normalization for mixed data types
- **Feature Importance**: Built-in analysis for feature selection

### Usage Examples - League of Legends Match Data

```python
import json_utils

# League of Legends match data
lol_match = {
    "match_id": "EUW1_1234567890",
    "game_mode": "CLASSIC",
    "game_duration": 1850,
    "participants": [
        {
            "summoner_name": "Hide on bush",
            "champion_name": "Kaisa",
            "team_id": 100,
            "win": True,
            "kills": 18,
            "deaths": 2,
            "assists": 12,
            "gold_earned": 18500,
            "cs": 245,
            "role": "CARRY",
            "items": [3085, 3005, 3035, 3086, 3006, 3153]
        }
    ],
    "teams": [
        {"team_id": 100, "win": True, "dragon_kills": 3, "baron_kills": 1}
    ]
}

# Basic usage (backward compatible)
vectors = json_utils.json_to_vector(lol_match, dim=10000)

# Enhanced features for better ML performance
vectors = json_utils.enhanced_json_to_vector(
    lol_match, 
    dim=10000,
    feature_combinations=True,
    n_grams=3,
    sparsity_threshold=0.01
)

# Sparse representation for memory optimization
sparse_vectors = json_utils.create_sparse_vector(
    lol_match, 
    dim=10000, 
    sparsity=0.05
)

# Feature importance analysis
importance = json_utils.get_feature_importance(lol_match, dim=10000)

# Compare multiple matches
match_data = [match1, match2, match3]
enhanced_vectors = json_utils.enhanced_json_to_vector(match_data, dim=10000)
```

## Performance Considerations

- **Memory Efficiency**: Feature hashing allows handling high-dimensional sparse data
- **Generalization**: Network can handle JSON schemas not seen during training
- **Pure Python**: No compilation required, easier deployment
- **Scalability**: SQLite database supports large training datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dependencies

### Python
- tensorflow
- numpy
- requests
- python-dotenv
- plotly (for visualization)

## Troubleshooting

### API Issues
- Verify Riot API key in `.env` file
- Check rate limits for API calls
- Ensure proper region configuration in `riot_api.py`

### Runtime Issues
- Check TensorFlow version compatibility
- Ensure all dependencies are installed
- Verify environment variables are set correctly

## Testing

Run comprehensive component tests:
```bash
python test_components.py
```

This tests all components independently:
- JSON utilities
- Database functionality
- TensorFlow operations
- Plotly visualization
- NumPy operations
- Riot API integration

## Examples

See the `main.py` file for a complete working example that demonstrates:
- JSON feature training
- Loss visualization (saved to `loss_plot.html`)
- Prediction on unseen JSON structures

## Neural Network Visualization

The `plot_graph.py` module provides interactive visualization of the neural network architecture:

```bash
# Create network architecture visualization
python plot_graph.py
```

### Generated Visualizations

- **nn_architecture.html**: Complete network architecture with all layers and connections
- **Layer-specific views**: Detailed views of individual layers (input, projection, output)
- **Interactive features**: Hover information, color-coded layers, responsive design

### Visualization Features

- **Interactive Network Graph**: Shows all layers and connections
- **Layer Details**: Neuron-level detail for key layers
- **Connection Types**: Different visual styles for dense, residual, and skip connections
- **Data Flow**: Demonstrates how JSON data flows through the network
- **Educational**: Helps understand the architecture and data processing pipeline

### Integration with Training

The visualization can be integrated with actual training data to show:
- How JSON features are processed
- Data flow through each layer
- Feature dimensionality reduction
- Final classification output

## Web Interface for Match Outcome Prediction

A comprehensive Flask-based web application is available for League of Legends match outcome prediction:

### 🚀 Quick Start

1. **Install Web Dependencies**
   ```bash
   pip install -r web_requirements.txt
   ```

2. **Run the Standalone Web Server**
   ```bash
   python run.py
   ```

3. **Access the Interface**
   Open your browser and navigate to `http://localhost:5000`

### 🎯 Features

- **Real-time Prediction**: Instant match outcome predictions using enhanced JSON vectorization
- **Interactive Interface**: Beautiful, responsive web interface with real-time feedback
- **Model Training**: Train the neural network with custom match data
- **API Endpoints**: RESTful API for programmatic access
- **Professional Theme**: League of Legends-inspired dark theme

### 📊 Web Interface Components

1. **Match Explorer**: Historical match records with detailed player KDA, Gold, and Vision stats.
2. **Custom Match Predictor**: Searchable 5v5 simulator using historical player averages and champion data.
3. **Global Navigation**: Seamless switching between modules via a sticky navigation panel.
4. **Real-time Results**: Live probability bars and confidence indicators.
5. **Model Training**: Train the neural network with custom match data.

### 🔌 API Endpoints

- **GET /api/status**: Check model status and system information
- **POST /api/predict**: Predict match outcome from JSON data
- **POST /api/train**: Train model with new match data
- **GET /api/sample-match**: Get sample match structure for reference

### 🎨 User Interface Features

- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **JSON Editor**: Built-in JSON validation and formatting
- **Loading States**: Animated spinners and progress indicators
- **Error Handling**: Comprehensive error messages and validation
- **Professional Styling**: Dark theme with League of Legends-inspired colors

## Community & Governance

- **Contributing**: Contributions are welcome! Please read [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) before submitting pull requests.
- **Security Policy**: For reporting security vulnerabilities, please refer to [.github/SECURITY.md](.github/SECURITY.md).
- **Issue Templates**: Use our standard [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) or [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) templates when filing issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

