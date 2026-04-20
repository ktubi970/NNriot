#!/usr/bin/env python3
"""
Neural Network Graph Visualization using Plotly
Creates an interactive HTML visualization of the NNriot neural network architecture.
"""

import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np
from typing import List, Tuple, Dict, Any


class NeuralNetworkVisualizer:
    """Creates an interactive visualization of the neural network architecture."""

    def __init__(self):
        self.layers = []
        self.connections = []
        self.node_positions = {}

    def create_network_architecture(self):
        """Define the neural network architecture for visualization."""
        # Layer definitions: (layer_name, number_of_neurons, layer_type)
        self.layers = [
            ("Input", 10000, "input"),
            ("Projection", 1024, "dense"),
            ("ResBlock1", 1024, "residual"),
            ("ResBlock2", 1024, "residual"),
            ("Dense512", 512, "dense"),
            ("Dense128", 128, "dense"),
            ("Output", 2, "output"),
        ]

        # Connection definitions: (from_layer, to_layer, connection_type)
        self.connections = [
            ("Input", "Projection", "dense"),
            ("Projection", "ResBlock1", "residual"),
            ("ResBlock1", "ResBlock2", "residual"),
            ("ResBlock2", "Dense512", "dense"),
            ("Dense512", "Dense128", "dense"),
            ("Dense128", "Output", "dense"),
            # Residual connections
            ("Projection", "ResBlock1", "skip"),
            ("ResBlock1", "ResBlock2", "skip"),
        ]

    def calculate_node_positions(self):
        """Calculate positions for all nodes in the network visualization."""
        num_layers = len(self.layers)
        layer_spacing = 200
        max_neurons = max(layer[1] for layer in self.layers)

        for i, (layer_name, num_neurons, layer_type) in enumerate(self.layers):
            x_pos = i * layer_spacing
            y_spacing = 10 if num_neurons <= 10 else 5

            # Sample neurons for visualization (show all for small layers, sample for large layers)
            if num_neurons <= 50:
                neuron_indices = range(num_neurons)
            else:
                # For large layers, show a sample
                sample_size = min(20, num_neurons)
                neuron_indices = np.linspace(0, num_neurons - 1, sample_size, dtype=int)

            layer_positions = []
            for j, neuron_idx in enumerate(neuron_indices):
                y_pos = (j - len(neuron_indices) / 2) * y_spacing
                layer_positions.append((x_pos, y_pos, neuron_idx))

            self.node_positions[layer_name] = {
                "positions": layer_positions,
                "type": layer_type,
                "total_neurons": num_neurons,
                "visible_neurons": len(neuron_indices),
            }

    def create_node_traces(self):
        """Create scatter traces for all nodes in the network."""
        node_traces = []

        for layer_name, data in self.node_positions.items():
            x_coords = [pos[0] for pos in data["positions"]]
            y_coords = [pos[1] for pos in data["positions"]]

            # Determine color based on layer type
            if data["type"] == "input":
                color = "#2E86AB"
            elif data["type"] == "output":
                color = "#F18F01"
            elif data["type"] == "residual":
                color = "#A23B72"
            else:
                color = "#C73E1D"

            # Create hover text
            hover_text = []
            for pos in data["positions"]:
                if data["total_neurons"] <= 50:
                    neuron_info = f"Neuron {pos[2]}"
                else:
                    neuron_info = f"Neuron {pos[2]} (sampled)"

                hover_text.append(
                    f"<b>{layer_name}</b><br>"
                    f"{neuron_info}<br>"
                    f"Layer Type: {data['type'].title()}<br>"
                    f"Total Neurons: {data['total_neurons']}"
                )

            node_trace = go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=6 if data["type"] in ["input", "output"] else 4,
                    color=color,
                    line=dict(width=1, color="white"),
                    opacity=0.9,
                ),
                text=hover_text,
                hoverinfo="text",
                name=layer_name,
            )
            node_traces.append(node_trace)

        return node_traces

    def create_connection_traces(self):
        """Create line traces for all connections between layers."""
        connection_traces = []

        for from_layer, to_layer, conn_type in self.connections:
            from_data = self.node_positions[from_layer]
            to_data = self.node_positions[to_layer]

            # Get positions
            from_x = [pos[0] for pos in from_data["positions"]]
            from_y = [pos[1] for pos in from_data["positions"]]
            to_x = [pos[0] for pos in to_data["positions"]]
            to_y = [pos[1] for pos in to_data["positions"]]

            # Determine line style based on connection type
            if conn_type == "dense":
                line_color = "#CCCCCC"
                line_width = 1
                line_dash = "solid"
            elif conn_type == "residual":
                line_color = "#FF6B6B"
                line_width = 2
                line_dash = "dash"
            else:  # skip connections
                line_color = "#4ECDC4"
                line_width = 1.5
                line_dash = "dot"

            # Create connection lines (sample some connections for visualization)
            num_connections = min(len(from_x), len(to_x), 50)
            connection_x = []
            connection_y = []

            for i in range(num_connections):
                # Connect corresponding neurons
                f_idx = i % len(from_x)
                t_idx = i % len(to_x)

                connection_x.extend([from_x[f_idx], to_x[t_idx], None])
                connection_y.extend([from_y[f_idx], to_y[t_idx], None])

            connection_trace = go.Scatter(
                x=connection_x,
                y=connection_y,
                mode="lines",
                line=dict(color=line_color, width=line_width, dash=line_dash),
                hoverinfo="none",
                showlegend=False,
                opacity=0.3,
            )
            connection_traces.append(connection_trace)

        return connection_traces

    def create_legend_annotations(self):
        """Create custom legend annotations for the visualization."""
        annotations = []

        legend_items = [
            ("Input Layer", "#2E86AB", "circle"),
            ("Dense Layer", "#C73E1D", "circle"),
            ("Residual Block", "#A23B72", "circle"),
            ("Output Layer", "#F18F01", "circle"),
            ("Dense Connection", "#CCCCCC", "line"),
            ("Residual Connection", "#FF6B6B", "line"),
            ("Skip Connection", "#4ECDC4", "line"),
        ]

        for i, (label, color, marker_type) in enumerate(legend_items):
            y_pos = -150 - (i * 20)

            if marker_type == "circle":
                annotations.append(
                    dict(
                        x=100,
                        y=y_pos,
                        xref="x",
                        yref="y",
                        text=f"● {label}",
                        font=dict(color=color, size=12),
                        showarrow=False,
                    )
                )
            else:
                annotations.append(
                    dict(
                        x=90,
                        y=y_pos,
                        xref="x",
                        yref="y",
                        text=f"━━ {label}",
                        font=dict(color=color, size=12),
                        showarrow=False,
                    )
                )

        return annotations

    def create_network_graph(self, save_html=True, filename="nn_architecture.html"):
        """Create and save the complete neural network visualization."""
        # Set up the architecture
        self.create_network_architecture()
        self.calculate_node_positions()

        # Create traces
        node_traces = self.create_node_traces()
        connection_traces = self.create_connection_traces()
        legend_annotations = self.create_legend_annotations()

        # Combine all traces
        all_traces = connection_traces + node_traces

        # Create layout
        layout = go.Layout(
            title={
                "text": "NNriot Neural Network Architecture",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 24, "color": "#333333"},
            },
            width=1200,
            height=800,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            annotations=legend_annotations,
            margin=dict(l=50, r=50, t=80, b=200),
        )

        # Create figure
        fig = go.Figure(data=all_traces, layout=layout)

        # Add layer labels
        for layer_name, data in self.node_positions.items():
            x_pos = data["positions"][0][0]
            y_pos = max(pos[1] for pos in data["positions"]) + 20

            layer_label = f"{layer_name}<br><sub>{data['total_neurons']} neurons</sub>"

            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                xref="x",
                yref="y",
                text=layer_label,
                showarrow=False,
                font=dict(size=14, color="#333333"),
                bgcolor="rgba(255,255,255,0.8)",
            )

        if save_html:
            # Save as HTML file
            pyo.plot(fig, filename=filename, auto_open=False)
            print(f"Neural network graph saved to {filename}")

        return fig

    def create_detailed_layer_view(self, layer_name: str):
        """Create a detailed view of a specific layer showing all neurons."""
        if layer_name not in self.node_positions:
            raise ValueError(f"Layer {layer_name} not found in network")

        data = self.node_positions[layer_name]

        # Create detailed scatter plot for this layer
        x_coords = [pos[0] for pos in data["positions"]]
        y_coords = [pos[1] for pos in data["positions"]]

        # Determine color based on layer type
        if data["type"] == "input":
            color = "#2E86AB"
        elif data["type"] == "output":
            color = "#F18F01"
        elif data["type"] == "residual":
            color = "#A23B72"
        else:
            color = "#C73E1D"

        # Create hover text for detailed view
        hover_text = [
            f"Neuron {pos[2]}<br>Position in layer: {i}"
            for i, pos in enumerate(data["positions"])
        ]

        fig = go.Figure(
            data=go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=8, color=color, line=dict(width=2, color="white"), opacity=0.8
                ),
                text=hover_text,
                hoverinfo="text",
            )
        )

        fig.update_layout(
            title=f"Detailed View: {layer_name} Layer ({data['total_neurons']} neurons)",
            width=800,
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig


def main():
    """Main function to create and display the neural network graph."""
    print("Creating Neural Network Architecture Visualization...")

    # Create visualizer
    visualizer = NeuralNetworkVisualizer()

    # Generate the main network graph
    fig = visualizer.create_network_graph(
        save_html=True, filename="nn_architecture.html"
    )

    print("✓ Main network graph created successfully")

    # Create detailed views for key layers
    key_layers = ["Input", "Projection", "Output"]

    for layer in key_layers:
        try:
            detailed_fig = visualizer.create_detailed_layer_view(layer)
            detailed_filename = f"{layer.lower()}_detailed.html"
            pyo.plot(detailed_fig, filename=detailed_filename, auto_open=False)
            print(f"✓ Detailed view for {layer} layer saved to {detailed_filename}")
        except ValueError as e:
            print(f"⚠ Could not create detailed view for {layer}: {e}")

    print("\n🎉 All visualizations created successfully!")
    print("📁 Generated files:")
    print("   - nn_architecture.html (Main network architecture)")
    print("   - input_detailed.html (Detailed input layer view)")
    print("   - projection_detailed.html (Detailed projection layer view)")
    print("   - output_detailed.html (Detailed output layer view)")

    return fig


if __name__ == "__main__":
    main()
