#!/usr/bin/env python3
"""Entry point for the NNriot Flask web app."""
import os

from app import create_app

app = create_app()


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 5000)),
    )
