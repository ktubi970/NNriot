"""Page-render routes for the NNriot web UI."""
from flask import Blueprint, render_template

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def index():
    """Serve the main web interface."""
    return render_template("index.html", active_page="index")


@views_bp.route("/explorer")
def explorer():
    """Serve the Match Explorer interface."""
    return render_template("explorer.html", active_page="explorer")


@views_bp.route("/predictor")
def predictor():
    """Serve the Custom Match Predictor interface."""
    return render_template("predictor.html", active_page="predictor")


@views_bp.route("/smurfs")
def smurfs():
    """Serve the Smurf Account Management interface."""
    return render_template("smurfs.html", active_page="smurfs")


@views_bp.route("/live")
def live():
    """Serve the Live Game Tracking interface."""
    return render_template("live.html", active_page="live")


@views_bp.route("/monitor")
def monitor():
    """Page to monitor model performance and training statistics."""
    return render_template("monitor.html", active_page="monitor")
