"""Page-render routes for the NNriot web UI with React SPA support."""
import os
from flask import Blueprint, render_template, send_from_directory, current_app

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def index():
    """Serve the modern React SPA Cockpit if built and not testing, else fallback to Jinja2 template."""
    if not current_app.config.get("TESTING"):
        dist_dir = os.path.join(current_app.static_folder, "dist")
        spa_index = os.path.join(dist_dir, "index.html")
        if os.path.exists(spa_index):
            return send_from_directory(dist_dir, "index.html")
    return render_template("index.html", active_page="index")


@views_bp.route("/predictor")
def predictor():
    return render_template("predictor.html", active_page="predictor")


@views_bp.route("/explorer")
def explorer():
    return render_template("explorer.html", active_page="explorer")


@views_bp.route("/monitor")
def monitor():
    return render_template("monitor.html", active_page="monitor")


@views_bp.route("/live")
def live():
    return render_template("live.html", active_page="live")


@views_bp.route("/smurfs")
def smurfs():
    return render_template("smurfs.html", active_page="smurfs")


@views_bp.route("/analytics")
def analytics():
    return render_template("analytics.html", active_page="analytics")


@views_bp.route("/assets/<path:path>")
def serve_spa_assets(path):
    """Serve compiled Vite assets."""
    dist_assets = os.path.join(current_app.static_folder, "dist", "assets")
    if os.path.exists(dist_assets):
        return send_from_directory(dist_assets, path)
    return "", 404

