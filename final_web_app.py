#!/usr/bin/env python3
"""
Final Web Interface for League of Legends Match Outcome Prediction

A Flask-based web application that provides an interactive interface for
predicting League of Legends match outcomes using completely standalone
mock predictions. This version has no dependencies on TensorFlow or
any external model loading.
"""

import os

# Silence TensorFlow oneDNN performance warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import random
import re
import glob
import threading
import time

import logging
import base64

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import database
import json_utils
import live_game
import data_collector
import riot_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Absolute directory of this script — used for checkpoint discovery,
# so it works regardless of the working directory the server is started from.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
_allowed_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000").split(",")
    if o.strip()
]
CORS(app, origins=_allowed_origins)

# History Collection Background Job State
collection_lock = threading.Lock()
collection_status = {
    "status": "idle",  # idle, collecting, completed, error
    "game_name": None,
    "tag_line": None,
    "count": 0,
    "error": None,
}

# Prediction Threading Lock to prevent Keras concurrent prediction crashes
predict_lock = threading.Lock()


# Global Trainer
def init_trainer():
    try:
        from continuous_trainer import ContinuousTrainer

        trainer = ContinuousTrainer()
        logger.info("TensorFlow model loaded successfully.")
        return trainer, True
    except Exception as e:
        logger.error(f"Failed to load TF trainer: {e}", exc_info=True)
        return None, False


global_trainer, tf_available = init_trainer()
VECTOR_DIM = 20000


def _format_multi_output_response(preds: dict) -> dict:
    """
    Convert raw Keras dict-output (shape (1, k) per head) into the
    public API response shape.

    Regression heads are trained on normalized targets (see
    continuous_trainer.REGRESSION_STATS); _denorm restores them to
    original units for the public API.
    """
    # Lazy import: avoids importing TF-heavy continuous_trainer at module
    # top-level (init_trainer() already handles the TF-unavailable path).
    from continuous_trainer import REGRESSION_STATS

    def _scalar(name):
        """Extract the single scalar from a (1, 1) regression/sigmoid head."""
        return float(preds[name][0][0])

    def _denorm(name):
        """Denormalize a regression head's output back to original scale."""
        mean, std = REGRESSION_STATS[name]
        return _scalar(name) * std + mean

    def _two_class(name):
        """Extract (p_team_a, p_team_b) from a (1, 2) softmax head."""
        return float(preds[name][0][0]), float(preds[name][0][1])

    def _three_class(name):
        """Extract (p_team_a, p_team_b, p_none) from a (1, 3) softmax head."""
        return float(preds[name][0][0]), float(preds[name][0][1]), float(preds[name][0][2])

    p_a, p_b = _two_class("winner")
    pk_a, pk_b = _two_class("team_b_kill_lead")

    response = {
        "success": True,
        "winner": {
            "team_a": round(p_a, 3),
            "team_b": round(p_b, 3),
            "predicted": "B" if p_b > p_a else "A",
            "confidence": round(max(p_a, p_b), 3),
        },
        "team_b_kill_lead": {
            "team_a": round(pk_a, 3),
            "team_b": round(pk_b, 3),
            "predicted": "B" if pk_b > pk_a else "A",
        },
        "kills": {
            "total":           round(_denorm("total_kills"), 1),
            "team_a":          round(_denorm("team_a_kills"), 1),
            "team_b":          round(_denorm("team_b_kills"), 1),
            "handicap":        round(_denorm("kill_handicap"), 1),
            "odd_probability": round(_scalar("kills_odd"), 3),
        },
        "first": {
            event: {
                "team_a": round(a, 3),
                "team_b": round(b, 3),
                "none":   round(n, 3),
            }
            for event, (a, b, n) in (
                ("blood",     _three_class("first_blood")),
                ("baron",     _three_class("first_baron")),
                ("inhibitor", _three_class("first_inhibitor")),
                ("tower",     _three_class("first_tower")),
                # Timeline kills thresholds — additive keys, backward-compatible
                ("kills_5",   _three_class("first_to_5_kills")),
                ("kills_10",  _three_class("first_to_10_kills")),
                ("kills_15",  _three_class("first_to_15_kills")),
                ("kills_20",  _three_class("first_to_20_kills")),
            )
        },
        "totals": {
            "barons":  round(_denorm("total_barons"), 2),
            "dragons": round(_denorm("total_dragons"), 2),
            "towers":  round(_denorm("total_towers"), 1),
        },
        "both_teams": {
            "baron":     round(_scalar("both_baron"), 3),
            "inhibitor": round(_scalar("both_inhibitor"), 3),
            "dragon":    round(_scalar("both_dragon"), 3),
        },
        "elder_dragon": round(_scalar("elder_dragon"), 3),
    }

    return response


def final_predict_match_outcome(match_data: dict) -> dict:
    """
    Multi-output prediction. Returns a dict with named keys: winner,
    team_b_kill_lead, kills, first.*, totals.*, both_teams.*, elder_dragon.
    """
    try:
        if not tf_available or global_trainer is None:
            return {"error": "TensorFlow model is not loaded or available"}

        feature_dict = json_utils.extract_match_features(match_data)
        sparse_vec = json_utils.json_to_vector([feature_dict], dim=VECTOR_DIM)
        dense_vec = sparse_vec.toarray()

        with predict_lock:
            preds = global_trainer.predict(dense_vec)
        # preds is a dict: {head_name: ndarray(1, k)}

        return _format_multi_output_response(preds)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {"error": str(e)}


@app.route("/")
def index():
    """Serve the main web interface."""
    return render_template("index.html", active_page="index")


@app.route("/explorer")
def explorer():
    """Serve the Match Explorer interface."""
    return render_template("explorer.html", active_page="explorer")


@app.route("/predictor")
def predictor():
    """Serve the Custom Match Predictor interface."""
    return render_template("predictor.html", active_page="predictor")


@app.route("/smurfs")
def smurfs():
    """Serve the Smurf Account Management interface."""
    return render_template("smurfs.html", active_page="smurfs")


@app.route("/live")
def live():
    """Serve the Live Game Tracking interface."""
    return render_template("live.html", active_page="live")


@app.route("/api/explorer/stats", methods=["GET"])
def api_explorer_stats():
    """API endpoint to get database statistics."""
    try:
        stats = database.get_db_stats()
        # Add human readable size
        db_path = database.DB_PATH
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            stats["db_size_mb"] = round(size_mb, 2)
        else:
            stats["db_size_mb"] = 0.0

        return jsonify(stats)
    except Exception as e:
        logger.error(f"Explorer stats error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/explorer/matches", methods=["GET"])
def api_explorer_matches():
    """API endpoint to get paginated match records."""
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 20))

        # Ensure reasonable limits
        limit = max(1, min(100, limit))
        page = max(1, page)
        offset = (page - 1) * limit

        records = database.get_recent_training_records(limit=limit, offset=offset)
        total_records = database.get_training_count()

        return jsonify(
            {
                "success": True,
                "page": page,
                "limit": limit,
                "total_records": total_records,
                "total_pages": (total_records + limit - 1) // limit,
                "records": records,
            }
        )
    except Exception as e:
        logger.error(f"Explorer matches error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/explorer/match/<match_id>", methods=["GET"])
def api_explorer_match_details(match_id):
    """API endpoint to get full raw details for a specific match."""
    try:
        raw_match = database.get_raw_match(match_id)
        if not raw_match:
            return jsonify({"error": "Match not found"}), 404

        return jsonify({"success": True, "match": raw_match})
    except Exception as e:
        logger.error(f"Explorer match details error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/players/aliases", methods=["GET"])
def api_players_aliases():
    """Retrieve all linked player aliases."""
    try:
        aliases = database.get_all_aliases()
        return jsonify(aliases)
    except Exception as e:
        logger.error(f"Error fetching aliases: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/db-health", methods=["GET"])
def api_db_health():
    """Returns database health metrics for the stats strip."""
    try:
        stats = database.get_db_stats()
        return jsonify(
            {
                "players": stats.get("players", 0),
                "matches": stats.get("matches", 0),
                "training_records": stats.get("training_records", 0),
                "trained_records": stats.get("trained_records", 0),
                "untrained_records": stats.get("untrained_records", 0),
                "trained_ratio": stats.get("trained_ratio", 0),
                "success": True,
                "ok": True,
            }
        )
    except Exception as e:
        logger.error(f"Error fetching db-health: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/players/unlink", methods=["DELETE"])
def api_players_unlink():
    """Unlink an alias from its canonical account."""
    try:
        alias_puuid = request.args.get("alias_puuid")
        if not alias_puuid:
            return jsonify({"error": "alias_puuid is required"}), 400
        database.unlink_accounts(alias_puuid)
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error unlinking accounts: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/players/search", methods=["GET"])
def api_players_search():
    """Search for existing players in the local database by name."""
    try:
        q = request.args.get("q", "").strip()
        limit = request.args.get("limit", 10, type=int)
        if len(q) < 2:
            return jsonify([])

        players = database.search_players(q, limit=limit)
        return jsonify(players)
    except Exception as e:
        logger.error(f"Player search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/players/random", methods=["GET"])
def api_players_random():
    """Return a random sample of players from the database."""
    limit = request.args.get("limit", 10, type=int)
    try:
        players = database.get_top_players(
            limit=limit
        )  # Using top players as a proxy for interesting ones
        return jsonify(players)
    except Exception as e:
        logger.error(f"Random players error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/players/top", methods=["GET"])
def api_players_top():
    """API endpoint to get top players by match count."""
    limit = request.args.get("limit", 10, type=int)

    try:
        players = database.get_top_players(limit=limit)
        return jsonify(players)
    except Exception as e:
        logger.error(f"Top players error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/stream/process_frame", methods=["POST"])
def api_stream_process_frame():
    """
    Receives a base64 image from the frontend, saves it,
    and returns a mocked OCR player roster.
    """
    if os.getenv("MOCK_OCR_ENABLED", "0") != "1":
        return jsonify({"error": "OCR endpoint disabled. Set MOCK_OCR_ENABLED=1 to enable mock OCR for testing."}), 503
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # Ensure directory exists
        screenshot_dir = os.path.join(app.static_folder or "static", "screenshots")
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)

        # Save image
        filepath = os.path.join(screenshot_dir, "latest_capture.jpg")
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        # DYNAMIC OCR RESULTS (PICK 10 RANDOM FROM DB)
        try:
            db_players = database.get_top_players(limit=20)
            if len(db_players) >= 10:
                random.shuffle(db_players)
                blue_names = [p["game_name"] for p in db_players[:5]]
                red_names = [p["game_name"] for p in db_players[5:10]]
            else:
                blue_names = ["Wunder", "Skeanz", "LIDER", "Jopa", "Mikyx"]
                red_names = ["Rhilech", "Maynter", "Poby", "SamD", "Parus"]
        except Exception:
            blue_names = ["Wunder", "Skeanz", "LIDER", "Jopa", "Mikyx"]
            red_names = ["Rhilech", "Maynter", "Poby", "SamD", "Parus"]

        mock_roster = {
            "blue": blue_names,
            "blue_team": "Stream Team 1 (Blue)",
            "red": red_names,
            "red_team": "Stream Team 2 (Red)",
            "champions": {
                name: random.choice(["Gnar", "LeeSin", "Akali", "Ezreal", "Rell", "Renekton", "Sejuani", "Azir", "Varus", "Bard"])
                for name in blue_names + red_names
            },
        }

        return jsonify(
            {
                "success": True,
                "screenshot_url": "/static/screenshots/latest_capture.jpg",
                "roster": mock_roster,
            }
        )
    except Exception as e:
        logger.error(f"Stream processing error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/players/link", methods=["POST"])
def api_link_players():
    """Link a smurf/alias account to a canonical main account."""
    try:
        data = request.get_json()
        main_puuid = data.get("canonical_puuid")
        smurf_puuid = data.get("alias_puuid")
        if not main_puuid or not smurf_puuid:
            return (
                jsonify({"error": "canonical_puuid and alias_puuid are required"}),
                400,
            )

        database.link_accounts(main_puuid, smurf_puuid)
        return jsonify({"success": True, "message": "Accounts completely linked."})
    except Exception as e:
        logger.error(f"Link accounts error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/collect", methods=["POST"])
def api_history_collect():
    """API endpoint to trigger background collection for a specific player."""
    try:
        data = request.get_json()
        game_name = data.get("game_name")
        tag_line = data.get("tag_line")
        count = data.get("count", 10)

        if not game_name or not tag_line:
            return jsonify({"error": "game_name and tag_line required"}), 400

        with collection_lock:
            if collection_status["status"] == "collecting":
                return (
                    jsonify({"error": "A collection job is already in progress"}),
                    409,
                )

            collection_status.update(
                {
                    "status": "collecting",
                    "game_name": game_name,
                    "tag_line": tag_line,
                    "count": 0,
                    "error": None,
                }
            )

        def run_collection():
            try:
                # We'll use a modified version of collect_training_data or use it as is
                # Note: collect_training_data prints progress, we can't easily capture it for the UI without major changes
                # but we'll simulate completion.
                seeds = [(game_name, tag_line)]
                data_collector.collect_training_data(seeds, matches_per_player=count)

                with collection_lock:
                    collection_status["status"] = "completed"
            except Exception as e:
                logger.error(f"Background collection error: {e}", exc_info=True)
                with collection_lock:
                    collection_status["status"] = "error"
                    collection_status["error"] = str(e)

        thread = threading.Thread(target=run_collection, daemon=True)
        thread.start()

        return jsonify({"success": True, "message": "Collection started"})
    except Exception as e:
        logger.error(f"History collect error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/collect_batch", methods=["POST"])
def api_history_collect_batch():
    """API endpoint to trigger background collection for multiple players."""
    try:
        data = request.get_json()
        players = data.get("players", [])

        if not players:
            return jsonify({"error": "players list required"}), 400

        with collection_lock:
            if collection_status["status"] == "collecting":
                return (
                    jsonify({"error": "A collection job is already in progress"}),
                    409,
                )

            collection_status.update(
                {
                    "status": "collecting",
                    "game_name": f"Batch ({len(players)} players)",
                    "tag_line": "BATCH",
                    "count": 0,
                    "error": None,
                }
            )

        def run_collection():
            try:
                seeds = [
                    (p.get("game_name"), p.get("tag_line"))
                    for p in players
                    if p.get("game_name") and p.get("tag_line")
                ]
                data_collector.collect_training_data(seeds, matches_per_player=10)

                with collection_lock:
                    collection_status["status"] = "completed"
            except Exception as e:
                logger.error(f"Background batch collection error: {e}", exc_info=True)
                with collection_lock:
                    collection_status["status"] = "error"
                    collection_status["error"] = str(e)

        thread = threading.Thread(target=run_collection, daemon=True)
        thread.start()

        return jsonify({"success": True, "message": "Batch collection started"})
    except Exception as e:
        logger.error(f"History batch collect error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/status", methods=["GET"])
def api_history_status():
    """API endpoint to check the status of the current collection job."""
    with collection_lock:
        return jsonify(collection_status)


@app.route("/api/batch/collect", methods=["POST"])
def api_batch_collect():
    try:
        data = request.get_json()
        players_raw = data.get("players", "")
        sources = data.get("sources", ["lolpros"])
        count = data.get("count", 50)
        
        # Parse players into (name, tag) tuples
        player_list = []
        for line in players_raw.split("\n"):
            line = line.strip()
            if not line: continue
            if "#" in line:
                name, tag = line.split("#", 1)
                player_list.append((name.strip(), tag.strip()))
        
        if not player_list:
            return jsonify({"error": "No valid players found"}), 400

        with data_collector.batch_lock:
            if data_collector.batch_status["status"] == "running":
                return jsonify({"error": "A batch job is already running"}), 409
                
        thread = threading.Thread(target=data_collector.collect_batch_with_smurfs, args=(player_list, sources, count), daemon=True)
        thread.start()
        
        return jsonify({"success": True, "message": "Batch started", "players_count": len(player_list)})
    except Exception as e:
        logger.error(f"Batch collect error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch/status", methods=["GET"])
def api_batch_status():
    with data_collector.batch_lock:
        # Create a copy to send
        status_copy = dict(data_collector.batch_status)
        # Clear logs after sending them to save bandwidth
        data_collector.batch_status["logs"] = []
    return jsonify(status_copy)


@app.route("/api/train/manual", methods=["POST"])
def api_train_manual():
    """API endpoint to trigger a single training step manually."""
    try:
        if not tf_available or global_trainer is None:
            # Fallback for demonstration/UI testing if model isn't truly loaded in this environment
            untrained = database.get_untrained_records(limit=10)
            if untrained:
                record_ids = [r["id"] for r in untrained]
                database.mark_as_trained(record_ids)
                return jsonify(
                    {
                        "success": True,
                        "records_processed": len(untrained),
                        "message": f"Simulation: Processed {len(untrained)} new records.",
                    }
                )
            return jsonify({"error": "No untrained data available for simulation"}), 404

        # Run a real training step
        records_processed = global_trainer.run_training_step()

        return jsonify(
            {
                "success": True,
                "records_processed": records_processed,
                "message": f"Retraining step complete. Processed {records_processed} new records.",
            }
        )
    except Exception as e:
        logger.error(f"Manual train error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/custom", methods=["POST"])
def api_predict_custom():
    """API endpoint for custom 5v5 match prediction based on selected players and champions."""
    try:
        data = request.get_json()
        blue_team_data = data.get("blue_team", [])  # List of {puuid}
        red_team_data = data.get("red_team", [])  # List of {puuid}

        if len(blue_team_data) != 5 or len(red_team_data) != 5:
            return jsonify({"error": "Precisely 5 players per team required"}), 400

        blue_team_puuids = [p.get("puuid") for p in blue_team_data]
        red_team_puuids = [p.get("puuid") for p in red_team_data]

        # Warm-up: enqueue background refresh for any player whose data is stale (>7 days).
        # Non-blocking — the current prediction uses whatever data is in the DB now.
        all_puuids = blue_team_puuids + red_team_puuids
        try:
            stale_puuids = database.get_stale_puuids(all_puuids, stale_after_days=7)
            if stale_puuids:
                logger.info("Warm-up: %d/%d players have stale data — enqueueing refresh",
                            len(stale_puuids), len(all_puuids))
                # Get account info (game_name, tag_line) for each stale puuid to feed collect_by_puuid
                with database.get_connection() as conn:
                    placeholders = ",".join(["?"] * len(stale_puuids))
                    rows = conn.execute(
                        f"SELECT puuid, game_name, tag_line FROM players WHERE puuid IN ({placeholders})",
                        tuple(stale_puuids),
                    ).fetchall()
                accounts = [
                    {
                        "puuid": r["puuid"],
                        "server": r["tag_line"] or "EUW",  # tag_line ≈ server (KR1, EUW, etc.)
                        "gamename": r["game_name"] or "?",
                        "tagline": r["tag_line"] or "?",
                    }
                    for r in rows
                ]
                if accounts:
                    def _refresh_stale():
                        try:
                            result = data_collector.collect_by_puuid(accounts, matches_per_player=20)
                            logger.info("Warm-up refresh complete: %s", result)
                        except Exception:
                            logger.error("Warm-up refresh failed", exc_info=True)
                    threading.Thread(target=_refresh_stale, daemon=True).start()
        except Exception:
            logger.error("Warm-up enqueue failed (non-fatal)", exc_info=True)

        # Fetch stats for all players
        blue_stats = [database.get_player_stats(p) for p in blue_team_puuids]
        red_stats = [database.get_player_stats(p) for p in red_team_puuids]

        if not tf_available or global_trainer is None:
            return jsonify({"error": "TensorFlow model is not loaded"}), 503

        # Construct a synthetic match structure for feature extraction
        participants = []
        for i, p in enumerate(blue_team_data):
            stats = blue_stats[i]
            _avg_kda = float(stats.get("avg_kda", 0)) if stats else 0.0
            participants.append(
                {
                    "teamId": 100,
                    "championName": p.get("champion_name")
                    or p.get("championName")
                    or "Aatrox",
                    "kills": _avg_kda * 0.7,
                    "deaths": 1,
                    "assists": _avg_kda * 0.3,
                    "goldEarned": stats.get("avg_gold", 0) if stats else 0,
                    "teamPosition": p.get("role") or "UNKNOWN",
                }
            )

        for i, p in enumerate(red_team_data):
            stats = red_stats[i]
            _avg_kda = float(stats.get("avg_kda", 0)) if stats else 0.0
            participants.append(
                {
                    "teamId": 200,
                    "championName": p.get("champion_name")
                    or p.get("championName")
                    or "Aatrox",
                    "kills": _avg_kda * 0.7,
                    "deaths": 1,
                    "assists": _avg_kda * 0.3,
                    "goldEarned": stats.get("avg_gold", 0) if stats else 0,
                    "teamPosition": p.get("role") or "UNKNOWN",
                }
            )

        match_data = {
            "info": {
                "participants": participants,
                "gameMode": data.get("game_mode", "CLASSIC"),
            }
        }

        # Use centralized feature extraction
        feature_dict = json_utils.extract_match_features(match_data)

        sparse_vec = json_utils.json_to_vector([feature_dict], dim=VECTOR_DIM)
        dense_vec = sparse_vec.toarray()

        with predict_lock:
            preds = global_trainer.predict(dense_vec)

        formatted = _format_multi_output_response(preds)
        # Keep the legacy fields for the custom predictor UI
        blue_prob_nn = formatted["winner"]["team_a"]
        red_prob_nn = formatted["winner"]["team_b"]

        response_data = {
            "success": True,
            "nn_results": {
                "blue_win_probability": blue_prob_nn,
                "red_win_probability":  red_prob_nn,
            },
            "predicted_winner": "BLUE" if blue_prob_nn > red_prob_nn else "RED",
            "confidence": round(abs(blue_prob_nn - red_prob_nn) * 2, 2),
            "blue_win_probability": blue_prob_nn,
            "red_win_probability":  red_prob_nn,
            "explanation": "Multi-head NN analysis complete.",
            "llm_active": False,
            # Include the full multi-output dict for clients that want it
            "multi_output": {
                k: v for k, v in formatted.items()
                if k not in ("success", "predicted_outcome", "win_probability",
                              "lose_probability", "confidence")
            },
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Custom prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/warmup/status", methods=["GET"])
def api_warmup_status():
    """Check warm-up freshness for a list of puuids passed as ?puuids=p1,p2,..."""
    raw = request.args.get("puuids", "")
    puuids = [p.strip() for p in raw.split(",") if p.strip()]
    if not puuids:
        return jsonify({"error": "puuids query param required"}), 400
    stale = database.get_stale_puuids(puuids, stale_after_days=7)
    return jsonify({
        "puuids_checked": puuids,
        "stale_count": len(stale),
        "stale_puuids": stale,
    })


@app.route("/api/spectator/featured", methods=["GET"])
def api_spectator_featured():
    """API endpoint to get featured games for a specific region."""
    try:
        region = request.args.get("region", "KR").upper()
        # Basic validation of region
        api = riot_api.RiotAPI(region)
        games = live_game.fetch_featured_games(region)
        return jsonify({"success": True, "region": region, "games": games})
    except Exception as e:
        logger.error(f"Featured games error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/spectator/active", methods=["GET"])
def api_spectator_active():
    """API endpoint to fetch active game for a specific Riot ID."""
    try:
        name = request.args.get("name", "").strip()
        tag = request.args.get("tag", "").strip()

        if not name or not tag:
            return jsonify({"success": False, "error": "name and tag required"}), 400

        game_data = live_game.fetch_active_game_by_summoner(name, tag)
        if not game_data:
            return (
                jsonify(
                    {"success": False, "error": f"Player {name}#{tag} is not in game."}
                ),
                404,
            )

        # Enhance participants with local database info (puuid and stats)
        # We need to resolve each participant
        from data_collector import resolve_region

        region = resolve_region(tag)
        participants = game_data.get("participants", [])

        # Add PUUID info using our utility
        participants = live_game.resolve_puuid_for_participants(region, participants)

        # Check which participants are new to our database
        new_players = []
        for p in participants:
            puuid = p.get("puuid")
            if puuid:
                p["stats"] = database.get_player_stats(puuid)
                if not p["stats"]:
                    new_players.append(
                        {"name": p.get("summonerName"), "tag": tag, "puuid": puuid}
                    )

        # If there are new players, trigger background collection (Auto-Import)
        if new_players:
            logger.info(f"Triggering auto-collection for {len(new_players)} new players")

            def auto_collect_task():
                try:
                    # Use seeds approach
                    seeds = [(p["name"], p["tag"]) for p in new_players]
                    data_collector.collect_training_data(seeds, matches_per_player=10)
                except Exception as e:
                    logger.error(f"Auto-collection error: {e}", exc_info=True)

            threading.Thread(target=auto_collect_task, daemon=True).start()

        return jsonify({"success": True, "game": game_data})

    except Exception as e:
        logger.error(f"Active game search error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for match outcome prediction."""
    try:
        # Get JSON data from request
        match_data = request.get_json()

        if not match_data:
            return jsonify({"error": "No match data provided"}), 400

        # Validate required fields
        required_fields = ["participants", "teams"]
        for field in required_fields:
            if field not in match_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Make final prediction
        result = final_predict_match_outcome(match_data)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample-match", methods=["GET"])
def api_sample_match():
    """API endpoint to get a sample match structure."""
    import test_data

    return jsonify(test_data.get_sample_match())


@app.route("/api/train", methods=["POST"])
def api_train():
    """API endpoint to train the model with new data."""
    try:
        # Get training data from request
        training_data = request.get_json()

        if not training_data or "matches" not in training_data:
            return jsonify({"error": "No training data provided"}), 400

        matches = training_data["matches"]

        if not matches:
            return jsonify({"error": "No matches provided for training"}), 400

        # Setup background thread training process
        start_time = time.time()

        def background_train():
            if tf_available and global_trainer:
                # Trigger real continuous training step over the database
                try:
                    logger.info("Triggering background training...")
                    global_trainer.run_training_step()
                    logger.info("Background training completed.")
                except Exception as e:
                    logger.error(f"Background training failed: {e}", exc_info=True)
            else:
                # Simulate training process if model isn't active
                training_time = min(3.0, len(matches) * 0.05)
                time.sleep(training_time)
                logger.info("Mock background training completed.")

        thread = threading.Thread(target=background_train)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "success": True,
                "message": "Training job started in the background",
                "training_samples_queued": len(matches),
                "queued_time": round(time.time() - start_time, 2),
            }
        )

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def api_status():
    """API endpoint to check system status."""
    import tensorflow as tf

    tf_version = "N/A"
    try:
        tf_version = tf.__version__
    except Exception:
        pass

    return jsonify(
        {
            "model_loaded": tf_available and global_trainer is not None,
            "tensorflow_version": tf_version,
            "input_dimension": VECTOR_DIM if tf_available else "N/A",
            "output_classes": 2,
            "status": (
                "Model ready"
                if (tf_available and global_trainer is not None)
                else "Model unavailable"
            ),
        }
    )


@app.route("/monitor")
def monitor():
    """Page to monitor model performance and training statistics."""
    return render_template("monitor.html", active_page="monitor")


@app.route("/api/monitor/metrics", methods=["GET"])
def api_monitor_metrics():
    """API endpoint to get model performance and training metrics."""
    try:
        import os
        import glob

        metrics = {
            "database_stats": {},
            "training_records": 0,
            "players": 0,
            "matches": 0,
            "model_checkpoint": None,
            "training_config": {},
        }

        # Get database statistics using context manager
        with database.get_connection() as conn:
            # Count training records
            count_records = conn.execute(
                "SELECT COUNT(*) FROM training_dataset"
            ).fetchone()[0]
            metrics["training_records"] = count_records

            # Count unique players
            count_players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
            metrics["players"] = count_players

            # Count matches
            count_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            metrics["matches"] = count_matches

            # Get region distribution
            regions = conn.execute(
                """
                SELECT
                    SUBSTR(match_id, 1, INSTR(match_id, '_') - 1) as region,
                    COUNT(*) as count
                FROM matches
                GROUP BY region
                ORDER BY count DESC
            """
            ).fetchall()
            metrics["regions"] = {r[0]: r[1] for r in regions}

        # Check for model checkpoints
        checkpoint_files = glob.glob(os.path.join(PROJECT_DIR, "model_checkpoint-*"))
        checkpoint_versions = []
        for f in checkpoint_files:
            # Extract version number (e.g., "model_checkpoint-290.meta" -> 290)
            try:
                match = re.search(r"model_checkpoint-(\d+)", f)
                if match:
                    version = int(match.group(1))
                    if version not in checkpoint_versions:
                        checkpoint_versions.append(version)
            except Exception:
                pass

        if checkpoint_versions:
            latest_checkpoint = max(checkpoint_versions)
            metrics["model_checkpoint"] = {
                "latest_version": latest_checkpoint,
                "total_versions": len(checkpoint_versions),
                "versions": sorted(checkpoint_versions),
            }

        # Training configuration (from data_collector defaults)
        metrics["training_config"] = {
            "regions": [
                "KR",
                "NA",
                "EUW",
                "EUN",
                "BR",
                "LA1",
                "LA2",
                "OCE",
                "JP",
                "RU",
                "TR",
            ],
            "players_per_region": 500,
            "matches_per_player": 20,
            "lookback_days": 60,
            "model_type": "ResMLP",
            "batch_size": 500,
            "epochs": 100,
        }

        # Add Riot API global stats
        metrics["riot_api"] = riot_api.RiotAPI.get_global_stats()

        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Monitor metrics error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── lolpros.gg proxy endpoints ──────────────────────────────────────────────

LOLPROS_API = "https://api.lolpros.gg"
_LOLPROS_HEADERS = {
    "Referer": "https://lolpros.gg/",
    "Origin": "https://lolpros.gg",
    "User-Agent": "Mozilla/5.0 (compatible; NNriot/1.0)",
}


@app.route("/api/lolpros/search", methods=["GET"])
def api_lolpros_search():
    """Proxy: search lolpros.gg pro players by name."""
    try:
        import urllib.request, urllib.parse

        q = request.args.get("q", "").strip()
        if len(q) < 2:
            return jsonify([])
        url = f"{LOLPROS_API}/es/search/profile?query={urllib.parse.quote(q)}"
        req = urllib.request.Request(url, headers=_LOLPROS_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return jsonify(data)
    except Exception as e:
        logger.error(f"lolpros search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 502


@app.route("/api/lolpros/ladder", methods=["GET"])
def api_lolpros_ladder():
    """Proxy: top-ranked ladder from lolpros.gg."""
    try:
        import urllib.request

        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 20, type=int)
        server = request.args.get("server", "EUW")
        url = (
            f"{LOLPROS_API}/es/ladder"
            f"?page={page}&page_size={page_size}&sort=rank&order=desc"
        )
        hdrs = dict(_LOLPROS_HEADERS)
        hdrs["lpgg-server"] = server
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return jsonify(data)
    except Exception as e:
        logger.error(f"lolpros ladder error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 502


@app.route("/api/lolpros/import", methods=["POST"])
def api_lolpros_import():
    """
    Import a lolpros.gg player's accounts into the NNriot smurf system.

    Expected body:
      { "player": <lolpros player object from /es/search/profile> }

    Steps:
      1. Upsert all accounts as player stubs (PUUID + gamename + tagline).
      2. Elect the account with the highest rank score as the canonical.
      3. Link all others as aliases to the canonical.
      4. Trigger background history collection for every account.

    Returns a summary of what was done.
    """
    try:
        data = request.get_json()
        player = data.get("player")
        if not player:
            return jsonify({"error": "player object required"}), 400

        accounts = (player.get("league_player") or {}).get("accounts", [])
        if not accounts:
            return jsonify({"error": "No accounts found for this player"}), 400

        # 1. Elect canonical = highest rank score; 0 fallback for unranked
        def _score(acc):
            r = acc.get("rank") or {}
            return r.get("score", 0)

        accounts_sorted = sorted(accounts, key=_score, reverse=True)
        canonical = accounts_sorted[0]
        aliases = accounts_sorted[1:]

        canonical_puuid = canonical.get("encrypted_puuid", "")
        if not canonical_puuid:
            return jsonify({"error": "Canonical account has no PUUID"}), 400

        # 2. Upsert all players
        database.save_player(
            canonical_puuid,
            canonical.get("gamename", canonical.get("summoner_name", "?")),
            canonical.get("tagline", canonical.get("server", "?")),
        )
        for acc in aliases:
            puuid = acc.get("encrypted_puuid", "")
            if puuid:
                database.save_player(
                    puuid,
                    acc.get("gamename", acc.get("summoner_name", "?")),
                    acc.get("tagline", acc.get("server", "?")),
                )

        # 3. Link aliases
        linked = []
        for acc in aliases:
            puuid = acc.get("encrypted_puuid", "")
            if puuid and puuid != canonical_puuid:
                database.link_accounts(canonical_puuid, puuid)
                linked.append(puuid)

        # 4. Trigger background history collection for all accounts via PUUID
        accounts_for_collection = accounts_sorted  # include canonical

        def _collect_all():
            try:
                # Build account dicts expected by collect_by_puuid
                accs_payload = []
                for acc in accounts_for_collection:
                    puuid = acc.get("encrypted_puuid", "")
                    if puuid:
                        accs_payload.append(
                            {
                                "puuid": puuid,
                                "server": acc.get("server", ""),
                                "gamename": acc.get("gamename")
                                or acc.get("summoner_name", "?"),
                                "tagline": acc.get("tagline") or acc.get("server", "?"),
                            }
                        )
                if accs_payload:
                    with collection_lock:
                        collection_status.update(
                            {
                                "status": "collecting",
                                "game_name": f"lolpros:{player.get('name', '?')} ({len(accs_payload)} accs)",
                                "tag_line": "LOLPROS",
                                "count": 0,
                                "error": None,
                            }
                        )
                    result = data_collector.collect_by_puuid(
                        accs_payload, matches_per_player=20
                    )
                    logger.info(f"lolpros collection done: {result}")
                    with collection_lock:
                        collection_status["status"] = "completed"
            except Exception as exc:
                logger.error(f"lolpros import collection error: {exc}", exc_info=True)
                with collection_lock:
                    collection_status["status"] = "error"
                    collection_status["error"] = str(exc)

        with collection_lock:
            already_running = collection_status["status"] == "collecting"

        if not already_running:
            t = threading.Thread(target=_collect_all, daemon=True)
            t.start()
            collect_msg = "History collection started in background."
        else:
            collect_msg = "A collection job is already running; skipped."

        pro_name = player.get("name", "Unknown")
        canon_name = canonical.get("gamename", canonical.get("summoner_name", "?"))
        canon_tag = canonical.get("tagline", canonical.get("server", "?"))

        return jsonify(
            {
                "success": True,
                "pro_player": pro_name,
                "canonical": f"{canon_name}#{canon_tag}",
                "canonical_puuid": canonical_puuid,
                "linked_count": len(linked),
                "total_accounts": len(accounts),
                "collection": collect_msg,
            }
        )
    except Exception as e:
        logger.error(f"lolpros import error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    if not os.path.exists("templates"):
        os.makedirs("templates")

    # Run the Flask application
    # Never use debug=True in production — read from environment instead
    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 5000)),
    )
