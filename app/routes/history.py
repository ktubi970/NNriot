"""History/batch collection routes."""
import threading
import logging

from flask import Blueprint, request, jsonify

import data_collector

from .. import core

history_bp = Blueprint("history", __name__)
logger = logging.getLogger(__name__)


@history_bp.route("/api/history/collect", methods=["POST"])
def api_history_collect():
    """API endpoint to trigger background collection for a specific player."""
    try:
        data = request.get_json()
        game_name = data.get("game_name")
        tag_line = data.get("tag_line")
        count = data.get("count", 10)

        if not game_name or not tag_line:
            return jsonify({"error": "game_name and tag_line required"}), 400

        with core.collection_lock:
            if core.collection_status["status"] == "collecting":
                return (
                    jsonify({"error": "A collection job is already in progress"}),
                    409,
                )

            core.collection_status.update(
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
                seeds = [(game_name, tag_line)]
                data_collector.collect_training_data(seeds, matches_per_player=count)

                with core.collection_lock:
                    core.collection_status["status"] = "completed"
            except Exception as e:
                logger.error(f"Background collection error: {e}", exc_info=True)
                with core.collection_lock:
                    core.collection_status["status"] = "error"
                    core.collection_status["error"] = str(e)

        thread = threading.Thread(target=run_collection, daemon=True)
        thread.start()

        return jsonify({"success": True, "message": "Collection started"})
    except Exception as e:
        logger.error(f"History collect error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/history/collect_batch", methods=["POST"])
def api_history_collect_batch():
    """API endpoint to trigger background collection for multiple players."""
    try:
        data = request.get_json()
        players = data.get("players", [])

        if not players:
            return jsonify({"error": "players list required"}), 400

        with core.collection_lock:
            if core.collection_status["status"] == "collecting":
                return (
                    jsonify({"error": "A collection job is already in progress"}),
                    409,
                )

            core.collection_status.update(
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

                with core.collection_lock:
                    core.collection_status["status"] = "completed"
            except Exception as e:
                logger.error(f"Background batch collection error: {e}", exc_info=True)
                with core.collection_lock:
                    core.collection_status["status"] = "error"
                    core.collection_status["error"] = str(e)

        thread = threading.Thread(target=run_collection, daemon=True)
        thread.start()

        return jsonify({"success": True, "message": "Batch collection started"})
    except Exception as e:
        logger.error(f"History batch collect error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/history/status", methods=["GET"])
def api_history_status():
    """API endpoint to check the status of the current collection job."""
    with core.collection_lock:
        return jsonify(core.collection_status)


@history_bp.route("/api/batch/collect", methods=["POST"])
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
            if not line:
                continue
            if "#" in line:
                name, tag = line.split("#", 1)
                player_list.append((name.strip(), tag.strip()))

        if not player_list:
            return jsonify({"error": "No valid players found"}), 400

        with data_collector.batch_lock:
            if data_collector.batch_status["status"] == "running":
                return jsonify({"error": "A batch job is already running"}), 409

        thread = threading.Thread(
            target=data_collector.collect_batch_with_smurfs,
            args=(player_list, sources, count),
            daemon=True,
        )
        thread.start()

        return jsonify(
            {
                "success": True,
                "message": "Batch started",
                "players_count": len(player_list),
            }
        )
    except Exception as e:
        logger.error(f"Batch collect error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/batch/status", methods=["GET"])
def api_batch_status():
    with data_collector.batch_lock:
        # Create a copy to send
        status_copy = dict(data_collector.batch_status)
        # Clear logs after sending them to save bandwidth
        data_collector.batch_status["logs"] = []
    return jsonify(status_copy)
