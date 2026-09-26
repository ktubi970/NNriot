"""History/batch collection routes."""
import os
import threading
import logging

from flask import Blueprint, request, jsonify, current_app

from scripts import data_collector
from core import database
from core.task_runner import global_task_runner
from ..services.system_service import SystemService

def _system_service():
    from core.repository import SystemRepository
    return SystemService(SystemRepository(db_path=current_app.config["NNRIOT_DB_PATH"]))

from .. import core

history_bp = Blueprint("history", __name__)
logger = logging.getLogger(__name__)


@history_bp.route("/api/history/collect", methods=["POST"])
def api_history_collect():
    """API endpoint to trigger background collection for a specific player."""
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400
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

        db_path = current_app.config["NNRIOT_DB_PATH"]

        def run_collection():
            try:
                seeds = [(game_name, tag_line)]
                data_collector.collect_training_data(seeds, matches_per_player=count, db_path=db_path)

                with core.collection_lock:
                    core.collection_status["status"] = "completed"
            except Exception as e:
                logger.error(f"Background collection error: {e}", exc_info=True)
                with core.collection_lock:
                    core.collection_status["status"] = "error"
                    core.collection_status["error"] = str(e)
                raise

        job_id = global_task_runner.submit_task("history_collect", run_collection)

        return jsonify({"success": True, "message": "Collection started", "job_id": job_id})
    except Exception as e:
        logger.error(f"History collect error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/history/collect_batch", methods=["POST"])
def api_history_collect_batch():
    """API endpoint to trigger background collection for multiple players."""
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400
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

        db_path = current_app.config["NNRIOT_DB_PATH"]

        def run_collection():
            try:
                seeds = [
                    (p.get("game_name"), p.get("tag_line"))
                    for p in players
                    if p.get("game_name") and p.get("tag_line")
                ]
                data_collector.collect_training_data(seeds, matches_per_player=10, db_path=db_path)

                with core.collection_lock:
                    core.collection_status["status"] = "completed"
            except Exception as e:
                logger.error(f"Background batch collection error: {e}", exc_info=True)
                with core.collection_lock:
                    core.collection_status["status"] = "error"
                    core.collection_status["error"] = str(e)
                raise

        job_id = global_task_runner.submit_task("history_collect_batch", run_collection)

        return jsonify({"success": True, "message": "Batch collection started", "job_id": job_id})
    except Exception as e:
        logger.error(f"History batch collect error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/history/status", methods=["GET"])
def api_history_status():
    """API endpoint to check the status of the current collection job."""
    with core.collection_lock:
        status_copy = dict(core.collection_status)
    candidates = [
        j for j in [
            global_task_runner.get_latest_by_type("history_collect"),
            global_task_runner.get_latest_by_type("history_collect_batch"),
            global_task_runner.get_latest_by_type("lolpros_import"),
        ] if j is not None
    ]
    latest_job = max(candidates, key=lambda x: x["created_at"]) if candidates else None
    if latest_job and "job_id" not in status_copy:
        status_copy["job_id"] = latest_job["job_id"]
    return jsonify(status_copy)


@history_bp.route("/api/batch/collect", methods=["POST"])
def api_batch_collect():
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400
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

        job_id = global_task_runner.submit_task(
            "batch_collect",
            data_collector.collect_batch_with_smurfs,
            player_list,
            sources,
            count,
            db_path=current_app.config["NNRIOT_DB_PATH"],
        )

        return jsonify(
            {
                "success": True,
                "message": "Batch started",
                "players_count": len(player_list),
                "job_id": job_id,
            }
        )
    except Exception as e:
        logger.error(f"Batch collect error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/batch/status", methods=["GET"])
def api_batch_status():
    with data_collector.batch_lock:
        # Create a copy to send with a bounded window of logs
        status_copy = dict(data_collector.batch_status)
        status_copy["logs"] = list(data_collector.batch_status.get("logs", []))[-100:]
    latest_job = global_task_runner.get_latest_by_type("batch_collect")
    if latest_job and "job_id" not in status_copy:
        status_copy["job_id"] = latest_job["job_id"]
    return jsonify(status_copy)


@history_bp.route("/api/db/stats", methods=["GET"])
def api_db_stats():
    """Returns database telemetry and match ingestion counts."""
    try:
        db_path = current_app.config["NNRIOT_DB_PATH"]
        telemetry = _system_service().get_full_telemetry(db_path=db_path)
        riot_key = os.getenv("RIOT_API_KEY")

        response = dict(telemetry)
        response.update({
            "total_matches": telemetry.get("matches", 0),
            "total_training": telemetry.get("training_records", 0),
            "total_players": telemetry.get("players", 0),
            "crawl_queue_pending": telemetry.get("crawl_queue_pending", 0),
            "db_size_mb": telemetry.get("db_size_mb", 0.0),
            "riot_api_configured": bool(riot_key and len(riot_key) > 10),
        })
        return jsonify(response)
    except Exception as e:
        logger.error(f"DB stats error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/player/history", methods=["GET"])
def api_player_history():
    """Fetch rich match history for a player with optional live Riot API synchronization."""
    from core import riot_api

    name = request.args.get("name", "Faker").strip()
    tag = request.args.get("tag", "KR1").strip()
    count = request.args.get("count", 10, type=int)
    sync_riot = request.args.get("sync_riot", "0") == "1"

    db_path = current_app.config["NNRIOT_DB_PATH"]
    riot_key = os.getenv("RIOT_API_KEY")

    # 1. Live Riot API Fetch if requested and key is present
    if sync_riot and riot_key:
        try:
            region = data_collector.resolve_region(tag)
            with riot_api.RiotAPI(region) as api:
                puuid = api.get_puuid(name, tag)
                if puuid:
                    database.save_player(puuid, name, tag, db_path=db_path)
                    match_ids = api.get_match_ids(puuid, count=count)
                    new_mids = [
                        mid
                        for mid in match_ids
                        if not database.get_raw_match(mid, db_path=db_path)
                    ]
                    if new_mids:
                        details_map = api.get_match_details_batch(new_mids, max_workers=3)
                        matches_batch = []
                        training_batch = []
                        for mid, det in details_map.items():
                            matches_batch.append((mid, det))
                            rec = data_collector._build_training_record(
                                mid, det, region=region
                            )
                            if rec:
                                training_batch.append(rec)
                        if matches_batch:
                            database.save_matches_batch(matches_batch, db_path=db_path)
                        if training_batch:
                            database.save_training_records_batch(
                                training_batch, db_path=db_path
                            )
        except riot_api.RiotAuthError as e:
            logger.warning(f"Live Riot sync auth failed: {e}")
            return jsonify({
                "success": False,
                "error": "Riot API authentication failed (API key expired or invalid)",
                "details": str(e),
            }), 503
        except Exception as e:
            logger.warning(f"Live Riot sync failed: {e}", exc_info=True)

    # 2. Query real player and match history from DB
    target_player = database.get_player_by_name_tag(name, tag, db_path=db_path)
    if not target_player:
        player_rows = database.search_players(name, limit=5, db_path=db_path)
        target_player = next(
            (p for p in player_rows if p.get("tag_line", "").upper() == tag.upper()),
            None,
        )
        if not target_player and player_rows:
            target_player = next(
                (
                    p
                    for p in player_rows
                    if p.get("game_name", "").lower() == name.lower()
                ),
                None,
            )

    puuid = target_player["puuid"] if target_player else None

    matches = []
    stats = None
    if puuid:
        matches = database.get_player_match_history(puuid, limit=count, db_path=db_path)
        stats = database.get_player_stats(puuid, db_path=db_path)

    wins_count = sum(1 for m in matches if m["win"])
    total_games = (
        len(matches) if not stats else stats.get("total_matches", len(matches))
    )
    winrate = (
        round((wins_count / max(1, len(matches))) * 100, 1) if matches else 0.0
    )
    avg_kda = (
        str(round(stats.get("avg_kda", 0.0), 2))
        if stats
        else (
            "0.0"
            if not matches
            else str(round(sum(float(m["kda"]) for m in matches) / len(matches), 2))
        )
    )

    return jsonify(
        {
            "success": True,
            "player": {
                "name": name,
                "tag": tag,
                "puuid": puuid,
                "winrate": winrate,
                "total_games": total_games,
                "avg_kda": avg_kda,
                "has_data": len(matches) > 0,
            },
            "matches": matches,
        }
    )
