"""Main API blueprint: explorer, players, predict, train, status, monitor,
spectator, warmup, and stream OCR.

All trainer/state access goes through ``app.core`` attribute lookup so
tests can monkeypatch ``core.global_trainer`` / ``core.tf_available``
without having to reach into individual route module globals.
"""
import os
import re
import glob
import uuid
import random
import time
import base64
import threading
import logging

from flask import Blueprint, request, jsonify, current_app

from core import database
from scripts import live_game
from scripts import data_collector
from core import riot_api
from core import json_utils
from core.task_runner import global_task_runner
from ..services.player_service import PlayerService
from ..services.system_service import SystemService

_player_service = PlayerService()
_system_service = SystemService()
from core.feature_labels import VECTOR_DIM

from .. import core

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)


# ── Status / health ─────────────────────────────────────────────────────────

@api_bp.route("/api/status", methods=["GET"])
def api_status():
    """API endpoint to check system status."""
    import sys
    tf_mod = sys.modules.get("tensorflow")
    tf_version = getattr(tf_mod, "__version__", "N/A") if tf_mod else "N/A"

    return jsonify(
        {
            "model_loaded": bool(core.tf_available and core.global_trainer is not None),
            "tensorflow_version": tf_version,
            "input_dimension": VECTOR_DIM if core.tf_available else "N/A",
            "output_classes": 2,
            "status": (
                "Model ready"
                if (core.tf_available and core.global_trainer is not None)
                else "Model unavailable"
            ),
        }
    )


@api_bp.route("/api/db-health", methods=["GET"])
def api_db_health():
    try:
        db_path = (
            current_app.config.get("NNRIOT_DB_PATH")
            or os.getenv("NNRIOT_DB_PATH")
            or database.DB_PATH
        )
        result = _system_service.get_full_telemetry(db_path=db_path)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching db-health: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/jobs/<job_id>", methods=["GET"])
def api_get_job_status(job_id):
    status = global_task_runner.get_status(job_id)
    if not status:
        return jsonify({"success": False, "error": "Job not found"}), 404
    return jsonify({"success": True, "job": status}), 200


# ── Players ─────────────────────────────────────────────────────────────────

@api_bp.route("/api/players/aliases", methods=["GET"])
def api_players_aliases():
    try:
        result = _player_service.get_all_aliases()
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching aliases: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/players/unlink", methods=["DELETE"])
def api_players_unlink():
    try:
        alias_puuid = request.args.get("alias_puuid")
        result = _player_service.unlink_accounts(alias_puuid)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error unlinking accounts: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/players/search", methods=["GET"])
def api_players_search():
    try:
        q = (request.args.get("q") or request.args.get("query") or "").strip()
        limit = request.args.get("limit", 10, type=int)
        result = _player_service.search_players(q, limit)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Player search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/players/random", methods=["GET"])
def api_players_random():
    try:
        limit = request.args.get("limit", 10, type=int)
        result = _player_service.get_random_players(limit)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Random players error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/players/top", methods=["GET"])
def api_players_top():
    try:
        limit = request.args.get("limit", 10, type=int)
        result = _player_service.get_top_players(limit)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Top players error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/players/link", methods=["POST"])
def api_link_players():
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400
        main_puuid = data.get("canonical_puuid")
        smurf_puuid = data.get("alias_puuid")
        result = _player_service.link_accounts(main_puuid, smurf_puuid)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Link accounts error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Stream OCR ──────────────────────────────────────────────────────────────

@api_bp.route("/api/stream/process_frame", methods=["POST"])
def api_stream_process_frame():
    """
    Receives a base64 image from the frontend, saves it,
    and returns a mocked OCR player roster.
    """
    if os.getenv("MOCK_OCR_ENABLED", "0") != "1":
        return (
            jsonify(
                {
                    "error": "OCR endpoint disabled. Set MOCK_OCR_ENABLED=1 to enable mock OCR for testing."
                }
            ),
            503,
        )
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400
        if "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # Ensure directory exists
        screenshot_dir = os.path.join(
            current_app.static_folder or "static", "screenshots"
        )
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)

        # Save image with unique filename to prevent overwrite collisions
        frame_id = uuid.uuid4().hex
        filename = f"capture_{frame_id}.jpg"
        filepath = os.path.join(screenshot_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        # Retention janitor: prune oldest screenshots if exceeding threshold
        try:
            entries = [
                os.path.join(screenshot_dir, f)
                for f in os.listdir(screenshot_dir)
                if f.startswith("capture_") and f.endswith(".jpg")
            ]
            if len(entries) > 50:
                entries.sort(key=lambda p: os.path.getmtime(p))
                for old_file in entries[:-50]:
                    try:
                        os.remove(old_file)
                    except OSError:
                        pass
        except Exception:
            pass

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
            logger.error(
                "Failed to load top players for stream OCR fallback", exc_info=True
            )
            blue_names = ["Wunder", "Skeanz", "LIDER", "Jopa", "Mikyx"]
            red_names = ["Rhilech", "Maynter", "Poby", "SamD", "Parus"]

        mock_roster = {
            "blue": blue_names,
            "blue_team": "Stream Team 1 (Blue)",
            "red": red_names,
            "red_team": "Stream Team 2 (Red)",
            "champions": {
                name: random.choice(
                    [
                        "Gnar",
                        "LeeSin",
                        "Akali",
                        "Ezreal",
                        "Rell",
                        "Renekton",
                        "Sejuani",
                        "Azir",
                        "Varus",
                        "Bard",
                    ]
                )
                for name in blue_names + red_names
            },
        }

        return jsonify(
            {
                "success": True,
                "screenshot_url": f"/static/screenshots/{filename}",
                "roster": mock_roster,
            }
        )
    except Exception as e:
        logger.error(f"Stream processing error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Training ────────────────────────────────────────────────────────────────

@api_bp.route("/api/train/manual", methods=["POST"])
def api_train_manual():
    """API endpoint to trigger a single training step manually."""
    try:
        if not core.tf_available or core.global_trainer is None:
            return jsonify({
                "success": False,
                "error": "TensorFlow model is not loaded or available. Real training step cannot run.",
            }), 503

        records_processed = core.global_trainer.run_training_step()

        return jsonify(
            {
                "success": True,
                "records_processed": records_processed,
                "message": f"Retraining step complete. Processed {records_processed} new records.",
            }
        )
    except Exception as e:
        logger.error(f"Manual train error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/train", methods=["POST"])
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

        # Enforce veracity: fail fast if TF is unavailable unless ?simulate=true is requested
        if not core.tf_available or core.global_trainer is None:
            if request.args.get("simulate", "false").lower() != "true":
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Model trainer is unavailable. TensorFlow or model weights could not be loaded. Pass ?simulate=true for mock testing.",
                        }
                    ),
                    503,
                )

        # Setup background thread training process
        start_time = time.time()

        def background_train():
            if core.tf_available and core.global_trainer:
                # Trigger real continuous training step over the database
                try:
                    logger.info("Triggering background training...")
                    core.global_trainer.run_training_step()
                    logger.info("Background training completed.")
                except Exception as e:
                    logger.error(f"Background training failed: {e}", exc_info=True)
                    raise
            else:
                # Simulate training process if model isn't active
                training_time = min(3.0, len(matches) * 0.05)
                time.sleep(training_time)
                logger.info("Mock background training completed.")

        job_id = global_task_runner.submit_task("training", background_train)

        return jsonify(
            {
                "success": True,
                "message": "Training job started in the background",
                "training_samples_queued": len(matches),
                "queued_time": round(time.time() - start_time, 2),
                "job_id": job_id,
            }
        )

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Prediction ──────────────────────────────────────────────────────────────

@api_bp.route("/api/predict", methods=["POST"])
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
        result = core.final_predict_match_outcome(match_data)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/predict/custom", methods=["POST"])
def api_predict_custom():
    """API endpoint for custom 5v5 match prediction based on selected players and champions."""
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400
        blue_team_data = data.get("blue_team", [])  # List of {puuid}
        red_team_data = data.get("red_team", [])  # List of {puuid}

        if len(blue_team_data) != 5 or len(red_team_data) != 5:
            return jsonify({"error": "Precisely 5 players per team required"}), 400

        blue_team_puuids = [p.get("puuid") for p in blue_team_data]
        red_team_puuids = [p.get("puuid") for p in red_team_data]

        # Warm-up: enqueue background refresh for any player whose data is
        # stale (>7 days). Non-blocking — the current prediction uses
        # whatever data is in the DB now.
        all_puuids = blue_team_puuids + red_team_puuids
        try:
            stale_puuids = database.get_stale_puuids(all_puuids, stale_after_days=7)
            if stale_puuids:
                logger.info(
                    "Warm-up: %d/%d players have stale data — enqueueing refresh",
                    len(stale_puuids),
                    len(all_puuids),
                )
                # Get account info (game_name, tag_line) for each stale puuid
                # to feed collect_by_puuid
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
                            result = data_collector.collect_by_puuid(
                                accounts, matches_per_player=20
                            )
                            logger.info("Warm-up refresh complete: %s", result)
                        except Exception:
                            logger.error("Warm-up refresh failed", exc_info=True)
                            raise

                    global_task_runner.submit_task("warmup_refresh", _refresh_stale)
        except Exception:
            logger.error("Warm-up enqueue failed (non-fatal)", exc_info=True)

        # Fetch stats for all players
        blue_stats = [database.get_player_stats(p) for p in blue_team_puuids]
        red_stats = [database.get_player_stats(p) for p in red_team_puuids]

        if not core.tf_available or core.global_trainer is None:
            return jsonify({"error": "TensorFlow model is not loaded"}), 503

        trainer_model = getattr(core.global_trainer, "model", None)
        if (trainer_model is not None and getattr(trainer_model, "is_calibrated", True) is False) or getattr(core.global_trainer, "is_calibrated", True) is False:
            return jsonify({
                "error": "Model checkpoint is uncalibrated (untrained weights). Please train the model via 'cli.py train' before requesting predictions.",
                "uncalibrated": True,
            }), 503

        # Construct a synthetic match structure for pre-match feature extraction
        participants = []
        for i, p in enumerate(blue_team_data):
            stats = blue_stats[i]
            participants.append({
                "teamId": 100,
                "championName": p.get("champion_name") or p.get("championName") or "Aatrox",
                "teamPosition": p.get("role") or "UNKNOWN",
                "career_win_rate": float(stats.get("avg_kda", 1.0) > 1.5) if stats else 0.5,
                "career_matches": int(stats.get("total_matches", 0)) if stats else 0,
            })

        for i, p in enumerate(red_team_data):
            stats = red_stats[i]
            participants.append({
                "teamId": 200,
                "championName": p.get("champion_name") or p.get("championName") or "Aatrox",
                "teamPosition": p.get("role") or "UNKNOWN",
                "career_win_rate": float(stats.get("avg_kda", 1.0) > 1.5) if stats else 0.5,
                "career_matches": int(stats.get("total_matches", 0)) if stats else 0,
            })

        match_data = {
            "info": {
                "participants": participants,
                "gameMode": data.get("game_mode", "CLASSIC"),
            }
        }

        # Use centralized pre-match feature extraction
        feature_dict = json_utils.extract_pre_match_features(match_data)

        sparse_vec = json_utils.json_to_vector([feature_dict], dim=VECTOR_DIM)
        dense_vec = sparse_vec.toarray()

        with core.predict_lock:
            preds = core.global_trainer.predict(dense_vec)

        formatted = core._format_multi_output_response(preds)
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
            "blue_win_prob": blue_prob_nn,
            "red_win_prob": red_prob_nn,
            "explanation": "Multi-head NN analysis complete.",
            "llm_active": False,
            # Include the full multi-output dict for clients that want it
            "multi_output": {
                k: v for k, v in formatted.items()
                if k not in (
                    "success",
                    "predicted_outcome",
                    "win_probability",
                    "lose_probability",
                    "confidence",
                )
            },
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Custom prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Warmup ──────────────────────────────────────────────────────────────────

@api_bp.route("/api/warmup/status", methods=["GET"])
def api_warmup_status():
    """Check warm-up freshness for a list of puuids passed as ?puuids=p1,p2,..."""
    raw = request.args.get("puuids", "")
    puuids = [p.strip() for p in raw.split(",") if p.strip()]
    if not puuids:
        return jsonify({"error": "puuids query param required"}), 400
    stale = database.get_stale_puuids(puuids, stale_after_days=7)
    return jsonify(
        {
            "puuids_checked": puuids,
            "stale_count": len(stale),
            "stale_puuids": stale,
        }
    )


# ── Spectator ───────────────────────────────────────────────────────────────

@api_bp.route("/api/spectator/featured", methods=["GET"])
def api_spectator_featured():
    """API endpoint to get featured games for a specific region."""
    try:
        region = request.args.get("region", "KR").upper()
        # Basic validation of region
        api = riot_api.RiotAPI(region)  # noqa: F841
        games = live_game.fetch_featured_games(region)
        return jsonify({"success": True, "region": region, "games": games})
    except Exception as e:
        logger.error(f"Featured games error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/spectator/active", methods=["GET"])
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
        from scripts.data_collector import resolve_region

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
                    raise

            global_task_runner.submit_task("auto_collect", auto_collect_task)

        return jsonify({"success": True, "game": game_data})

    except Exception as e:
        logger.error(f"Active game search error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ── Misc ────────────────────────────────────────────────────────────────────

@api_bp.route("/api/sample-match", methods=["GET"])
def api_sample_match():
    """API endpoint to get a sample match structure."""
    import test_data

    return jsonify(test_data.get_sample_match())


@api_bp.route("/api/monitor/metrics", methods=["GET"])
def api_monitor_metrics():
    """API endpoint to get model performance and training metrics."""
    try:
        db_path = (
            current_app.config.get("NNRIOT_DB_PATH")
            or os.getenv("NNRIOT_DB_PATH")
            or database.DB_PATH
        )
        telemetry = _system_service.get_full_telemetry(db_path=db_path)
        regions = _system_service.get_region_distribution(db_path=db_path)

        metrics = dict(telemetry)
        metrics["database_stats"] = dict(telemetry)
        metrics["regions"] = regions

        # Check for model checkpoints
        checkpoint_files = glob.glob(
            os.path.join(core.PROJECT_DIR, "model_checkpoint-*")
        )
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

        # Training configuration — regions/lookback come from data_collector
        # so the UI tracks the collector defaults. batch_size/epochs are
        # hardcoded here to avoid importing TF-heavy continuous_trainer; if
        # the trainer constants change, bump these to match (continuous_trainer
        # BATCH_SIZE / EPOCHS_PER_BATCH).
        metrics["training_config"] = {
            "regions": data_collector.DEFAULT_REGIONS,
            "players_per_region": data_collector.DEFAULT_PLAYERS_PER_REGION,
            "matches_per_player": data_collector.DEFAULT_MATCHES_PER_PLAYER,
            "lookback_days": data_collector.DEFAULT_LOOKBACK_DAYS,
            "model_type": "ResMLP-22",  # 22-head multi-output model
            "batch_size": 500,
            "epochs": 100,
        }

        # Add Riot API global stats
        metrics["riot_api"] = riot_api.RiotAPI.get_global_stats()

        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Monitor metrics error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
