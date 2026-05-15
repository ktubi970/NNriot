"""Main API blueprint: explorer, players, predict, train, status, monitor,
spectator, warmup, and stream OCR.

All trainer/state access goes through ``app.core`` attribute lookup so
tests can monkeypatch ``core.global_trainer`` / ``core.tf_available``
without having to reach into individual route module globals.
"""
import os
import re
import glob
import random
import time
import base64
import threading
import logging

from flask import Blueprint, request, jsonify, current_app

import database
import live_game
import data_collector
import riot_api
import json_utils
from feature_labels import VECTOR_DIM

from .. import core

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)


# ── Status / health ─────────────────────────────────────────────────────────

@api_bp.route("/api/status", methods=["GET"])
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
            "model_loaded": core.tf_available and core.global_trainer is not None,
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


# ── Explorer ────────────────────────────────────────────────────────────────

@api_bp.route("/api/explorer/stats", methods=["GET"])
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


@api_bp.route("/api/explorer/matches", methods=["GET"])
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


@api_bp.route("/api/explorer/match/<match_id>", methods=["GET"])
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


# ── Players ─────────────────────────────────────────────────────────────────

@api_bp.route("/api/players/aliases", methods=["GET"])
def api_players_aliases():
    """Retrieve all linked player aliases."""
    try:
        aliases = database.get_all_aliases()
        return jsonify(aliases)
    except Exception as e:
        logger.error(f"Error fetching aliases: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/players/unlink", methods=["DELETE"])
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


@api_bp.route("/api/players/search", methods=["GET"])
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


@api_bp.route("/api/players/random", methods=["GET"])
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


@api_bp.route("/api/players/top", methods=["GET"])
def api_players_top():
    """API endpoint to get top players by match count."""
    limit = request.args.get("limit", 10, type=int)

    try:
        players = database.get_top_players(limit=limit)
        return jsonify(players)
    except Exception as e:
        logger.error(f"Top players error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/players/link", methods=["POST"])
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
        data = request.get_json()
        if not data or "image" not in data:
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
                "screenshot_url": "/static/screenshots/latest_capture.jpg",
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
            # Fallback for demonstration/UI testing if model isn't truly loaded
            # in this environment
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
        return jsonify({"error": str(e)}), 500


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
        data = request.get_json()
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

                    threading.Thread(target=_refresh_stale, daemon=True).start()
        except Exception:
            logger.error("Warm-up enqueue failed (non-fatal)", exc_info=True)

        # Fetch stats for all players
        blue_stats = [database.get_player_stats(p) for p in blue_team_puuids]
        red_stats = [database.get_player_stats(p) for p in red_team_puuids]

        if not core.tf_available or core.global_trainer is None:
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
