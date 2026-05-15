"""lolpros.gg proxy + import endpoints."""
import json
import logging
import threading

from flask import Blueprint, request, jsonify

import database
import data_collector

from .. import core

lolpros_bp = Blueprint("lolpros", __name__)
logger = logging.getLogger(__name__)

LOLPROS_API = "https://api.lolpros.gg"
_LOLPROS_HEADERS = {
    "Referer": "https://lolpros.gg/",
    "Origin": "https://lolpros.gg",
    "User-Agent": "Mozilla/5.0 (compatible; NNriot/1.0)",
}


@lolpros_bp.route("/api/lolpros/search", methods=["GET"])
def api_lolpros_search():
    """Proxy: search lolpros.gg pro players by name."""
    try:
        import urllib.request
        import urllib.parse

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


@lolpros_bp.route("/api/lolpros/ladder", methods=["GET"])
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


@lolpros_bp.route("/api/lolpros/import", methods=["POST"])
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
                    with core.collection_lock:
                        core.collection_status.update(
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
                    with core.collection_lock:
                        core.collection_status["status"] = "completed"
            except Exception as exc:
                logger.error(f"lolpros import collection error: {exc}", exc_info=True)
                with core.collection_lock:
                    core.collection_status["status"] = "error"
                    core.collection_status["error"] = str(exc)

        with core.collection_lock:
            already_running = core.collection_status["status"] == "collecting"

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
