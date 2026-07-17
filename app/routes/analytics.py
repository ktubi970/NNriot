"""Additive JSON API for evidence-backed analytics chat."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from analytics.service import AnalyticsService


analytics_bp = Blueprint("analytics", __name__)
MAX_CHAT_MESSAGE_LENGTH = 500
MAX_CHAT_REQUEST_BYTES = 65_536


def get_analytics_service() -> AnalyticsService:
    service = current_app.extensions.get("analytics_service")
    if service is None:
        service = AnalyticsService(
            current_app.config["ANALYTICS_DB_PATH"],
            current_app.config["NNRIOT_DB_PATH"],
        )
        current_app.extensions["analytics_service"] = service
    return service


@analytics_bp.post("/api/analytics/chat")
def analytics_chat():
    if (
        request.content_length is not None
        and request.content_length > MAX_CHAT_REQUEST_BYTES
    ):
        return jsonify({"success": False, "code": "REQUEST_TOO_LARGE"}), 413
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), str):
        return jsonify({"success": False, "code": "INVALID_REQUEST"}), 400
    if len(payload["message"]) > MAX_CHAT_MESSAGE_LENGTH:
        return jsonify({"success": False, "code": "REQUEST_TOO_LARGE"}), 413
    context = payload.get("context")
    filters = payload.get("filters", {})
    if (context is not None and not isinstance(context, dict)) or not isinstance(
        filters, dict
    ):
        return jsonify({"success": False, "code": "INVALID_REQUEST"}), 400
    try:
        response = get_analytics_service().chat(
            payload["message"], context, filters
        )
        return jsonify(response.to_dict())
    except ValueError as exc:
        code = str(exc).split(":", 1)[0]
        return jsonify({"success": False, "code": code}), 400


@analytics_bp.get("/api/analytics/health")
def analytics_health():
    health = get_analytics_service().health()
    return jsonify(health), 200 if health.get("ok") else 503


@analytics_bp.get("/api/analytics/evidence/<evidence_id>")
def analytics_evidence(evidence_id: str):
    packet = get_analytics_service().get_evidence(evidence_id)
    if packet is None:
        return jsonify({"success": False, "code": "EVIDENCE_NOT_FOUND"}), 404
    return jsonify({"success": True, "evidence": packet})
