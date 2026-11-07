"""Serving-side guards composed around the hash-pinned feature modules.

``core/serving_contract.py`` and ``core/pre_match_history.py`` are SHA-256
bound by model manifests and frozen datasets, so serving checks that are not
part of the feature semantics live here instead.
"""
from __future__ import annotations

from core.serving_contract import ContractError


class ModelOutputError(ContractError):
    """The loaded model produced output that violates its declared heads."""


class ServiceLoading(RuntimeError):
    """Another request is currently loading the prediction facade."""


def validate_participant_teams(match) -> None:
    """Riot-style participants must declare teamId 100 or 200.

    ``extract_pre_match_features`` falls back to ``team_id`` and to team 100,
    while ``to_serving_features`` aligns identities on ``teamId`` only; any
    other shape would shift puuids onto the wrong champions.
    """
    if not isinstance(match, dict):
        return
    info = match.get('info', match)
    if not isinstance(info, dict) or 'participants' not in info:
        return
    participants = info['participants']
    if not isinstance(participants, list) or any(
        not isinstance(p, dict) or type(p.get('teamId')) is not int or p['teamId'] not in (100, 200)
        for p in participants
    ):
        raise ContractError('Each participant requires an integer teamId of 100 or 200')
