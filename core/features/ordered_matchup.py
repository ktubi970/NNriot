"""Extract matchup features from completed train results only.

Recorded game end is an optimistic proxy for result availability: ingestion
latency is unobserved. This helper only controls the matchup tracker. It does
not reconstruct or certify the temporal validity of attached player priors.
The caller supplies split records and explicit result end timestamps.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

from core.features.draft_composition import MACRO_CLASH_DIM
from core.features.matchup_matrix import LaneMatchupTracker
from core.features.pipeline import (
    DraftFeaturePipeline,
    ENRICHED_PLAYER_TENSOR_DIM,
    ENRICHED_TABULAR_DIM,
)


def _positive_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value > 0


def _allocate(count: int, representation: str):
    if representation == 'tabular':
        return np.empty((count, ENRICHED_TABULAR_DIM), dtype=np.float32)
    result = {
        team: np.empty((count, 5, ENRICHED_PLAYER_TENSOR_DIM), dtype=np.float32)
        for team in ('team_a', 'team_b')
    }
    if representation == 'v3':
        result['macro_clash'] = np.empty((count, MACRO_CLASH_DIM), dtype=np.float32)
    return result


def _extract_into(pipeline, draft, destination, index, representation):
    if representation == 'tabular':
        destination[index] = pipeline.extract_tabular(draft)
        return
    team_a, team_b = pipeline.extract_tensors(draft)
    destination['team_a'][index] = team_a
    destination['team_b'][index] = team_b
    if representation == 'v3':
        destination['macro_clash'][index] = pipeline.extract_macro_clash(draft)


def extract_ordered_features(
    splits: Mapping[str, list[dict]],
    result_end_times: Mapping[str, int],
    *,
    representation: str = 'tabular',
    matchup_weight: float = 15.0,
) -> tuple[dict, dict, dict]:
    """Return forward features, swapped features, and extraction diagnostics.

    For a query at ``game_creation_time``, update a fresh matchup tracker only
    with train results whose recorded end is strictly earlier than the query.
    A valid end must also be strictly later than its own match creation. Ends
    and query timestamps use the same units. Missing or invalid ends exclude
    that result permanently; there is no synthetic chronological fallback.

    Output rows retain their input order. Forward and swapped drafts use the
    same tracker state. Validation/test labels are neither read nor ingested.
    The caller retains ownership of the input dictionaries and lists.
    """
    if representation not in ('tabular', 'tensors', 'v3'):
        raise ValueError('representation must be tabular, tensors, or v3')

    diagnostics = {
        'train_results_total': 0,
        'train_results_eligible': 0,
        'train_results_missing_end': 0,
        'train_results_invalid_end': 0,
        'train_results_applied': 0,
        'queries_total': 0,
        'queries_without_history': 0,
        'availability_assumption': (
            'recorded_game_end_strictly_before_creation; ingestion_latency_unobserved'
        ),
        'scope': 'matchup_tracker_only; attached_player_priors_not_reconstructed',
        'representation': representation,
        'matchup_weight': float(matchup_weight),
    }
    queries = []
    events = []
    seen = set()
    for split_name, rows in splits.items():
        for index, row in enumerate(rows):
            match_id = row.get('match_id')
            if not isinstance(match_id, str) or not match_id.strip():
                raise ValueError('every row must have a nonempty match_id')
            if match_id in seen:
                raise ValueError(f'duplicate match_id: {match_id}')
            seen.add(match_id)
            query_time = row.get('game_creation_time')
            if not _positive_integer(query_time):
                raise ValueError(f'invalid query timestamp for {match_id}')
            draft = row.get('feature_json')
            if not isinstance(draft, dict):
                raise ValueError(f'feature_json must be a dictionary for {match_id}')
            queries.append((query_time, split_name, index, draft))
            if split_name != 'train':
                continue
            diagnostics['train_results_total'] += 1
            if match_id not in result_end_times:
                diagnostics['train_results_missing_end'] += 1
                continue
            end = result_end_times[match_id]
            if not _positive_integer(end) or end <= query_time:
                diagnostics['train_results_invalid_end'] += 1
                continue
            winner = row['labels_json']['winner']
            if isinstance(winner, (bool, np.bool_)) or not isinstance(winner, Integral) or winner not in (0, 1):
                raise ValueError(f'invalid train winner for {match_id}')
            events.append((end, match_id, draft, int(winner)))

    queries.sort(key=lambda item: (item[0], item[1], item[2]))
    events.sort(key=lambda item: (item[0], item[1]))
    diagnostics['queries_total'] = len(queries)
    diagnostics['train_results_eligible'] = len(events)
    features = {name: _allocate(len(rows), representation) for name, rows in splits.items()}
    swapped = {name: _allocate(len(rows), representation) for name, rows in splits.items()}
    tracker = LaneMatchupTracker(matchup_weight=matchup_weight)
    pipeline = DraftFeaturePipeline(matchup_tracker=tracker)
    cursor = 0
    for query_time, split_name, index, draft in queries:
        while cursor < len(events) and events[cursor][0] < query_time:
            _, _, completed_draft, winner = events[cursor]
            tracker.update_match(
                completed_draft.get('team_a') or [],
                completed_draft.get('team_b') or [],
                winner=winner,
            )
            cursor += 1
        if cursor == 0:
            diagnostics['queries_without_history'] += 1
        _extract_into(pipeline, draft, features[split_name], index, representation)
        reversed_draft = {
            **draft,
            'team_a': draft.get('team_b') or [],
            'team_b': draft.get('team_a') or [],
        }
        _extract_into(pipeline, reversed_draft, swapped[split_name], index, representation)
    diagnostics['train_results_applied'] = cursor
    return features, swapped, diagnostics
