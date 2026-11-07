# tests/test_frozen_history.py
import numpy as np
import pytest
from core.aligned_features import extract_aligned_inputs
from core.frozen_history import FrozenHistoryProvider
from tests.aligned_dataset_fixture import source_row, target

def test_strict_end_and_global_window():
    rows = [source_row('NA1_1', 10, 100), source_row('NA1_2', 20, 30, 1)]
    query = target(source_row('NA1_q', 30, 130))
    frozen = FrozenHistoryProvider(rows, history_limit=1)
    at_end = frozen.enrich(query, include_timeline=False)
    assert at_end['priors_provenance']['history_matches'] == 0
    query['prediction_time_ms'] = 31
    later = frozen.enrich(query, include_timeline=False)
    assert later['priors_provenance']['history_matches'] == 1
    assert later['priors_provenance']['latest_result_end_ms'] == 30

def test_label_free_history_and_strict_shapes():
    old = source_row('NA1_1', 10, 20, timeline=True)
    old['winner_label'] = None
    old['labels_json'] = None
    query = target(source_row('NA1_2', 30, 130))
    for dim in (14, 18):
        enriched = FrozenHistoryProvider([old]).enrich(
            query, include_timeline=dim == 18)
        assert enriched['priors_provenance']['history_matches'] == 1
        x = extract_aligned_inputs(enriched, f'nnriot.players{dim}.v1')
        assert x['team_a'].shape == (1, 5, dim)
        assert x['macro_clash'].shape == (1, 15)
        assert all(v.dtype == np.float32 for v in x.values())

def test_creation_ties_are_not_history():
    old = source_row('NA1_1', 30, 40)
    query = target(source_row('NA1_2', 30, 140))
    assert FrozenHistoryProvider([old]).enrich(
        query, include_timeline=False)['priors_provenance']['history_matches'] == 0

def test_duplicate_sources_fail():
    row = source_row('NA1_1', 10, 20)
    with pytest.raises(ValueError, match='duplicate'):
        FrozenHistoryProvider([row, row])
