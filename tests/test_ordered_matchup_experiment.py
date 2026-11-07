"""Behavioral checks for the isolated result-ordered matchup experiment."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / 'docs/research-improvement/ordered_matchup_features.py'
_SPEC = importlib.util.spec_from_file_location('ordered_matchup_features', _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
extract_ordered_features = _MODULE.extract_ordered_features


def _row(match_id, start, winner=0):
    roles = ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
    a = ('Darius', 'LeeSin', 'Ahri', 'Jinx', 'Thresh')
    b = ('Sion', 'Amumu', 'Syndra', 'Caitlyn', 'Leona')
    return {
        'match_id': match_id,
        'game_creation_time': start,
        'labels_json': {'winner': winner},
        'feature_json': {
            'team_a': [{'champion': c, 'role': r} for c, r in zip(a, roles)],
            'team_b': [{'champion': c, 'role': r} for c, r in zip(b, roles)],
        },
    }


def _splits():
    return {
        'train': [_row('first', 100), _row('overlap', 110), _row('later', 200)],
        'validation': [_row('val', 250)],
        'test': [_row('test', 300)],
    }


def _assert_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    else:
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('representation', ['tabular', 'tensors', 'v3'])
def test_own_and_future_labels_do_not_change_earlier_features(representation):
    splits = _splits()
    ends = {'first': 150, 'overlap': 180, 'later': 220}
    before = extract_ordered_features(splits, ends, representation=representation)[0]
    changed = deepcopy(splits)
    changed['train'][2]['labels_json']['winner'] = 1
    after = extract_ordered_features(changed, ends, representation=representation)[0]
    _assert_equal(before['train'], after['train'])
    if representation == 'tabular':
        assert not np.array_equal(before['validation'], after['validation'])
    else:
        assert not np.array_equal(before['validation']['team_a'], after['validation']['team_a'])


def test_overlap_and_equal_end_are_unavailable_but_known_past_contributes():
    splits = {'train': [_row('first', 100), _row('overlap', 110)],
              'validation': [_row('at_end', 150), _row('after_end', 151)]}
    features, _, diag = extract_ordered_features(
        splits, {'first': 150, 'overlap': 190}, representation='tensors')
    np.testing.assert_array_equal(features['train']['team_a'][:, :, 12], 0.)
    np.testing.assert_array_equal(features['validation']['team_a'][0, :, 12], 0.)
    np.testing.assert_allclose(features['validation']['team_a'][1, :, 12], .03125)
    assert diag['train_results_applied'] == 1
    assert diag['queries_without_history'] == 3


def test_changing_known_past_result_changes_future_matchup_direction():
    splits = {'train': [_row('past', 100)], 'test': [_row('query', 200)]}
    positive = extract_ordered_features(splits, {'past': 150}, representation='tensors')[0]
    splits['train'][0]['labels_json']['winner'] = 1
    negative = extract_ordered_features(splits, {'past': 150}, representation='tensors')[0]
    np.testing.assert_allclose(positive['test']['team_a'][0, :, 12], .03125)
    np.testing.assert_allclose(negative['test']['team_a'][0, :, 12], -.03125)


def test_holdout_results_never_enter_history():
    splits = _splits()
    ends = {'first': 150, 'overlap': 180, 'later': 220, 'val': 260, 'test': 310}
    before = extract_ordered_features(splits, ends)
    for name in ('validation', 'test'):
        splits[name][0]['labels_json'] = {}  # No holdout label is needed for features.
    after = extract_ordered_features(splits, ends)
    _assert_equal(before[0], after[0])
    _assert_equal(before[1], after[1])
    assert after[2]['train_results_total'] == 3


def test_unknown_and_invalid_result_ends_are_excluded():
    bad_ends = [None, True, 0, -5, 100, 99, 150., '150']
    rows = [_row('missing', 100)] + [_row(f'bad_{i}', 100) for i in range(len(bad_ends))]
    splits = {'train': rows, 'test': [_row('query', 1000)]}
    ends = {f'bad_{i}': end for i, end in enumerate(bad_ends)}
    features, _, diag = extract_ordered_features(splits, ends, representation='tensors')
    np.testing.assert_array_equal(features['test']['team_a'][:, :, 12], 0.)
    assert diag['train_results_missing_end'] == 1
    assert diag['train_results_invalid_end'] == len(bad_ends)
    assert diag['train_results_eligible'] == 0


@pytest.mark.parametrize('representation', ['tabular', 'tensors', 'v3'])
def test_unsorted_inputs_preserve_output_alignment_and_are_not_mutated(representation):
    splits = _splits()
    splits['train'] = [splits['train'][2], splits['train'][0], splits['train'][1]]
    original = deepcopy(splits)
    ends = {'later': 220, 'overlap': 180, 'first': 150}
    before = deepcopy(ends)
    actual, swapped, _ = extract_ordered_features(splits, ends, representation=representation)
    ordered = _splits()
    expected, expected_swap, _ = extract_ordered_features(ordered, ends, representation=representation)
    for result, reference in ((actual, expected), (swapped, expected_swap)):
        if representation == 'tabular':
            np.testing.assert_array_equal(result['train'], reference['train'][[2, 0, 1]])
        else:
            for key in result['train']:
                np.testing.assert_array_equal(result['train'][key], reference['train'][key][[2, 0, 1]])
    assert splits == original
    assert ends == before


def test_repeated_extraction_uses_fresh_history():
    splits = _splits()
    ends = {'first': 150, 'overlap': 180, 'later': 220}
    first = extract_ordered_features(splits, ends)
    second = extract_ordered_features(splits, ends)
    _assert_equal(first[0], second[0])
    _assert_equal(first[1], second[1])
    assert first[2] == second[2]


@pytest.mark.parametrize('representation', ['tabular', 'tensors', 'v3'])
def test_swaps_share_history_and_player_permutations_preserve_features(representation):
    splits = _splits()
    ends = {'first': 150, 'overlap': 180, 'later': 220}
    forward, swapped, _ = extract_ordered_features(splits, ends, representation=representation)
    for rows in splits.values():
        for row in rows:
            for team in row['feature_json'].values():
                team.reverse()
    permuted = extract_ordered_features(splits, ends, representation=representation)
    _assert_equal(forward, permuted[0])
    _assert_equal(swapped, permuted[1])
    for name in splits:
        if representation == 'tabular':
            np.testing.assert_array_equal(swapped[name][:, :5], forward[name][:, 5:10])
            np.testing.assert_array_equal(swapped[name][:, 5:10], forward[name][:, :5])
            np.testing.assert_allclose(swapped[name][:, 213:218], -forward[name][:, 213:218])
        else:
            np.testing.assert_array_equal(swapped[name]['team_a'], forward[name]['team_b'])
            np.testing.assert_array_equal(swapped[name]['team_b'], forward[name]['team_a'])
            if representation == 'v3':
                np.testing.assert_array_equal(swapped[name]['macro_clash'], -forward[name]['macro_clash'])


@pytest.mark.parametrize('representation', ['tabular', 'tensors', 'v3'])
def test_dimensions_and_empty_splits(representation):
    outputs = extract_ordered_features({'train': [_row('first', 100)], 'test': []}, {},
                                      representation=representation)
    for features in outputs[:2]:
        for name, n in [('train', 1), ('test', 0)]:
            if representation == 'tabular':
                assert features[name].shape == (n, 232)
                assert features[name].dtype == np.float32
            else:
                assert features[name]['team_a'].shape == (n, 5, 14)
                assert features[name]['team_b'].shape == (n, 5, 14)
                if representation == 'v3':
                    assert features[name]['macro_clash'].shape == (n, 15)


@pytest.mark.parametrize('timestamp', [None, True, 0, -1, 100., '100'])
def test_invalid_query_timestamp_is_rejected(timestamp):
    with pytest.raises(ValueError, match='timestamp'):
        extract_ordered_features({'train': [_row('first', timestamp)]}, {})


def test_duplicate_match_ids_across_splits_are_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        extract_ordered_features({'train': [_row('same', 100)], 'test': [_row('same', 200)]}, {})


def test_invalid_representation_is_rejected():
    with pytest.raises(ValueError, match='representation'):
        extract_ordered_features({'train': []}, {}, representation='other')
