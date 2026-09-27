"""Keep point-in-time player history when richer priors are attached."""
from copy import deepcopy

import numpy as np
import pytest

from core.enriched_features import extract_enriched_features
from core.features.pipeline import DraftFeaturePipeline
from core.features.matchup_matrix import LaneMatchupTracker
from core.player_priors import PlayerPriorsTracker, extract_player_priors_vector


def _draft(extended=True):
    roles = ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
    champions = ('Darius', 'LeeSin', 'Ahri', 'Jinx', 'Thresh')
    teams = {}
    for team, base in (
        ('team_a', [.7, .8, 2., .6, .9, 4.]),
        ('team_b', [.4, .3, 1., .2, .5, 3.]),
    ):
        teams[team] = [
            {'champion': champion, 'role': role,
             'priors': list(base) + ([.2, .3, .4, .5] if extended else [])}
            for champion, role in zip(champions, roles)
        ]
    return teams


@pytest.mark.parametrize('container', [list, tuple])
def test_extended_priors_preserve_known_team_and_lane_values(container):
    draft = _draft()
    for team in draft.values():
        for player in team:
            player['priors'] = container(player['priors'])
    actual = extract_player_priors_vector(draft['team_a'], draft['team_b'])
    expected = [.7, .8, 2., .6, .9, 4., .4, .3, 1., .2, .5, 3.,
                .3, .5, 1., .4, .4, 1.] + [.5, 1., .4] * 5
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual.shape == (33,)


def test_attached_extended_priors_take_precedence_over_current_tracker():
    draft = _draft()
    tracker = PlayerPriorsTracker()
    actual = extract_player_priors_vector(draft['team_a'], draft['team_b'], tracker)
    np.testing.assert_allclose(actual[:6], [.7, .8, 2., .6, .9, 4.])


@pytest.mark.parametrize('extractor', [
    lambda draft: extract_enriched_features(draft, mode='dense'),
    lambda draft: extract_enriched_features(draft, mode='tabular'),
    lambda draft: DraftFeaturePipeline(matchup_tracker=LaneMatchupTracker()).extract_tabular(draft),
])
def test_extended_prefix_matches_legacy_features_and_is_order_invariant(extractor):
    draft = _draft()
    original = deepcopy(draft)
    expected = extractor(_draft(extended=False))
    np.testing.assert_array_equal(extractor(draft), expected)
    reversed_draft = {team: list(reversed(players)) for team, players in draft.items()}
    np.testing.assert_array_equal(extractor(reversed_draft), expected)
    assert draft == original


def test_extended_priors_swap_team_aggregates_and_negate_differences():
    draft = _draft()
    ab = extract_player_priors_vector(draft['team_a'], draft['team_b'])
    ba = extract_player_priors_vector(draft['team_b'], draft['team_a'])
    assert np.any(ab[12:] != 0)
    np.testing.assert_array_equal(ba[:6], ab[6:12])
    np.testing.assert_array_equal(ba[6:12], ab[:6])
    np.testing.assert_array_equal(ba[12:], -ab[12:])


def test_incomplete_priors_keep_neutral_fallback():
    draft = _draft()
    for team in draft.values():
        for player in team:
            player['priors'] = [.9, .9]
    result = extract_player_priors_vector(draft['team_a'], draft['team_b'])
    np.testing.assert_allclose(result[:6], PlayerPriorsTracker.get_default_priors())
    np.testing.assert_array_equal(result[12:], np.zeros(21))


def test_distinct_role_histories_stay_aligned_under_independent_permutations():
    draft = _draft()
    for i, (a, b) in enumerate(zip(draft['team_a'], draft['team_b'])):
        a['priors'][1:4] = [.2 + .1*i, 2. + i, .3 + .1*i]
        b['priors'][1:4] = [.1 + .02*i, 1. + .5*i, .2]
    expected_lanes = [.1, 1., .1, .18, 1.5, .2, .26, 2., .3,
                      .34, 2.5, .4, .42, 3., .5]
    ab = extract_player_priors_vector(draft['team_a'], draft['team_b'])
    np.testing.assert_allclose(ab[18:], expected_lanes, atol=1e-6)
    shuffled = {
        'team_a': [draft['team_a'][i] for i in (2, 0, 4, 1, 3)],
        'team_b': [draft['team_b'][i] for i in (1, 4, 0, 3, 2)],
    }
    actual = extract_player_priors_vector(shuffled['team_a'], shuffled['team_b'])
    np.testing.assert_allclose(actual, ab, atol=1e-6)
    swapped = extract_player_priors_vector(shuffled['team_b'], shuffled['team_a'])
    np.testing.assert_allclose(swapped[12:], -ab[12:], atol=1e-6)
    pipeline = DraftFeaturePipeline(matchup_tracker=LaneMatchupTracker())
    np.testing.assert_allclose(pipeline.extract_tabular(shuffled),
                               pipeline.extract_tabular(draft), atol=1e-6)
