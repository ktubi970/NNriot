import pytest

pytestmark = pytest.mark.unit


def test_chronological_split_is_70_15_15_without_shuffle():
    import numpy as np
    from analytics.splits import chronological_split, player_disjoint_split

    train, validation, test = chronological_split(np.arange(100))

    assert train.tolist() == list(range(70))
    assert validation.tolist() == list(range(70, 85))
    assert test.tolist() == list(range(85, 100))


def test_player_disjoint_split_has_no_shared_puuid():
    from analytics.splits import chronological_split, player_disjoint_split

    puuids = (
        frozenset({"a", "bridge"}),
        frozenset({"c"}),
        frozenset({"d"}),
        frozenset({"bridge", "z"}),
    )

    split = player_disjoint_split(puuids, min_test_matches=1)
    train_players = set().union(*(puuids[index] for index in split.train))
    test_players = set().union(*(puuids[index] for index in split.test))

    assert train_players.isdisjoint(test_players)
    assert 0 in split.discarded


def test_grouped_split_reports_underpowered_without_fallback():
    from analytics.splits import chronological_split, player_disjoint_split

    puuids = tuple(frozenset({"shared"}) for _ in range(20))

    split = player_disjoint_split(puuids, min_test_matches=5)

    assert split.available is False
    assert split.code == "INSUFFICIENT_MATCHES"
    assert len(split.train) == 0
