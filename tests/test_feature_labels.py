import os
import sys

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_labels import LABEL_KEYS, extract_labels


def _make_match(
    team_a_win=True,
    team_b_win=False,
    team_a_kills=20,
    team_b_kills=10,
    objectives_a=None,
    objectives_b=None,
    elder_a=None,
    elder_b=None,
    skip_team_a=False,
    skip_team_b=False,
):
    """Build a minimal Riot Match-V5 dict for testing."""
    participants = []
    if not skip_team_a:
        participants.append({"teamId": 100, "kills": team_a_kills})
    if not skip_team_b:
        participants.append({"teamId": 200, "kills": team_b_kills})

    def _team(team_id, win, objectives, elder):
        # Make a shallow copy so callers can reuse the same dict across tests.
        obj = dict(objectives or {})
        if elder is not None:
            obj["elderDragon"] = {"kills": elder}
        return {"teamId": team_id, "win": win, "objectives": obj}

    teams = []
    if not skip_team_a:
        teams.append(_team(100, team_a_win, objectives_a, elder_a))
    if not skip_team_b:
        teams.append(_team(200, team_b_win, objectives_b, elder_b))

    return {"info": {"participants": participants, "teams": teams}}


def _all_objectives_for(team_letter):
    """Build a full objectives dict where THIS team owns everything 'first' and the kills."""
    # The dict is meant to be assigned as objectives_a or objectives_b, so this
    # team is always the owner: set first=True everywhere and stack the kills.
    _ = team_letter  # accepted for API symmetry; behaviour is identical for a/b
    return {
        "champion": {"first": True, "kills": 0},
        "baron": {"first": True, "kills": 2},
        "inhibitor": {"first": True, "kills": 3},
        "tower": {"first": True, "kills": 11},
        "dragon": {"first": True, "kills": 3},
    }


def test_extract_labels_team_a_wins():
    """Team A wins, all objectives to team A — verify all 18 keys present with expected values."""
    match = _make_match(
        team_a_win=True,
        team_b_win=False,
        team_a_kills=25,
        team_b_kills=10,
        objectives_a=_all_objectives_for("a"),
        objectives_b={
            "champion": {"first": False, "kills": 0},
            "baron": {"first": False, "kills": 0},
            "inhibitor": {"first": False, "kills": 0},
            "tower": {"first": False, "kills": 2},
            "dragon": {"first": False, "kills": 1},
        },
    )
    labels = extract_labels(match)
    assert labels is not None
    # All 18 keys must be present.
    for k in LABEL_KEYS:
        assert k in labels, f"missing label {k}"
    assert len(labels) == len(LABEL_KEYS)

    assert labels["winner"] == 0
    assert labels["team_b_kill_lead"] == 0  # A has more kills
    assert labels["kill_handicap"] == 15
    assert labels["total_kills"] == 35
    assert labels["team_a_kills"] == 25
    assert labels["team_b_kills"] == 10
    assert labels["kills_odd"] == 1
    assert labels["first_blood"] == 0
    assert labels["first_baron"] == 0
    assert labels["first_inhibitor"] == 0
    assert labels["first_tower"] == 0
    assert labels["total_barons"] == 2
    assert labels["total_dragons"] == 4
    assert labels["total_towers"] == 13
    assert labels["both_baron"] == 0
    assert labels["both_inhibitor"] == 0
    assert labels["both_dragon"] == 1  # A=3, B=1 → both >= 1
    assert labels["elder_dragon"] == 0  # A has 3 dragons, B has 1; no elderDragon key, proxy needs 4


def test_extract_labels_team_b_wins():
    """Team B wins — verify winner=1 and team_b_kill_lead mirrors kill totals."""
    match = _make_match(
        team_a_win=False,
        team_b_win=True,
        team_a_kills=5,
        team_b_kills=20,
        objectives_a={
            "champion": {"first": False, "kills": 0},
            "baron": {"first": False, "kills": 0},
            "inhibitor": {"first": False, "kills": 0},
            "tower": {"first": False, "kills": 1},
            "dragon": {"first": False, "kills": 0},
        },
        objectives_b=_all_objectives_for("b"),
    )
    labels = extract_labels(match)
    assert labels is not None
    assert labels["winner"] == 1
    assert labels["team_b_kill_lead"] == 1  # B has more kills
    assert labels["kill_handicap"] == -15
    assert labels["total_kills"] == 25
    assert labels["team_a_kills"] == 5
    assert labels["team_b_kills"] == 20
    assert labels["kills_odd"] == 1
    assert labels["first_blood"] == 1
    assert labels["first_baron"] == 1
    assert labels["first_inhibitor"] == 1
    assert labels["first_tower"] == 1


def test_extract_labels_no_winner_returns_none():
    """Both teams have win=False — must return None."""
    match = _make_match(team_a_win=False, team_b_win=False)
    assert extract_labels(match) is None


def test_extract_labels_both_win_returns_none():
    """Both teams have win=True (malformed) — must return None."""
    match = _make_match(team_a_win=True, team_b_win=True)
    assert extract_labels(match) is None


def test_extract_labels_missing_team_returns_none():
    """Only one team present — must return None."""
    match = _make_match(skip_team_b=True)
    assert extract_labels(match) is None
    match2 = _make_match(skip_team_a=True)
    assert extract_labels(match2) is None


def test_extract_labels_kills_odd():
    """Construct match with odd/even total kills."""
    odd_match = _make_match(team_a_kills=10, team_b_kills=5)  # 15 total
    even_match = _make_match(team_a_kills=10, team_b_kills=10)  # 20 total

    odd_labels = extract_labels(odd_match)
    even_labels = extract_labels(even_match)

    assert odd_labels is not None and even_labels is not None
    assert odd_labels["kills_odd"] == 1
    assert even_labels["kills_odd"] == 0


def test_extract_labels_first_events_neither():
    """No team has first=True for an objective → first_X = 2 ('neither')."""
    match = _make_match(
        objectives_a={
            "champion": {"first": False, "kills": 0},
            "baron": {"first": False, "kills": 0},
            "inhibitor": {"first": False, "kills": 0},
            "tower": {"first": False, "kills": 0},
            "dragon": {"first": False, "kills": 0},
        },
        objectives_b={
            "champion": {"first": False, "kills": 0},
            "baron": {"first": False, "kills": 0},
            "inhibitor": {"first": False, "kills": 0},
            "tower": {"first": False, "kills": 0},
            "dragon": {"first": False, "kills": 0},
        },
    )
    labels = extract_labels(match)
    assert labels is not None
    assert labels["first_blood"] == 2
    assert labels["first_baron"] == 2
    assert labels["first_inhibitor"] == 2
    assert labels["first_tower"] == 2


def test_extract_labels_both_teams_objectives():
    """Both teams kill barons / dragons / inhibitors → both_X = 1."""
    match = _make_match(
        objectives_a={
            "champion": {"first": True, "kills": 0},
            "baron": {"first": True, "kills": 1},
            "inhibitor": {"first": True, "kills": 1},
            "tower": {"first": True, "kills": 5},
            "dragon": {"first": True, "kills": 2},
        },
        objectives_b={
            "champion": {"first": False, "kills": 0},
            "baron": {"first": False, "kills": 1},
            "inhibitor": {"first": False, "kills": 1},
            "tower": {"first": False, "kills": 4},
            "dragon": {"first": False, "kills": 1},
        },
    )
    labels = extract_labels(match)
    assert labels is not None
    assert labels["both_baron"] == 1
    assert labels["both_inhibitor"] == 1
    assert labels["both_dragon"] == 1
    assert labels["total_barons"] == 2
    assert labels["total_dragons"] == 3
    assert labels["total_towers"] == 9


def test_extract_labels_elder_dragon_explicit():
    """Team has objectives.elderDragon.kills = 1 → elder_dragon = 1."""
    match = _make_match(elder_a=1)
    labels = extract_labels(match)
    assert labels is not None
    assert labels["elder_dragon"] == 1

    match_b = _make_match(elder_b=2)
    labels_b = extract_labels(match_b)
    assert labels_b is not None
    assert labels_b["elder_dragon"] == 1


def test_extract_labels_elder_dragon_proxy():
    """No elderDragon key, but dragon.kills >= 5 → elder_dragon = 1 via proxy."""
    match = _make_match(
        objectives_a={
            "champion": {"first": True, "kills": 0},
            "baron": {"first": True, "kills": 1},
            "inhibitor": {"first": True, "kills": 1},
            "tower": {"first": True, "kills": 5},
            "dragon": {"first": True, "kills": 5},  # proxy threshold
        },
        objectives_b={
            "champion": {"first": False, "kills": 0},
            "baron": {"first": False, "kills": 0},
            "inhibitor": {"first": False, "kills": 0},
            "tower": {"first": False, "kills": 4},
            "dragon": {"first": False, "kills": 1},
        },
    )
    labels = extract_labels(match)
    assert labels is not None
    assert labels["elder_dragon"] == 1

    # 4 dragons should NOT trigger the proxy (elder spawns after 4th drake but
    # isn't necessarily taken until a team kills 5+).
    match_no_elder = _make_match(
        objectives_a={
            "champion": {"first": True, "kills": 0},
            "baron": {"first": True, "kills": 1},
            "inhibitor": {"first": True, "kills": 1},
            "tower": {"first": True, "kills": 5},
            "dragon": {"first": True, "kills": 4},
        },
        objectives_b={
            "champion": {"first": False, "kills": 0},
            "baron": {"first": False, "kills": 0},
            "inhibitor": {"first": False, "kills": 0},
            "tower": {"first": False, "kills": 4},
            "dragon": {"first": False, "kills": 1},
        },
    )
    labels_no_elder = extract_labels(match_no_elder)
    assert labels_no_elder is not None
    assert labels_no_elder["elder_dragon"] == 0


def test_extract_labels_team_b_kill_lead_tie_favors_team_a():
    """Ties on total kills must collapse to team_b_kill_lead=0 (team-A side)."""
    match = _make_match(team_a_kills=15, team_b_kills=15, team_a_win=True, team_b_win=False)
    labels = extract_labels(match)
    assert labels is not None
    assert labels["team_b_kill_lead"] == 0
    assert labels["total_kills"] == 30
    assert labels["kill_handicap"] == 0


def test_extract_labels_label_keys_constant_matches():
    """LABEL_KEYS list contains exactly the 18 keys returned by extract_labels."""
    match = _make_match()
    labels = extract_labels(match)
    assert labels is not None
    assert set(labels.keys()) == set(LABEL_KEYS)
    assert len(LABEL_KEYS) == 18


def test_extract_labels_coerces_non_int_kills():
    """_safe_int handles None and string-number kills values."""
    # _make_match takes team_a_kills/team_b_kills as ints, but we override below
    match = _make_match(team_a_kills=0, team_b_kills=0)
    # Override participant kills with non-int values
    match["info"]["participants"][0]["kills"] = None
    match["info"]["participants"][1]["kills"] = "7"
    labels = extract_labels(match)
    assert labels is not None
    assert labels["team_a_kills"] == 0  # None -> 0
    assert labels["team_b_kills"] == 7  # "7" -> 7


def test_extract_timeline_labels_team_a_5_kills():
    """Team A reaches 5 kills first."""
    from feature_labels import extract_timeline_labels
    # Build a timeline where killer participants 1-5 each get one kill in sequence
    timeline = {
        "info": {
            "frames": [
                {"events": [{"type": "CHAMPION_KILL", "killerId": i, "victimId": 6}] }
                for i in range(1, 6)
            ]
        }
    }
    labels = extract_timeline_labels(timeline)
    assert labels is not None
    assert labels["first_to_5_kills"] == 0  # team A
    assert labels["first_to_10_kills"] == 2  # neither
    assert labels["first_to_15_kills"] == 2
    assert labels["first_to_20_kills"] == 2


def test_extract_timeline_labels_team_b_10_kills():
    """Team B reaches 10 kills first."""
    from feature_labels import extract_timeline_labels
    # Team A: 9 kills. Team B: 10 kills. Team B should "win" first_to_10.
    events = []
    for i in range(1, 10):  # 9 team-A kills (killerId 1-9 cycling through 1-5)
        events.append({"type": "CHAMPION_KILL", "killerId": ((i - 1) % 5) + 1, "victimId": 6})
    for i in range(10):  # 10 team-B kills (killerId 6-10 cycling)
        events.append({"type": "CHAMPION_KILL", "killerId": ((i) % 5) + 6, "victimId": 1})

    timeline = {"info": {"frames": [{"events": events}]}}
    labels = extract_timeline_labels(timeline)
    assert labels is not None
    assert labels["first_to_5_kills"] == 0  # team A (was reached after 5 of the 9 team-A kills)
    assert labels["first_to_10_kills"] == 1  # team B
    assert labels["first_to_15_kills"] == 2
    assert labels["first_to_20_kills"] == 2


def test_extract_timeline_labels_ignores_non_champion_kill():
    """Events other than CHAMPION_KILL are ignored."""
    from feature_labels import extract_timeline_labels
    timeline = {
        "info": {
            "frames": [{
                "events": [
                    {"type": "BUILDING_KILL", "killerId": 1},
                    {"type": "MONSTER_KILL", "killerId": 2},
                    {"type": "WARD_PLACED", "killerId": 3},
                ]
            }]
        }
    }
    labels = extract_timeline_labels(timeline)
    assert labels is not None
    # No CHAMPION_KILL events → no team reaches any threshold
    assert labels == {
        "first_to_5_kills": 2,
        "first_to_10_kills": 2,
        "first_to_15_kills": 2,
        "first_to_20_kills": 2,
    }


def test_extract_timeline_labels_returns_none_on_malformed():
    """Empty / non-dict / missing frames returns None."""
    from feature_labels import extract_timeline_labels
    assert extract_timeline_labels({}) is None
    assert extract_timeline_labels({"info": {}}) is None
    assert extract_timeline_labels({"info": {"frames": []}}) is None


def test_timeline_label_keys_constant():
    """TIMELINE_LABEL_KEYS contains exactly the 4 keys."""
    from feature_labels import TIMELINE_LABEL_KEYS
    assert TIMELINE_LABEL_KEYS == ["first_to_5_kills", "first_to_10_kills", "first_to_15_kills", "first_to_20_kills"]
