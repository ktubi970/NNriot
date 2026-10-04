import pytest
from core.features.timeline_metrics import extract_timeline_lane_metrics


def test_extract_timeline_lane_metrics_standard():
    # Synthetic timeline with 16 frames (indices 0..15) and 10 participants
    frames = []
    for m in range(16):
        p_frames = {}
        for pid in range(1, 11):
            # Blue side (1-5) gets 500 gold/min, Red side (6-10) gets 400 gold/min
            gold = 500 * m if pid <= 5 else 400 * m
            cs = 8 * m if pid <= 5 else 7 * m
            xp = 400 * m if pid <= 5 else 350 * m
            p_frames[str(pid)] = {
                "participantId": pid,
                "totalGold": gold,
                "minionsKilled": cs,
                "jungleMinionsKilled": 0,
                "xp": xp,
            }
        events = []
        if m == 5:
            # First blood at min 5: killer pid 1, assist pid 2, victim pid 6
            events.append({
                "type": "CHAMPION_KILL",
                "timestamp": 300000,
                "killerId": 1,
                "assistingParticipantIds": [2],
                "victimId": 6,
            })
        frames.append({"participantFrames": p_frames, "events": events})

    timeline_data = {
        "info": {
            "participants": [{"participantId": i, "puuid": f"puuid_{i}"} for i in range(1, 11)],
            "frames": frames,
        }
    }

    metrics = extract_timeline_lane_metrics(timeline_data)
    assert len(metrics) == 10

    # Participant 1 (puuid_1, Top Blue) vs Participant 6 (puuid_6, Top Red)
    # At min 15: Gold A = 7500, Gold B = 6000 -> GD15 = +1500
    p1 = metrics["puuid_1"]
    assert p1["gd15"] == 1500.0
    assert p1["csd15"] == 15.0  # 120 - 105
    assert p1["xpd15"] == 750.0  # 6000 - 5250
    assert p1["fbi10"] == 1.0   # Killer in FB at min 5

    p6 = metrics["puuid_6"]
    assert p6["gd15"] == -1500.0  # Exact anti-symmetry
    assert p6["csd15"] == -15.0
    assert p6["xpd15"] == -750.0
    assert p6["fbi10"] == 0.0

    p2 = metrics["puuid_2"]
    assert p2["fbi10"] == 1.0  # Assist in FB

    # Check anti-symmetry across all 5 lane matchups (1v6, 2v7, 3v8, 4v9, 5v10)
    for i in range(1, 6):
        puuid_blue = f"puuid_{i}"
        puuid_red = f"puuid_{i + 5}"
        assert metrics[puuid_blue]["gd15"] == -metrics[puuid_red]["gd15"]
        assert metrics[puuid_blue]["csd15"] == -metrics[puuid_red]["csd15"]
        assert metrics[puuid_blue]["xpd15"] == -metrics[puuid_red]["xpd15"]


def test_extract_timeline_lane_metrics_short_game():
    # Game with only 10 frames (indices 0..9)
    frames = []
    for m in range(10):
        p_frames = {}
        for pid in range(1, 11):
            gold = 300 * m if pid <= 5 else 200 * m
            cs = 6 * m if pid <= 5 else 5 * m
            xp = 250 * m if pid <= 5 else 200 * m
            p_frames[str(pid)] = {
                "participantId": pid,
                "totalGold": gold,
                "minionsKilled": cs,
                "jungleMinionsKilled": 1 * m,
                "xp": xp,
            }
        frames.append({"participantFrames": p_frames, "events": []})

    timeline_data = {
        "info": {
            "participants": [{"participantId": i, "puuid": f"puuid_{i}"} for i in range(1, 11)],
            "frames": frames,
        }
    }

    metrics = extract_timeline_lane_metrics(timeline_data)
    assert len(metrics) == 10
    # Should use frame 9 (last available)
    p1 = metrics["puuid_1"]
    assert p1["gd15"] == 900.0
    assert p1["csd15"] == 9.0
    assert p1["xpd15"] == 450.0
    assert p1["fbi10"] == 0.0


def test_extract_timeline_lane_metrics_fallback_participants():
    frames = [
        {
            "participantFrames": {
                str(i): {
                    "totalGold": 1000 * i,
                    "minionsKilled": 10 * i,
                    "jungleMinionsKilled": 0,
                    "xp": 500 * i,
                }
                for i in range(1, 11)
            },
            "events": [],
        }
    ]
    timeline_data = {"info": {"frames": frames}}
    ext_participants = [{"participantId": i, "puuid": f"ext_puuid_{i}"} for i in range(1, 11)]

    metrics = extract_timeline_lane_metrics(timeline_data, participants=ext_participants)
    assert len(metrics) == 10
    assert "ext_puuid_1" in metrics
    assert "ext_puuid_6" in metrics


def test_extract_timeline_first_blood_after_10min():
    # First blood occurs at minute 12 (720,000 ms) -> beyond 10 min cutoff
    frames = [
        {"participantFrames": {str(i): {"totalGold": 1000, "minionsKilled": 10, "jungleMinionsKilled": 0, "xp": 500} for i in range(1, 11)}, "events": []}
        for _ in range(16)
    ]
    frames[12]["events"] = [{
        "type": "CHAMPION_KILL",
        "timestamp": 720000,
        "killerId": 3,
        "assistingParticipantIds": [4],
        "victimId": 8,
    }]

    timeline_data = {
        "info": {
            "participants": [{"participantId": i, "puuid": f"puuid_{i}"} for i in range(1, 11)],
            "frames": frames,
        }
    }

    metrics = extract_timeline_lane_metrics(timeline_data)
    for p in metrics.values():
        assert p["fbi10"] == 0.0


def test_first_blood_neutral_execution_then_real_fb():
    # Execution at min 2 (killerId == 0), then real FB at min 4 (killerId == 3, assist 4)
    frames = [
        {"participantFrames": {str(i): {"totalGold": 1000, "minionsKilled": 10, "jungleMinionsKilled": 0, "xp": 500} for i in range(1, 11)}, "events": []}
        for _ in range(16)
    ]
    frames[2]["events"] = [{
        "type": "CHAMPION_KILL",
        "timestamp": 120000,
        "killerId": 0,
        "victimId": 6,
    }]
    frames[4]["events"] = [{
        "type": "CHAMPION_KILL",
        "timestamp": 240000,
        "killerId": 3,
        "assistingParticipantIds": [4],
        "victimId": 8,
    }]

    timeline_data = {
        "info": {
            "participants": [{"participantId": i, "puuid": f"puuid_{i}"} for i in range(1, 11)],
            "frames": frames,
        }
    }

    metrics = extract_timeline_lane_metrics(timeline_data)
    assert metrics["puuid_3"]["fbi10"] == 1.0
    assert metrics["puuid_4"]["fbi10"] == 1.0
    assert metrics["puuid_1"]["fbi10"] == 0.0
    assert metrics["puuid_6"]["fbi10"] == 0.0


def test_strict_participant_ids_rejection():
    # Non-standard participant IDs: 1..9, 11 (missing 10)
    timeline_data = {
        "info": {
            "participants": [{"participantId": i, "puuid": f"puuid_{i}"} for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]],
            "frames": [{"participantFrames": {}, "events": []}],
        }
    }
    assert extract_timeline_lane_metrics(timeline_data) == {}


def test_malformed_frames_and_events_resilience():
    # Non-list frames
    assert extract_timeline_lane_metrics({"info": {"frames": "not-a-list"}}) == {}
    assert extract_timeline_lane_metrics({"info": {"frames": 12345}}) == {}

    # Target frame is not a dict
    timeline_bad_frame = {
        "info": {
            "participants": [{"participantId": i, "puuid": f"puuid_{i}"} for i in range(1, 11)],
            "frames": ["invalid_frame_string"],
        }
    }
    assert extract_timeline_lane_metrics(timeline_bad_frame) == {}

    # Malformed stats, non-dict participant frames, malformed event attributes
    timeline_dirty = {
        "info": {
            "participants": [{"participantId": i, "puuid": f"puuid_{i}"} for i in range(1, 11)],
            "frames": [
                None,  # malformed frame element
                {
                    "participantFrames": {
                        "1": {"totalGold": "bad_gold", "minionsKilled": None, "xp": "invalid"},
                        "2": "not_a_dict",
                        # remaining 3..10 missing from participantFrames
                    },
                    "events": [
                        "not_a_dict_event",
                        {
                            "type": "CHAMPION_KILL",
                            "timestamp": "not_a_number",
                            "killerId": "not_an_int",
                            "assistingParticipantIds": "not_a_list",
                        },
                        {
                            "type": "CHAMPION_KILL",
                            "timestamp": 180000,
                            "killerId": 1,
                            "assistingParticipantIds": [2, "bad_assist"],
                            "victimId": 6,
                        },
                    ],
                },
            ],
        }
    }
    res = extract_timeline_lane_metrics(timeline_dirty)
    assert len(res) == 10
    # Safe float fallback to 0.0
    assert res["puuid_1"]["gd15"] == 0.0
    assert res["puuid_1"]["csd15"] == 0.0
    assert res["puuid_1"]["xpd15"] == 0.0
    # Event parsing safely identified killer 1 and assist 2
    assert res["puuid_1"]["fbi10"] == 1.0
    assert res["puuid_2"]["fbi10"] == 1.0
    assert res["puuid_3"]["fbi10"] == 0.0


def test_extract_timeline_lane_metrics_empty_or_malformed():
    assert extract_timeline_lane_metrics({}) == {}
    assert extract_timeline_lane_metrics({"info": {}}) == {}
    assert extract_timeline_lane_metrics(None) == {}
    assert extract_timeline_lane_metrics("not a dict") == {}
    # Missing participants (< 10)
    timeline_incomplete = {
        "info": {
            "participants": [{"participantId": 1, "puuid": "puuid_1"}],
            "frames": [{"participantFrames": {}, "events": []}],
        }
    }
    assert extract_timeline_lane_metrics(timeline_incomplete) == {}
