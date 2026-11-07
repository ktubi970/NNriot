import random
import pytest
from core.pre_match_dataset import extract_major_patch, split_records


def rows_for(patch, times):
    return [
        {
            "match_id": f"{patch}_{t}",
            "game_creation_time": t,
            "winner_label": 0,
            "labels_json": {
                "winner": 0, "team_b_kill_lead": 0, "kill_handicap": 0,
                "total_kills": 0, "team_a_kills": 0, "team_b_kills": 0,
                "kills_odd": 0, "first_blood": 0, "first_baron": 0,
                "first_inhibitor": 0, "first_tower": 0, "total_barons": 0,
                "total_dragons": 0, "total_towers": 0, "both_baron": 0,
                "both_inhibitor": 0, "both_dragon": 0, "elder_dragon": 0,
            },
            "feature_json": {
                "game_mode": "CLASSIC",
                "game_version": patch,
                "team_a": [{"champion": "Ahri", "role": "MIDDLE"} for _ in range(5)],
                "team_b": [{"champion": "Zed", "role": "MIDDLE"} for _ in range(5)],
            },
        }
        for t in times
    ]


def test_extract_major_patch_valid():
    assert extract_major_patch("14.10.1") == (14, 10)
    assert extract_major_patch("16.18") == (16, 18)
    assert extract_major_patch("0.1.2.3") == (0, 1)


@pytest.mark.parametrize("bad", ["UNKNOWN", "", None, "16", "x.18", "14.-1", "abc"])
def test_extract_major_patch_invalid(bad):
    with pytest.raises(ValueError):
        extract_major_patch(bad)


def test_patch_holdout_rejects_overlapping_time_ranges():
    rows = (
        rows_for("14.1.1", list(range(1, 19)) + [100])
        + rows_for("14.2.1", [20, 30])
        + rows_for("14.3.1", [40, 50])
    )
    with pytest.raises(ValueError, match="overlap"):
        split_records(rows, "patch")


def test_fallback_reports_effective_strategy():
    splits, meta = split_records(rows_for("16.18.1", range(1, 41)), "patch")
    assert meta["requested_strategy"] == "patch"
    assert meta["effective_strategy"] == "chronological"
    assert meta["fallback_reason"] == "fewer_than_three_patches"
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [28, 6, 6]


@pytest.mark.parametrize("version", ["UNKNOWN", "", None, "16", "x.18"])
def test_patch_mode_rejects_unknown_versions(version):
    rows = rows_for("16.18.1", range(1, 41))
    rows[0]["feature_json"]["game_version"] = version
    with pytest.raises(ValueError, match="patch"):
        split_records(rows, "patch")


def test_valid_patch_split_groups_patches_strictly():
    # 14.9: 20 rows, 14.10: 10 rows, 14.11: 10 rows
    train_rows = rows_for("14.9.1", range(100, 120))
    val_rows = rows_for("14.10.1", range(200, 210))
    test_rows = rows_for("14.11.1", range(300, 310))
    all_rows = train_rows + val_rows + test_rows

    splits, meta = split_records(all_rows, "patch")
    assert meta["requested_strategy"] == "patch"
    assert meta["effective_strategy"] == "patch"
    assert meta["fallback_reason"] is None
    assert meta["patches"]["train"] == ["14.9"]
    assert meta["patches"]["validation"] == ["14.10"]
    assert meta["patches"]["test"] == ["14.11"]

    assert [r["match_id"] for r in splits["train"]] == [r["match_id"] for r in train_rows]
    assert [r["match_id"] for r in splits["validation"]] == [r["match_id"] for r in val_rows]
    assert [r["match_id"] for r in splits["test"]] == [r["match_id"] for r in test_rows]


def test_patch_split_rejects_equal_timestamps_across_patch_boundaries():
    rows = (
        rows_for("14.1.1", range(1, 19))
        + rows_for("14.2.1", [18, 30])
        + rows_for("14.3.1", [40, 50])
    )
    with pytest.raises(ValueError, match="overlap"):
        split_records(rows, "patch")


def test_patch_split_rejects_fewer_than_20_records():
    rows = rows_for("14.1.1", range(1, 6)) + rows_for("14.2.1", range(6, 11)) + rows_for("14.3.1", range(11, 15))
    with pytest.raises(ValueError, match="20"):
        split_records(rows, "patch")


def test_patch_split_rejects_duplicate_ids():
    rows = rows_for("14.1.1", range(1, 19)) + rows_for("14.2.1", [20, 30]) + rows_for("14.3.1", [40, 50])
    rows[-1]["match_id"] = rows[0]["match_id"]
    with pytest.raises(ValueError, match="duplicate"):
        split_records(rows, "patch")


def test_patch_split_is_input_permutation_invariant():
    train_rows = rows_for("14.9.1", range(100, 120))
    val_rows = rows_for("14.10.1", range(200, 210))
    test_rows = rows_for("14.11.1", range(300, 310))
    all_rows = train_rows + val_rows + test_rows

    splits1, meta1 = split_records(all_rows, "patch")

    shuffled = list(all_rows)
    random.Random(12345).shuffle(shuffled)
    splits2, meta2 = split_records(shuffled, "patch")

    assert meta1 == meta2
    for k in ("train", "validation", "test"):
        assert [r["match_id"] for r in splits1[k]] == [r["match_id"] for r in splits2[k]]
