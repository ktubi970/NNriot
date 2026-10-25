"""Collection provenance flags: ladder seeds, legacy anchors and crawled imbalance per match."""

import os
import sqlite3

import pytest

from core import database
from core.collection.provenance import (
    LADDER_SEED_MIN_PRIORITY,
    CrawlRegistry,
    load_crawl_registry,
    match_provenance,
    parse_sqlite_timestamp_ms,
)

ENTRY = 1_000_000


@pytest.fixture
def registry():
    return CrawlRegistry(
        seed_entry_ms={"seed": ENTRY},
        anchor_saved_ms={"anchor": ENTRY},
        crawled=frozenset({"seed", "c1", "c2", "c3"}),
    )


def test_seed_is_counted_on_its_own_team(registry):
    prov = match_provenance(["x1", "x2"], ["seed", "x3"], ENTRY - 1, registry)
    assert (prov.team_a.seeds, prov.team_b.seeds) == (0, 1)
    assert prov.has_ladder_seed


def test_seed_match_is_pre_entry_only_strictly_before_entry(registry):
    before = match_provenance(["seed"], ["x"], ENTRY - 1, registry)
    at_entry = match_provenance(["seed"], ["x"], ENTRY, registry)
    assert before.team_a.pre_entry_seeds == 1 and before.has_pre_entry_seed
    assert at_entry.team_a.pre_entry_seeds == 0 and not at_entry.has_pre_entry_seed
    assert at_entry.has_ladder_seed


def test_legacy_anchor_is_pre_entry_before_its_last_save(registry):
    before = match_provenance(["anchor"], ["x"], ENTRY - 1, registry)
    after = match_provenance(["anchor"], ["x"], ENTRY, registry)
    assert before.team_a.legacy_anchors == 1 and before.team_a.pre_entry_anchors == 1
    assert before.has_pre_entry_anchor and not before.has_ladder_seed
    assert after.team_a.legacy_anchors == 1 and not after.has_pre_entry_anchor


def test_pre_entry_seed_also_counts_as_pre_entry_anchor(registry):
    assert match_provenance(["seed"], ["x"], ENTRY - 1, registry).has_pre_entry_anchor


def test_crawled_imbalance_is_team_b_minus_team_a(registry):
    prov = match_provenance(["c1", "x1"], ["c2", "c3"], ENTRY, registry)
    assert (prov.team_a.crawled, prov.team_b.crawled) == (1, 2)
    assert prov.crawled_imbalance == 1


def test_balanced_requires_no_seed_and_equal_crawled_counts(registry):
    assert match_provenance(["c1", "x1"], ["c2", "x2"], ENTRY, registry).is_balanced
    assert not match_provenance(["c1", "c3"], ["c2", "x2"], ENTRY, registry).is_balanced
    # Equal crawled counts (seed and c1) do not make a seed match balanced.
    assert not match_provenance(["seed", "x1"], ["c1", "x2"], ENTRY, registry).is_balanced


def test_team_swap_mirrors_counts_and_keeps_match_flags(registry):
    a, b = ["seed", "c1", "anchor"], ["c2", "x1", "x2"]
    prov = match_provenance(a, b, ENTRY - 1, registry)
    mirrored = match_provenance(b, a, ENTRY - 1, registry)
    assert mirrored == prov.swapped()
    assert mirrored.crawled_imbalance == -prov.crawled_imbalance
    for flag in ("has_ladder_seed", "has_pre_entry_seed", "has_pre_entry_anchor", "is_balanced"):
        assert getattr(mirrored, flag) == getattr(prov, flag)


def test_player_order_within_a_team_does_not_matter(registry):
    a, b = ["seed", "c1", "x1"], ["c2", "anchor", "x2"]
    prov = match_provenance(a, b, ENTRY - 1, registry)
    assert match_provenance(a[::-1], b[::-1], ENTRY - 1, registry) == prov


def test_players_crawled_after_the_match_are_counted_apart():
    registry = CrawlRegistry(seed_entry_ms={}, anchor_saved_ms={}, crawled=frozenset({"early", "late"}),
                             crawled_at_ms={"early": 100, "late": 300})
    prov = match_provenance(["early"], ["late"], 200, registry)
    assert (prov.team_a.crawled, prov.team_b.crawled) == (1, 1)
    assert (prov.team_a.crawled_after, prov.team_b.crawled_after) == (0, 1)
    assert prov.crawled_after_imbalance == 1
    # A crawl at the creation instant cannot have preceded the match.
    assert match_provenance(["early"], [], 100, registry).team_a.crawled_after == 1


def test_missing_puuids_count_as_unknown_players(registry):
    prov = match_provenance([None, "", "c1"], ["c2"], ENTRY, registry)
    assert (prov.team_a.crawled, prov.team_b.crawled) == (1, 1)
    assert prov.is_balanced


def test_as_row_uses_flat_team_suffixed_columns(registry):
    row = match_provenance(["seed"], ["c1", "c2"], ENTRY - 1, registry).as_row()
    assert row["seeds_a"] == 1 and row["seeds_b"] == 0
    assert row["crawled_a"] == 1 and row["crawled_b"] == 2
    assert row["crawled_imbalance"] == 1
    assert row["has_ladder_seed"] == 1 and row["has_pre_entry_seed"] == 1
    assert row["balanced"] == 0


def test_parse_sqlite_timestamp_is_utc_milliseconds():
    assert parse_sqlite_timestamp_ms("2026-10-11 09:25:52") == 1_791_710_752_000
    assert parse_sqlite_timestamp_ms(None) is None


@pytest.fixture
def crawl_db(tmp_path):
    db_file = str(tmp_path / "provenance.db")
    database.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.executemany(
            "INSERT INTO crawl_queue (puuid, region, priority, is_processed, created_at, processed_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("ladder10", "EUW", 10, 1, "2026-10-11 09:25:52", "2026-10-11 09:27:36"),
                ("ladder5", "KR", 5, 0, "2026-10-12 00:00:00", None),
                ("snow_done", "NA", 0, 1, "2026-10-11 10:00:00", "2026-10-11 10:47:05"),
                ("snow_pending", "NA", 0, 0, "2026-10-11 10:00:00", None),
            ],
        )
        conn.executemany(
            "INSERT INTO players (puuid, game_name, tag_line, last_updated) VALUES (?, ?, ?, ?)",
            [
                ("ladder10", "Seed", "EUW", "2026-05-01 00:00:00"),
                ("pro", "Pro", "KR1", "2026-05-02 12:00:00"),
            ],
        )
    return db_file


def test_load_crawl_registry_classifies_queue_and_players(crawl_db):
    reg = load_crawl_registry(crawl_db)
    assert LADDER_SEED_MIN_PRIORITY == 5
    assert reg.seed_entry_ms == {
        "ladder10": parse_sqlite_timestamp_ms("2026-10-11 09:25:52"),
        "ladder5": parse_sqlite_timestamp_ms("2026-10-12 00:00:00"),
    }
    # A ladder seed that was also saved earlier stays a seed, not a legacy anchor.
    assert reg.anchor_saved_ms == {"pro": parse_sqlite_timestamp_ms("2026-05-02 12:00:00")}
    assert reg.crawled == frozenset({"ladder10", "snow_done"})
    assert reg.crawled_at_ms == {"ladder10": parse_sqlite_timestamp_ms("2026-10-11 09:27:36"),
                                 "snow_done": parse_sqlite_timestamp_ms("2026-10-11 10:47:05")}


def test_load_crawl_registry_never_writes_the_database(crawl_db):
    before = os.stat(crawl_db).st_mtime_ns, open(crawl_db, "rb").read()
    load_crawl_registry(crawl_db)
    assert (os.stat(crawl_db).st_mtime_ns, open(crawl_db, "rb").read()) == before
