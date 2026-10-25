"""Read-only diagnostic of the survivorship bias left by the crawler's ladder seeds.

    python scripts/diagnose_crawl_seed_bias.py --db training_data.db \
        --dataset-dir runs/aligned_dataset_20261023_prospective \
        --output-dir runs/crawl_provenance_20261025

Writes ``report.json`` (win rates by provenance and month), ``crawl_provenance.csv``
(one row per train/validation match of the aligned dataset) and ``manifest.json``
(input hashes). The sealed test is never opened: the database part stops strictly
before the first sealed-test creation and the dataset part reads only train.json
and validation.json.
"""
import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.collection.provenance import (  # noqa: E402
    MatchProvenance,
    load_crawl_registry,
    match_provenance,
)
from core.selective_classification import wilson_score_interval  # noqa: E402

MODEL_SPLITS = ("train", "validation")
PLAYER_CATEGORIES = ("seed_pre_entry", "seed_post_entry", "anchor_pre_save", "anchor_post_save",
                     "crawled", "queued_pending", "never_queued")


class Rate:
    def __init__(self):
        self.n = self.k = 0

    def add(self, success):
        self.n += 1
        self.k += int(bool(success))

    def as_dict(self):
        if not self.n:
            return {"n": 0}
        low, high = wilson_score_interval(self.k, self.n)
        return {"n": self.n, "rate": round(self.k / self.n, 4), "wilson95": [round(low, 4), round(high, 4)]}


def rates_table():
    return defaultdict(lambda: defaultdict(Rate))


def dump(table):
    return {key: {period: rate.as_dict() for period, rate in sorted(periods.items())}
            for key, periods in table.items()}


def month_of(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_test_cutoff_ms(dataset_dir):
    """First sealed-test creation; only its timestamps are read, never its matches."""
    splits = json.loads((Path(dataset_dir) / "model.split.json").read_text(encoding="utf-8"))["splits"]
    return min(int(entry["game_creation_time"]) for entry in splits["test"])


def player_category(puuid, creation_ms, registry, queued):
    entry = registry.seed_entry_ms.get(puuid)
    if entry is not None:
        return "seed_pre_entry" if creation_ms < entry else "seed_post_entry"
    saved = registry.anchor_saved_ms.get(puuid)
    if saved is not None:
        return "anchor_pre_save" if creation_ms < saved else "anchor_post_save"
    if puuid in registry.crawled:
        return "crawled"
    return "queued_pending" if puuid in queued else "never_queued"


def read_db_teams(db_path, cutoff_ms, min_duration_s):
    """Yield (match_id, creation_ms, winners, losers) for complete 5v5 CLASSIC matches before the cutoff.

    One sequential scan, grouped in memory: an ORDER BY through the match_id index is
    random I/O on the HDD that holds the database.
    """
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    grouped = defaultdict(lambda: [0, [], []])
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.execute("PRAGMA query_only=ON")
        for match_id, puuid, win, created in conn.execute(
            "SELECT match_id, puuid, win, game_creation_time FROM match_participants "
            "WHERE game_mode = 'CLASSIC' AND game_creation_time > 0 AND game_creation_time < ? "
            "AND game_duration >= ?",
            (cutoff_ms, min_duration_s),
        ):
            entry = grouped[match_id]
            entry[0] = created
            entry[1 if win else 2].append(puuid)
    for match_id, (creation, winners, losers) in grouped.items():
        if len(winners) == len(losers) == 5:
            yield match_id, creation, winners, losers


def queued_puuids(db_path):
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.execute("PRAGMA query_only=ON")
        return frozenset(p for (p,) in conn.execute("SELECT puuid FROM crawl_queue WHERE puuid IS NOT NULL"))


def add_oriented(table, prov, b_won, periods):
    """Win rates of the side that carries a collection artifact, when only one side carries it.

    Team B is the reference side: ``b_won`` is its outcome and ``prov`` is oriented A/B.
    Each match counts once in every period of ``periods`` and once in ``"all"``.
    """
    a, b = prov.team_a, prov.team_b

    def add(key, won):
        for period in (*periods, "all"):
            table[key][period].add(won)

    def one_sided(count_a, count_b, key):
        if (count_a > 0) != (count_b > 0):
            add(key, b_won if count_b > 0 else not b_won)

    def larger(diff, key):
        if diff:
            add(key, b_won if diff > 0 else not b_won)

    one_sided(a.seeds, b.seeds, "ladder_seed_team")
    one_sided(a.pre_entry_seeds, b.pre_entry_seeds, "pre_entry_seed_team")
    if a.pre_entry_seeds == b.pre_entry_seeds == 0:
        one_sided(a.seeds, b.seeds, "post_entry_only_seed_team")
    one_sided(a.pre_entry_seeds + a.pre_entry_anchors, b.pre_entry_seeds + b.pre_entry_anchors,
              "pre_entry_anchor_team")
    if not prov.has_ladder_seed:
        larger(prov.crawled_imbalance, "more_crawled_team_no_seed")
    if not prov.has_pre_entry_anchor:
        larger(prov.crawled_imbalance, "more_crawled_team_no_pre_entry_anchor")
        gap = min(abs(prov.crawled_imbalance), 3)
        larger(prov.crawled_imbalance, f"more_crawled_team_no_pre_entry_anchor_gap{gap}")
        # Was the extra crawled player's history fetched before or after this match was played?
        larger(prov.crawled_after_imbalance, "more_crawled_after_team_no_pre_entry_anchor")
        larger(prov.crawled_imbalance - prov.crawled_after_imbalance, "more_crawled_before_team_no_pre_entry_anchor")


def db_report(db_path, registry, cutoff_ms, min_duration_s):
    queued = queued_puuids(db_path)
    players, teams, shares = rates_table(), rates_table(), rates_table()
    matches = 0
    for _, creation, winners, losers in read_db_teams(db_path, cutoff_ms, min_duration_s):
        matches += 1
        period = month_of(creation)
        for won, side in ((True, winners), (False, losers)):
            for puuid in side:
                category = player_category(puuid, creation, registry, queued)
                players[category][period].add(won)
                players[category]["all"].add(won)
        # Orient winners as team B so that b_won is always True.
        prov = match_provenance(losers, winners, creation, registry)
        add_oriented(teams, prov, True, (period,))
        for key, value in (("has_ladder_seed", prov.has_ladder_seed),
                           ("has_pre_entry_anchor", prov.has_pre_entry_anchor),
                           ("balanced", prov.is_balanced)):
            shares[key][period].add(value)
            shares[key]["all"].add(value)
    return {"matches": matches, "player_games": dump(players), "artifact_side_wins": dump(teams),
            "match_shares": dump(shares)}


def load_split_rows(dataset_dir):
    """Train and validation rows of an aligned dataset; any sealed-test file is left unopened."""
    for split in MODEL_SPLITS:
        rows = json.loads((Path(dataset_dir) / f"{split}.json").read_text(encoding="utf-8"))
        for row in rows:
            features = row["feature_json"]
            yield {
                "split": split,
                "match_id": row["match_id"],
                "game_creation_time": int(row["game_creation_time"]),
                "team_a": [p.get("puuid") for p in features["team_a"]],
                "team_b": [p.get("puuid") for p in features["team_b"]],
                "winner": int(row["labels_json"]["winner"]),
            }


def dataset_report(rows, registry):
    sides, shares, red = rates_table(), rates_table(), rates_table()
    companion = []
    for row in rows:
        prov: MatchProvenance = match_provenance(row["team_a"], row["team_b"], row["game_creation_time"], registry)
        period = f"{row['split']}:{month_of(row['game_creation_time'])}"
        b_won = row["winner"] == 1
        add_oriented(sides, prov, b_won, (period, row["split"]))
        for key, value in (("has_ladder_seed", prov.has_ladder_seed),
                           ("has_pre_entry_seed", prov.has_pre_entry_seed),
                           ("has_pre_entry_anchor", prov.has_pre_entry_anchor),
                           ("balanced", prov.is_balanced)):
            shares[key][period].add(value)
            shares[key][row["split"]].add(value)
        red["team_b_all"][row["split"]].add(b_won)
        if prov.is_balanced:
            red["team_b_balanced"][row["split"]].add(b_won)
        if not prov.has_pre_entry_anchor and prov.crawled_imbalance == 0:
            red["team_b_strict_balanced"][row["split"]].add(b_won)
        companion.append({"match_id": row["match_id"], "split": row["split"],
                          "game_creation_time": row["game_creation_time"], **prov.as_row()})
    return {"artifact_side_wins": dump(sides), "match_shares": dump(shares), "team_b_wins": dump(red)}, companion


def registry_summary(registry):
    seeds = sorted(registry.seed_entry_ms)
    entries = sorted(registry.seed_entry_ms.values())
    return {
        "ladder_seeds": len(seeds),
        "ladder_seed_entry_utc": [datetime.fromtimestamp(v / 1000, timezone.utc).isoformat()
                                  for v in entries[:1] + entries[-1:]],
        "ladder_seeds_sha256": hashlib.sha256("\n".join(seeds).encode()).hexdigest(),
        "legacy_anchors": len(registry.anchor_saved_ms),
        "crawled": len(registry.crawled),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-duration-s", type=int, default=300, help="remake guard for the database part")
    args = parser.parse_args(argv)

    registry = load_crawl_registry(str(args.db))
    cutoff_ms = sealed_test_cutoff_ms(args.dataset_dir)
    report = {"registry": registry_summary(registry), "cutoff_ms": cutoff_ms,
              "cutoff_utc": datetime.fromtimestamp(cutoff_ms / 1000, timezone.utc).isoformat()}
    report["database"] = db_report(str(args.db), registry, cutoff_ms, args.min_duration_s)
    report["dataset"], companion = dataset_report(load_split_rows(args.dataset_dir), registry)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "crawl_provenance.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(companion[0]))
        writer.writeheader()
        writer.writerows(companion)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # crawl_queue can be reset or merged later; keep the exact registry behind the counts.
    snapshot = {"seed_entry_ms": dict(sorted(registry.seed_entry_ms.items())),
                "anchor_saved_ms": dict(sorted(registry.anchor_saved_ms.items())),
                "crawled_at_ms": {p: registry.crawled_at_ms.get(p) for p in sorted(registry.crawled)}}
    (args.output_dir / "registry.json").write_text(json.dumps(snapshot), encoding="utf-8")

    dataset_manifest = json.loads((args.dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    inputs = {}
    for split in MODEL_SPLITS:
        name = f"{split}.json"
        actual = sha256_file(args.dataset_dir / name)
        if actual != dataset_manifest["files"][name]["sha256"]:
            raise SystemExit(f"{name} does not match dataset.json")
        inputs[name] = actual
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "nnriot.crawl_provenance.v1",
        "dataset_dir": str(args.dataset_dir),
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "split_sha256": dataset_manifest["split_sha256"],
        "inputs": inputs,
        "db": str(args.db),
        "cutoff_ms": cutoff_ms,
        "rows": len(companion),
        "registry": report["registry"],
        "sources": {rel: sha256_file(root / rel) for rel in
                    ("core/collection/provenance.py", "scripts/diagnose_crawl_seed_bias.py")},
        "files": {name: sha256_file(args.output_dir / name)
                  for name in ("crawl_provenance.csv", "report.json", "registry.json")},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"matches_db": report["database"]["matches"], "rows": len(companion),
                      "registry": report["registry"]}, indent=2))


if __name__ == "__main__":
    main()
