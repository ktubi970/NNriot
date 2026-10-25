"""September crawler (scripts/crawler_1m.py) left no crawl_queue rows, only crawler_frontier.json.

Its seeds (Challenger, then Grandmaster) were not recorded; only the visited set survives. This
check measures, on train/validation matches without any pre-entry anchor, how often the team with
more September-visited players wins, by split and month.

    python sept_frontier_check.py crawler_frontier.json DATASET_DIR crawl_provenance.csv
"""
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

frontier, dataset_dir, provenance = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
visited = set(json.load(open(frontier, encoding="utf-8"))["visited_puuids"])
anchored = {r["match_id"] for r in csv.DictReader(open(provenance, newline="", encoding="utf-8"))
            if r["has_pre_entry_anchor"] == "1"}
stats = defaultdict(lambda: [0, 0])
for split in ("train", "validation"):
    for row in json.loads((dataset_dir / f"{split}.json").read_text(encoding="utf-8")):
        if row["match_id"] in anchored:
            continue
        teams = row["feature_json"]
        diff = sum(p["puuid"] in visited for p in teams["team_b"]) - sum(p["puuid"] in visited for p in teams["team_a"])
        if not diff:
            continue
        b_won = row["labels_json"]["winner"] == 1
        month = datetime.fromtimestamp(row["game_creation_time"] / 1000, timezone.utc).strftime("%Y-%m")
        entry = stats[f"{split}:{month}"]
        entry[0] += 1
        entry[1] += b_won if diff > 0 else not b_won
print(json.dumps({"visited": len(visited), "more_sept_visited_team_wins": {
    k: {"n": n, "rate": round(w / n, 4)} for k, (n, w) in sorted(stats.items())}}, indent=2))
