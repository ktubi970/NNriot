"""One-off, bounded operational collection using the existing NNriot service."""
import dataclasses
from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from core import config, riot_api
from core.collection import CollectionService

DB = Path(config.get_db_path(str(ROOT / "training_data.db"), require_exists=True))
TARGET = sys.maxsize  # Duration, not a match count, ends this campaign.
MAX_SECONDS = 20 * 3600
stop = threading.Event()
finished = threading.Event()
auth_failed = threading.Event()
reason = []
started = time.monotonic()
started_at = datetime.now(timezone.utc)
deadline_at = started_at + timedelta(seconds=MAX_SECONDS)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enrichment")


def write_json(name, data):
    path = RUN / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def connect():
    conn = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    return conn


def counts():
    with connect() as conn:
        return {table: conn.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                for table in ("matches", "training_dataset", "match_participants", "match_timelines")}


class GuardedAPI(riot_api.RiotAPI):
    def _make_request(self, url, max_retries=3):
        if auth_failed.is_set():
            raise riot_api.RiotAuthError("Collection stopped after authentication failure")
        try:
            return super()._make_request(url, max_retries=max_retries)
        except riot_api.RiotAuthError:
            auth_failed.set()
            stop.set()
            raise


class GuardedCollection(CollectionService):
    def collect_player_matches(self, *args, **kwargs):
        result = super().collect_player_matches(*args, **kwargs)
        if auth_failed.is_set():
            raise RuntimeError("Riot authentication failed; current queue entry retained for retry")
        log.info("Batch: matches=%s training=%s timelines=%s spatial_matches=%s errors=%s",
                 result.matches_saved, result.records_saved, result.timelines_saved,
                 result.embeddings_computed, result.errors)
        return result


def monitor(service):
    try:
        _monitor(service)
    except Exception:
        reason[:] = ["monitor_error"]
        stop.set()
        log.exception("Monitoring failed; stopping collection safely")


def _monitor(service):
    while not finished.wait(15):
        elapsed = time.monotonic() - started
        free = shutil.disk_usage(DB.parent).free
        if not reason and (RUN / "STOP").exists():
            reason[:] = ["stop_file"]
            stop.set()
        elif not reason and elapsed >= MAX_SECONDS:
            reason[:] = ["duration_limit"]
            stop.set()
        elif not reason and free < 10 * 2**30:
            reason[:] = ["disk_reserve"]
            stop.set()
        status = service.tracker.to_dict()
        status.pop("player", None)
        status.update(total=None, eta_minutes=None, remaining_seconds=max(0, MAX_SECONDS-elapsed),
                      duration_hours=20, deadline_at=deadline_at.isoformat())
        status.update(pid=os.getpid(), database=str(DB), updated_at=datetime.now(timezone.utc).isoformat(),
                      free_gib=round(free / 2**30, 2), stop_reason=reason)
        write_json("progress.json", status)


def validate_new_rows(first_rowid):
    with connect() as conn:
        ids = [row[0] for row in conn.execute("SELECT match_id FROM matches WHERE rowid > ?", (first_rowid,))]
        checks = dict(new_matches=len(ids), training_records=0, participants=0, timelines_nonnull=0,
                      spatial_rows=0, missing_training=[], missing_timelines=[], invalid_participant_count=[])
        for mid in ids:
            training = conn.execute("SELECT COUNT(*) FROM training_dataset WHERE match_id=?", (mid,)).fetchone()[0]
            participants = conn.execute("SELECT COUNT(*) FROM match_participants WHERE match_id=?", (mid,)).fetchone()[0]
            timeline = conn.execute("SELECT COUNT(*) FROM match_timelines WHERE match_id=? AND timeline_json IS NOT NULL", (mid,)).fetchone()[0]
            spatial = conn.execute("SELECT COUNT(*) FROM player_spatial_embeddings WHERE match_id=?", (mid,)).fetchone()[0]
            checks["training_records"] += training
            checks["participants"] += participants
            checks["timelines_nonnull"] += timeline
            checks["spatial_rows"] += spatial
            if not training:
                checks["missing_training"].append(mid)
            if not timeline:
                checks["missing_timelines"].append(mid)
            if participants != 10:
                checks["invalid_participant_count"].append(mid)
        write_json("new_match_ids.json", ids)
        return checks


def main():
    if (RUN / "STOP").exists() or shutil.disk_usage(DB.parent).free < 10 * 2**30:
        raise RuntimeError("STOP file present or insufficient free disk space")
    log.info("Reading initial database counters (spatial validation is restricted to new matches)")
    with connect() as conn:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        first_rowid = conn.execute("SELECT MAX(rowid) FROM matches").fetchone()[0] or 0
    if version != 14:
        raise RuntimeError("Expected existing schema v14; no migration authorized by this runner")
    baseline = counts()
    write_json("baseline.json", dict(database=str(DB), schema_version=version, counts=baseline,
               first_rowid=first_rowid, pid=os.getpid(), target_new_matches=None,
               deadline_at=deadline_at.isoformat(),
               max_hours=20, regions=["EUW", "KR", "NA"], days_back=14,
               started_at=started_at.isoformat()))
    log.info("Starting twenty-hour enrichment: database=%s baseline=%s deadline=%s", DB, baseline, deadline_at.isoformat())
    service = GuardedCollection(db_path=str(DB), api_factory=GuardedAPI)
    thread = threading.Thread(target=monitor, args=(service,), daemon=True)
    thread.start()
    result = None
    error = None
    try:
        result = service.run_crawler(target_matches=TARGET, regions=["EUW", "KR", "NA"],
                                     days_back=14, batch_size=10, stop_event=stop)
    except Exception as exc:
        error = str(exc)
        log.exception("Collection stopped with error")
    finally:
        finished.set()
        thread.join(timeout=20)
    after = counts()
    validation = validate_new_rows(first_rowid)
    outcome = "completed" if "duration_limit" in reason else "stopped"
    if error or auth_failed.is_set() or "monitor_error" in reason:
        outcome = "error"
    summary = dict(outcome=outcome, error=error, auth_failed=auth_failed.is_set(), stop_reason=reason,
                   elapsed_seconds=round(time.monotonic()-started, 2), duration_hours=20,
                   started_at=started_at.isoformat(), deadline_at=deadline_at.isoformat(),
                   before=baseline, after=after,
                   delta={key: after[key]-value for key, value in baseline.items()}, validation=validation)
    if result is not None:
        summary["service_result"] = {k: v for k, v in dataclasses.asdict(result).items() if k != "discovered_puuids"}
    write_json("result.json", summary)
    write_json("progress.json", dict(status=outcome, current=summary["delta"]["matches"], total=None,
               database=str(DB), pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat()))
    log.info("Finished: %s", json.dumps(summary))
    return 1 if outcome == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())

