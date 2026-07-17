"""Build the disposable analytical mart from the operational database."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile

from .config import AnalyticsConfig
from .db import connect_analytics, get_meta, initialize_schema, set_meta
from .extract import ExtractionError, ExtractionResult, extract_match
from .source import iter_source_matches, source_fingerprint, source_snapshot


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildReport:
    eligible_matches: int
    inserted_matches: int
    issue_counts: dict[str, int]
    unexplained_failures: int
    dataset_version: str
    quality_gate_passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _insert_result(conn, result: ExtractionResult) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO match_facts VALUES "
        "(:match_id,:region,:platform_prefix,:game_version,:patch,:queue_id,"
        ":game_creation_ms,:game_duration_s,:winner_team_id,:participant_count,"
        ":source_season_id)",
        result.match,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO team_facts VALUES "
        "(:match_id,:team_id,:side,:win,:kills,:deaths,:assists,:gold,"
        ":champion_damage,:vision_score,:lane_cs,:jungle_cs,:barons,:dragons,"
        ":towers,:inhibitors,:rift_heralds,:atakhan,:first_blood,:first_baron,"
        ":first_dragon,:first_tower,:first_inhibitor)",
        result.teams,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO player_facts VALUES "
        "(:match_id,:puuid,:team_id,:side,:win,:champion,:role,:kills,:deaths,"
        ":assists,:gold,:champion_damage,:vision_score,:lane_cs,:jungle_cs)",
        result.players,
    )


def _quality_gate(eligible: int, unexplained: int) -> bool:
    return unexplained <= eligible * 0.005


def _report(
    fingerprint: dict[str, str],
    inserted: int,
    issues: Counter[str],
    unexplained: int,
) -> BuildReport:
    eligible = int(fingerprint["source_match_count"])
    return BuildReport(
        eligible_matches=eligible,
        inserted_matches=inserted,
        issue_counts=dict(sorted(issues.items())),
        unexplained_failures=unexplained,
        dataset_version=fingerprint["dataset_version"],
        quality_gate_passed=_quality_gate(eligible, unexplained),
    )


def _write_metadata(
    conn,
    fingerprint: dict[str, str],
    report: BuildReport,
    config: AnalyticsConfig,
    mode: str,
) -> None:
    values = {
        **fingerprint,
        "extractor_version": config.extractor_version,
        "method_version": config.method_version,
        "build_mode": mode,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "eligible_matches": str(report.eligible_matches),
        "inserted_matches": str(report.inserted_matches),
        "unexplained_failures": str(report.unexplained_failures),
        "issue_counts": json.dumps(report.issue_counts, sort_keys=True),
        "quality_gate_passed": str(report.quality_gate_passed).lower(),
        "build_report": json.dumps(
            report.to_dict(), sort_keys=True, separators=(",", ":")
        ),
    }
    for key, value in values.items():
        set_meta(conn, key, value)


def _populate(
    source_path: str,
    conn,
    extractor_version: str,
    existing: set[str] | None = None,
    commit_every: int | None = None,
) -> tuple[int, Counter[str], int]:
    inserted = 0
    issue_counts: Counter[str] = Counter()
    unexplained = 0
    processed = 0
    existing = existing or set()
    for match_id, payload in iter_source_matches(source_path):
        if match_id in existing:
            continue
        processed += 1
        try:
            result = extract_match(match_id, payload)
            _insert_result(conn, result)
            inserted += 1
        except ExtractionError as exc:
            issue_counts[exc.code] += 1
            conn.execute(
                "INSERT OR IGNORE INTO extraction_issues VALUES (?,?,?,?)",
                (match_id, exc.code, exc.field, extractor_version),
            )
        except Exception:
            # Unexpected exceptions are deliberately not serialized into the mart:
            # their potentially sensitive payload belongs in application logs.
            unexplained += 1
            logger.exception("Unexpected analytics extraction failure for %s", match_id)
        if commit_every and processed % commit_every == 0:
            conn.commit()
    return inserted, issue_counts, unexplained


def _remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def _full_build(
    source_path: str,
    target: Path,
    config: AnalyticsConfig,
    fingerprint: dict[str, str],
) -> BuildReport:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        initialize_schema(str(temporary))
        with connect_analytics(str(temporary)) as conn:
            inserted, issues, unexplained = _populate(
                source_path, conn, config.extractor_version, commit_every=500
            )
            if source_snapshot(source_path) != _snapshot_from(fingerprint):
                raise RuntimeError("operational source changed during analytics build")
            report = _report(fingerprint, inserted, issues, unexplained)
            _write_metadata(conn, fingerprint, report, config, "full")
        if not report.quality_gate_passed:
            _remove_sqlite_files(temporary)
            return report
        os.replace(temporary, target)
        return report
    except Exception:
        _remove_sqlite_files(temporary)
        raise
    finally:
        _remove_sqlite_files(temporary)


def _incremental_build(
    source_path: str,
    target: Path,
    config: AnalyticsConfig,
    fingerprint: dict[str, str],
) -> BuildReport:
    if not target.exists():
        raise FileNotFoundError(
            f"analytics database does not exist for incremental update: {target}"
        )
    current_extractor = get_meta(str(target), "extractor_version")
    if current_extractor != config.extractor_version:
        raise ValueError(
            "extractor version mismatch; run a full analytics build"
        )

    with connect_analytics(str(target)) as conn:
        existing = {
            str(row[0]) for row in conn.execute("SELECT match_id FROM match_facts")
        }
        previous_dataset = conn.execute(
            "SELECT value FROM analytics_meta WHERE key='dataset_version'"
        ).fetchone()
        inserted, issues, unexplained = _populate(
            source_path, conn, config.extractor_version, existing
        )
        if source_snapshot(source_path) != _snapshot_from(fingerprint):
            raise RuntimeError("operational source changed during analytics update")
        report = _report(fingerprint, inserted, issues, unexplained)
        if not report.quality_gate_passed:
            conn.rollback()
            return report
        if previous_dataset is None or previous_dataset[0] != report.dataset_version:
            conn.execute("DELETE FROM evidence_cache")
        _write_metadata(conn, fingerprint, report, config, "incremental")
    return report


def build_analytics(
    source_path: str, analytics_path: str, incremental: bool = False
) -> BuildReport:
    """Build or update the analytical database without mutating the source."""

    config = AnalyticsConfig()
    initial_snapshot = source_snapshot(source_path)
    fingerprint = source_fingerprint(source_path, config.extractor_version)
    if initial_snapshot != _snapshot_from(fingerprint):
        raise RuntimeError("operational source changed while it was fingerprinted")
    target = Path(analytics_path)
    if incremental:
        return _incremental_build(source_path, target, config, fingerprint)
    return _full_build(source_path, target, config, fingerprint)


def _snapshot_from(fingerprint: dict[str, str]) -> dict[str, str]:
    return {
        key: fingerprint[key]
        for key in (
            "source_schema_version",
            "source_match_count",
            "source_max_creation_ms",
        )
    }
