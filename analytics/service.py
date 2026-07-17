"""Freshness-aware orchestration for controlled analytics chat."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import sqlite3

from .cohorts import AnalyticsAbstention
from .db import EvidenceRepository, all_meta
from .engine import AnalysisEngine
from .planner import FrenchPlanner
from .renderer import NarrativeAdapter, render_answer
from .source import source_snapshot
from .types import AnalysisRequest, Intent


@dataclass(frozen=True)
class ChatResponse:
    success: bool
    answer: str
    analysis_request: dict
    evidence: tuple[dict, ...]
    warnings: tuple[dict, ...]
    abstained: bool
    dataset_version: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class AnalyticsService:
    def __init__(
        self,
        analytics_path: str,
        source_path: str = "training_data.db",
        narrative_adapter: NarrativeAdapter | None = None,
    ):
        self.analytics_path = analytics_path
        self.source_path = source_path
        self.planner = FrenchPlanner()
        self.engine = AnalysisEngine(analytics_path)
        self.narrative_adapter = narrative_adapter

    def health(self) -> dict:
        try:
            metadata = all_meta(self.analytics_path)
            freshness = self.freshness()
            return {
                "ok": bool(metadata) and freshness["fresh"],
                **metadata,
                **freshness,
            }
        except (OSError, sqlite3.Error, ValueError):
            return {"ok": False, "code": "ANALYTICS_DB_STALE"}

    def freshness(self) -> dict:
        metadata = all_meta(self.analytics_path)
        source = source_snapshot(self.source_path)
        fresh = all(metadata.get(key) == value for key, value in source.items())
        return {
            "fresh": fresh,
            "dataset_version": metadata.get("dataset_version"),
            "source_snapshot": source,
        }

    def get_evidence(self, evidence_id: str) -> dict | None:
        return EvidenceRepository(self.analytics_path).get_by_id(evidence_id)

    def chat(
        self,
        message: str,
        context: dict | None,
        filters: dict,
    ) -> ChatResponse:
        dataset_version: str | None = None
        try:
            metadata = all_meta(self.analytics_path)
            dataset_version = metadata.get("dataset_version")
            if not self.freshness()["fresh"]:
                raise AnalyticsAbstention("ANALYTICS_DB_STALE")
            request = self.planner.plan(message, context, filters)
            execution_request = (
                AnalysisRequest.from_dict(context)
                if request.intent in {Intent.SHOW_EVIDENCE, Intent.SHOW_LIMITATIONS}
                else request
            )
            packets = self.engine.run(execution_request, question=message)
            rendered = render_answer(packets, self.narrative_adapter)
            return ChatResponse(
                success=True,
                answer=rendered.answer,
                analysis_request=execution_request.to_dict(),
                evidence=tuple(packet.to_dict() for packet in packets),
                warnings=rendered.warnings,
                abstained=False,
                dataset_version=dataset_version,
            )
        except AnalyticsAbstention as exc:
            return ChatResponse(
                success=True,
                answer="Analyse non publiée : " + exc.code,
                analysis_request={},
                evidence=(),
                warnings=({"code": exc.code, "details": exc.details},),
                abstained=True,
                dataset_version=dataset_version,
            )
        except (OSError, sqlite3.Error):
            return ChatResponse(
                success=True,
                answer="Analyse non publiée : ANALYTICS_DB_STALE",
                analysis_request={},
                evidence=(),
                warnings=(
                    {"code": "ANALYTICS_DB_STALE", "details": {}},
                ),
                abstained=True,
                dataset_version=None,
            )
