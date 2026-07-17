"""Deterministic French rendering with character-level numeric provenance."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Protocol

from .types import EvidencePacket


NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?%?")
EVIDENCE_REF = re.compile(r"\[([A-Za-z0-9-]{4,64})\]")
COMPARATIVE = re.compile(
    r"\b(superieur|inferieur|meilleur|pire|davantage|plus que|moins que)\b",
    re.I,
)
CAUSAL_OR_PREDICTIVE = re.compile(
    r"\b(cause|causal|garantit|garantie|prouve|certain|assure|"
    r"entraine|fait gagner|permet de gagner|predit|prediction|va gagner)\b",
    re.I,
)


class NarrativeAdapter(Protocol):
    def narrate(self, evidence: list[dict]) -> str: ...


@dataclass(frozen=True)
class NumericSource:
    token: str
    start: int
    end: int
    evidence_id: str


@dataclass(frozen=True)
class RenderedAnswer:
    answer: str
    numeric_sources: tuple[NumericSource, ...]
    warnings: tuple[dict, ...]


def _normalized(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.lower())
        if not unicodedata.combining(character)
    )


def validate_narrative(
    text: str, allowed_evidence_ids: set[str]
) -> tuple[bool, str | None]:
    if not isinstance(text, str):
        return False, "LLM_OUTPUT_REJECTED"
    references = set(EVIDENCE_REF.findall(text))
    if references - allowed_evidence_ids:
        return False, "LLM_OUTPUT_REJECTED"
    without_references = EVIDENCE_REF.sub("", text)
    normalized = _normalized(without_references)
    if NUMBER.search(without_references) or COMPARATIVE.search(
        normalized
    ) or CAUSAL_OR_PREDICTIVE.search(normalized):
        return False, "LLM_OUTPUT_REJECTED"
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]
    if any(not EVIDENCE_REF.search(sentence) for sentence in sentences):
        return False, "LLM_OUTPUT_REJECTED"
    return True, None


def _format_number(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def _format_packet(
    packet: EvidencePacket,
) -> tuple[str, tuple[NumericSource, ...]]:
    estimate = packet.estimate
    sample = packet.sample
    evidence_id = packet.evidence_id
    strength = packet.recommendation.get("strength", "none")
    if strength == "strong":
        opening = "Association robuste observée"
    elif strength == "moderate":
        opening = "Association observée"
    else:
        opening = "Résultat descriptif incertain"

    clauses = [f"[{evidence_id}] {opening} sur {sample['matches']} matchs"]
    value = estimate.get("value")
    if value is None:
        clauses.append("écart non disponible")
    else:
        clauses.append(
            "écart de "
            + _format_number(value, 2)
            + f" {estimate.get('unit', '')}".rstrip()
        )
    interval = estimate.get("confidence_interval_95")
    if interval is not None:
        clauses.append(
            "IC à 95 % de "
            + _format_number(interval[0], 2)
            + " à "
            + _format_number(interval[1], 2)
        )
    q_value = estimate.get("q_value")
    if q_value is not None:
        clauses.append("q=" + _format_number(q_value, 3))
    text = "; ".join(clauses) + ". "
    text += "Cette mesure décrit une association rétrospective, pas un effet causal."
    if packet.limitations:
        text += " Limites : " + ", ".join(packet.limitations) + "."
    sources = tuple(
        NumericSource(match.group(), match.start(), match.end(), evidence_id)
        for match in NUMBER.finditer(text)
    )
    return text, sources


def validate_numeric_trace(
    answer: str, sources: tuple[NumericSource, ...]
) -> None:
    expected = {
        (match.start(), match.end(), match.group()) for match in NUMBER.finditer(answer)
    }
    actual = {(source.start, source.end, source.token) for source in sources}
    if expected != actual:
        raise ValueError("UNTRACED_NUMERIC_TOKEN")


def _narrative_sources(text: str, offset: int) -> tuple[NumericSource, ...]:
    references = list(EVIDENCE_REF.finditer(text))
    sources: list[NumericSource] = []
    for number in NUMBER.finditer(text):
        evidence_id = next(
            (
                reference.group(1)
                for reference in references
                if reference.start() <= number.start()
                and number.end() <= reference.end()
            ),
            None,
        )
        if evidence_id is None:
            raise ValueError("UNTRACED_NUMERIC_TOKEN")
        sources.append(
            NumericSource(
                number.group(),
                offset + number.start(),
                offset + number.end(),
                evidence_id,
            )
        )
    return tuple(sources)


def render_answer(
    packets: list[EvidencePacket] | tuple[EvidencePacket, ...],
    narrative_adapter: NarrativeAdapter | None = None,
) -> RenderedAnswer:
    strength_order = {"strong": 2, "moderate": 1, "none": 0}
    ordered = sorted(
        packets,
        key=lambda packet: (
            strength_order.get(packet.recommendation.get("strength", "none"), 0),
            abs(float(packet.estimate.get("effect_size") or 0.0)),
        ),
        reverse=True,
    )
    sections: list[str] = []
    sources: list[NumericSource] = []
    offset = 0
    separator = "\n\n"
    for packet in ordered:
        text, local_sources = _format_packet(packet)
        sections.append(text)
        sources.extend(
            NumericSource(
                source.token,
                source.start + offset,
                source.end + offset,
                source.evidence_id,
            )
            for source in local_sources
        )
        offset += len(text) + len(separator)
    deterministic = separator.join(sections)
    warnings: list[dict] = []
    answer = deterministic

    if narrative_adapter is not None:
        try:
            narrative = narrative_adapter.narrate(
                [packet.to_dict() for packet in ordered]
            )
            accepted, code = validate_narrative(
                narrative, {packet.evidence_id for packet in ordered}
            )
            if accepted and narrative.strip():
                prefix = "\n\nInterprétation qualitative : "
                narrative_offset = len(answer) + len(prefix)
                answer += prefix + narrative.strip()
                sources.extend(_narrative_sources(narrative.strip(), narrative_offset))
            elif not accepted:
                warnings.append({"code": code})
        except Exception:
            warnings.append({"code": "LLM_OUTPUT_REJECTED"})

    numeric_sources = tuple(sources)
    validate_numeric_trace(answer, numeric_sources)
    return RenderedAnswer(answer, numeric_sources, tuple(warnings))
