import pytest

pytestmark = pytest.mark.unit


from analytics.renderer import (
    render_answer,
    validate_narrative,
    validate_numeric_trace,
)


def test_every_deterministic_number_maps_to_evidence(sample_evidence):
    rendered = render_answer([sample_evidence], narrative_adapter=None)

    assert rendered.answer
    assert rendered.numeric_sources
    assert all(
        source.evidence_id == sample_evidence.evidence_id
        for source in rendered.numeric_sources
    )
    validate_numeric_trace(rendered.answer, rendered.numeric_sources)


def test_narrative_with_digits_is_rejected():
    accepted, code = validate_narrative(
        "L'effet est supérieur de 4 points.",
        allowed_evidence_ids={"ev-1"},
    )

    assert accepted is False
    assert code == "LLM_OUTPUT_REJECTED"


def test_narrative_with_unknown_evidence_reference_is_rejected():
    accepted, code = validate_narrative(
        "Effet cohérent selon [ev-9].",
        allowed_evidence_ids={"ev-1"},
    )

    assert accepted is False
    assert code == "LLM_OUTPUT_REJECTED"


def test_causal_narrative_is_rejected_even_with_known_evidence():
    accepted, code = validate_narrative(
        "La vision garantit la victoire [ev-1].",
        allowed_evidence_ids={"ev-1"},
    )

    assert accepted is False
    assert code == "LLM_OUTPUT_REJECTED"


def test_causal_paraphrases_are_rejected_even_with_known_evidence():
    for narrative in (
        "La vision augmente les chances de victoire [ev-1].",
        "Choisir la vision favorise la victoire [ev-1].",
    ):
        accepted, code = validate_narrative(
            narrative,
            allowed_evidence_ids={"ev-1"},
        )

        assert accepted is False
        assert code == "LLM_OUTPUT_REJECTED"


def test_uncited_factual_narrative_is_rejected():
    accepted, code = validate_narrative(
        "Le signal de vision paraît cohérent.",
        allowed_evidence_ids={"ev-1"},
    )

    assert accepted is False
    assert code == "LLM_OUTPUT_REJECTED"


def test_qualitative_narrative_with_known_evidence_is_allowed():
    accepted, code = validate_narrative(
        "Le signal paraît cohérent [ev-1].",
        allowed_evidence_ids={"ev-1"},
    )

    assert accepted is True
    assert code is None


def test_rejected_adapter_keeps_deterministic_answer(sample_evidence):
    class BadAdapter:
        def narrate(self, evidence):
            return "Je promets 99 pour cent."

    baseline = render_answer([sample_evidence], narrative_adapter=None)
    rendered = render_answer([sample_evidence], narrative_adapter=BadAdapter())

    assert rendered.answer == baseline.answer
    assert rendered.warnings == ({"code": "LLM_OUTPUT_REJECTED"},)
