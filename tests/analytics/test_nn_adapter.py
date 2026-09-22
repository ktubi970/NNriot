import pytest

pytestmark = pytest.mark.unit


from analytics.nn_adapter import (
    CandidateValidationGate,
    DiscoveryCandidate,
    NNDiscoveryAdapter,
)


def test_current_checkpoint_is_always_labeled_discovery_only():
    adapter = NNDiscoveryAdapter(
        lambda rows: [{"metric": "gold", "score": 0.8}],
        checkpoint="model_v2.keras",
    )

    candidate = adapter.discover([{"match_id": "m1"}])[0]

    assert candidate.discovery_only is True
    assert candidate.checkpoint_independently_validated is False


def test_candidate_cannot_validate_on_discovery_matches():
    candidate = DiscoveryCandidate("gold", {}, ("m1", "m2"), True, False)
    gate = CandidateValidationGate(
        lambda request, excluded: {"allowed": True, "evidence_id": "ev"}
    )

    result = gate.validate(candidate, validation_match_ids={"m2", "m3"})

    assert result.allowed is False
    assert result.code == "DISCOVERY_VALIDATION_OVERLAP"


def test_candidate_only_returns_registered_engine_evidence():
    candidate = DiscoveryCandidate(
        "gold", {"region": "EUW"}, ("m1",), True, False
    )
    gate = CandidateValidationGate(
        lambda request, excluded: {"evidence_id": "ev-1", "allowed": True}
    )

    result = gate.validate(candidate, validation_match_ids={"m2", "m3"})

    assert result.allowed is True
    assert result.evidence_id == "ev-1"


def test_allowed_without_engine_evidence_id_is_rejected():
    candidate = DiscoveryCandidate("gold", {}, ("m1",), True, False)
    gate = CandidateValidationGate(
        lambda request, excluded: {"allowed": True, "evidence_id": None}
    )

    result = gate.validate(candidate, validation_match_ids={"m2"})

    assert result.allowed is False
    assert result.code == "ENGINE_EVIDENCE_REQUIRED"
