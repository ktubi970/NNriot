# core/aligned_features.py
from core.features.pipeline import DraftFeaturePipeline
from core.features.matchup_matrix import LaneMatchupTracker
from core.serving_contract import FeatureContract

def extract_aligned_inputs(match, contract_version, *, macro=True, pipeline=None):
    actual = pipeline if pipeline is not None else DraftFeaturePipeline(
        matchup_tracker=LaneMatchupTracker())
    return FeatureContract(contract_version).extract(match, actual, macro=macro)
