"""Versioned pre-match input contracts and immutable serving artifact checks.

Tensor shape alone never establishes training provenance. Manifest null fields
mean unknown; they are deliberately not inferred from file modification times.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.calibration import sha256_file, load_calibration_artifact
from core.draft_features import ROLE_NAMES, _ROLE_ALIASES
from core.features.pipeline import DraftFeaturePipeline


class ContractError(ValueError):
    """An artifact or input cannot satisfy the declared serving contract."""


class FeatureUnavailableError(ContractError):
    """A required historical source cannot currently produce its features."""


CONTRACT_DIMS = {'nnriot.players14.v1': 14, 'nnriot.players18.v1': 18}
HISTORY_POLICY = 'result_end_strictly_before_start.v1'
FEATURE_SOURCE_FILES = ('core/features/pipeline.py', 'core/player_priors.py', 'core/champion_registry.py',
                        'core/features/static_meta.py', 'core/features/draft_composition.py', 'core/pre_match_history.py',
                        'core/features/matchup_matrix.py', 'core/pre_match_dataset.py',
                        'core/deepsets_layers.py', 'core/equivariant_model.py', 'core/draft_features.py',
                        'core/features/timeline_metrics.py', 'core/point_in_time_dataset.py',
                        'core/json_utils.py', 'core/collection/regions.py')


def runtime_feature_sources() -> dict:
    from core.config import PROJECT_ROOT
    return {name:sha256_file(PROJECT_ROOT/name) for name in FEATURE_SOURCE_FILES}


class FeatureContract:
    def __init__(self, version: str):
        if version not in CONTRACT_DIMS:
            raise ContractError(f'Unsupported feature contract {version!r}; spatial encoder provenance is required for 46D')
        self.version = version
        self.dim = CONTRACT_DIMS[version]
        self.prior_count = 14 if self.dim == 18 else 10

    def validate_model(self, model) -> None:
        shapes = {t.name.split(':')[0]: tuple(t.shape) for t in model.inputs}
        expected = {'team_a': (None, 5, self.dim), 'team_b': (None, 5, self.dim)}
        if 'macro_clash' in shapes:
            expected['macro_clash'] = (None, 15)
        if shapes != expected:
            raise ContractError(f'Checkpoint input shape/names {shapes} incompatible with {self.version}: {expected}')

    def validate_match(self, match: dict) -> None:
        if not isinstance(match, dict):
            raise ContractError('Match must be an object')
        if match.get('game_mode', 'CLASSIC') != 'CLASSIC':
            raise ContractError('Pre-match contract supports CLASSIC only')
        supplied = False
        identities = []
        for key in ('team_a', 'team_b'):
            players = match.get(key)
            if not isinstance(players, list) or len(players) != 5 or not all(isinstance(p, dict) for p in players):
                raise ContractError(f'{key} requires exactly five participants')
            roles = []
            for p in players:
                if p.get('puuid'):
                    identities.append(p['puuid'])
                role = str(p.get('role') or p.get('teamPosition') or '').strip().upper()
                roles.append(_ROLE_ALIASES.get(role, role))
                if not (p.get('champion') or p.get('championName')):
                    raise ContractError(f'{key} requires champion names (unknown names map explicitly to UNK)')
                if 'priors' in p:
                    supplied = True
                    if not isinstance(p['priors'], (list, tuple)):
                        raise ContractError('Priors must be a list or tuple, never an ignored tensor/array')
                    try:
                        a = np.asarray(p['priors'], dtype=float)
                    except (ValueError, TypeError) as exc:
                        raise ContractError('Priors must be numeric') from exc
                    if a.shape != (self.prior_count,) or not np.isfinite(a).all():
                        raise ContractError(f'{key} priors require exactly {self.prior_count} finite values; no padding or truncation')
                elif self.dim == 18:
                    raise ContractError('18D requires a result-end ordered timeline history provider; missing history')
            if sorted(roles) != sorted(ROLE_NAMES):
                raise ContractError(f'{key} roles must contain each canonical role exactly once')
        if len(identities) != len(set(identities)):
            raise ContractError('Each known player identity may appear only once in the match')
        if supplied:
            provenance = match.get('priors_provenance', {})
            if provenance.get('policy') != HISTORY_POLICY:
                raise ContractError('Supplied priors require verified result-end ordered history provenance')
            as_of = provenance.get('as_of_ms')
            last = provenance.get('latest_result_end_ms')
            if type(as_of) is not int or (last is not None and (type(last) is not int or last >= as_of)):
                raise ContractError('Historical results must end strictly before prediction start')
            if self.dim == 18 and (provenance.get('timeline_source_available') is not True or provenance.get('timeline_matches', 0) <= 0):
                raise FeatureUnavailableError('18D timeline history source unavailable; cannot replace missing source with cold defaults')

    def extract(self, match: dict, pipeline: DraftFeaturePipeline, *, macro: bool) -> dict[str, np.ndarray]:
        self.validate_match(match)
        if pipeline.player_tracker is not None and any('priors' not in p for k in ('team_a','team_b') for p in match[k]):
            raise ContractError('An undated mutable tracker cannot supply live priors; use the history provider')
        if pipeline.matchup_tracker.matchups or pipeline.matchup_tracker.champ_globals:
            raise ContractError('Contract v1 uses the declared neutral lane tracker; undated matchup state is incompatible')
        a, b = pipeline.extract_tensors(match, include_timeline=self.dim == 18)
        result = {'team_a': a[None], 'team_b': b[None]}
        if macro:
            result['macro_clash'] = pipeline.extract_macro_clash(match)[None]
        if any(not np.isfinite(v).all() for v in result.values()):
            raise ContractError('Feature extraction produced non-finite values')
        return result


def validate_probabilities(value, head: str, *, rows: int = 1) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if (arr.shape != (rows, 2) or not np.isfinite(arr).all() or
            (arr < 0).any() or (arr > 1).any() or not np.allclose(arr.sum(axis=1), 1, atol=1e-5)):
        raise ContractError(f'{head} must contain normalized finite probabilities with shape ({rows}, 2)')
    return arr


def _artifact_path(value: str, root: Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def load_model_manifest(path: str | Path, *, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    from core.config import PROJECT_ROOT
    p = Path(path).resolve()
    try:
        manifest = json.loads(p.read_text(encoding='utf-8'))
        if manifest['schema_version'] != 1 or not manifest['model_id']:
            raise ContractError('Unsupported or empty model manifest')
        FeatureContract(manifest['feature_contract'])
        checkpoint = _artifact_path(manifest['checkpoint']['path'], PROJECT_ROOT)
        if checkpoint_path is not None and checkpoint != Path(checkpoint_path).resolve():
            raise ContractError('Explicit checkpoint and manifest checkpoint disagree')
        if not checkpoint.is_file() or sha256_file(checkpoint) != manifest['checkpoint']['sha256']:
            raise ContractError(f'Checkpoint SHA-256 mismatch or missing file: {checkpoint}')
        manifest['_checkpoint_path'] = checkpoint
        manifest['_manifest_sha256'] = sha256_file(p)
        if manifest.get('runtime_feature_sources') != runtime_feature_sources():
            raise ContractError('Runtime feature source SHA-256 mismatch or missing source binding')
        validation = manifest.get('serving_validation', {})
        if validation.get('status') == 'validated':
            evidence = validation.get('independent_evaluation_report')
            if not isinstance(evidence, dict) or not evidence.get('path') or not evidence.get('sha256'):
                raise ContractError('Validated serving requires a hashed independent evaluation report')
            ep = _artifact_path(evidence['path'], PROJECT_ROOT)
            if not ep.is_file() or sha256_file(ep) != evidence['sha256']:
                raise ContractError('Independent evaluation report SHA-256 mismatch')
            edata = json.loads(ep.read_text(encoding='utf-8'))
            if (edata.get('status') != 'validated_for_serving' or edata.get('checkpoint_sha256') != manifest['checkpoint']['sha256']
                    or edata.get('feature_contract') != manifest['feature_contract']):
                raise ContractError('Independent evaluation report does not validate this checkpoint/contract')
        calibration = manifest.get('calibration')
        if calibration:
            cp = _artifact_path(calibration['path'], PROJECT_ROOT)
            if not cp.is_file() or sha256_file(cp) != calibration['sha256']:
                raise ContractError('Calibration artifact SHA-256 mismatch or missing file')
            if cp != checkpoint.with_suffix('.calibration.json'):
                raise ContractError('Calibration must be the checkpoint-bound adjacent artifact')
            data = load_calibration_artifact(checkpoint, input_dim=FeatureContract(manifest['feature_contract']).dim)
            if data is None:
                raise ContractError('Calibration failed schema/method/feature policy/split/fit verification')
            t = float(data['temperature'])
            if not np.isfinite(t) or t <= 0:
                raise ContractError('Calibration temperature must be positive and finite')
            if data.get('checkpoint_sha256') != manifest['checkpoint']['sha256']:
                raise ContractError('Calibration is not bound to this checkpoint SHA-256')
            if data.get('head') != 'winner' or data.get('class_order') != ['A', 'B'] or data.get('positive_class') != 1:
                raise ContractError('Calibration winner class semantics mismatch')
            if data.get('fit', {}).get('partition') != 'validation':
                raise ContractError('Calibration must be fitted on validation, never final test')
            split = checkpoint.with_suffix('.split.json')
            if not split.is_file() or sha256_file(split) != data.get('split_manifest_sha256'):
                raise ContractError('Calibration split manifest SHA-256 mismatch or missing')
            manifest['_temperature'] = t
        return manifest
    except ContractError:
        raise
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise ContractError(f'Invalid model manifest {p}: {exc}') from exc
