"""Freeze a recent cohort and compare candidate checkpoints without promotion.

Run with -m scripts.compare_served_candidates. A frozen replay uses the saved
compressed source snapshot, not a changing database. No fitting uses final test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

import numpy as np

from core import config
from core.calibration import sha256_file, apply_temperature_to_probabilities
from core.candidate_inventory import inventory_candidates, serving_manifest
from core.common_cohort_evaluation import compare_on_intersection
from core.features.pipeline import DraftFeaturePipeline
from core.features.matchup_matrix import LaneMatchupTracker
from core.pre_match_history import read_recent_rows, decode_history_row, ResultOrderedHistory, build_target_history
from core.serving_contract import ContractError, FeatureContract, load_model_manifest, validate_probabilities, runtime_feature_sources


def evaluation_runtime_sources():
    files = ('core/serving_contract.py', 'core/candidate_inventory.py', 'core/common_cohort_evaluation.py',
             'core/evaluation_plots.py', 'core/calibration.py', 'core/training_metrics.py',
             'core/selective_classification.py', 'core/config.py', 'scripts/compare_served_candidates.py',
             'scripts/train_tree_baseline.py')
    return runtime_feature_sources() | {name: sha256_file(config.PROJECT_ROOT/name) for name in files}


def runtime_environment():
    from importlib.metadata import version, PackageNotFoundError
    dependencies = {}
    for name in ('numpy', 'tensorflow', 'tensorflow-intel', 'keras', 'scikit-learn', 'scipy', 'joblib', 'plotly'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'dependencies': dependencies}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False)+'\n', encoding='utf-8')


def iso(ms):
    return datetime.fromtimestamp(ms/1000, timezone.utc).isoformat() if ms else None


def freeze_cohort(rows, history_size, validation_size, test_size):
    if any(type(n) is not int or n < 1 for n in (history_size,validation_size,test_size)):
        raise ValueError('History, validation and test sizes must be positive integers')
    if len(rows) < validation_size + test_size + 1:
        raise ValueError('Insufficient rows for separate history, validation and test')
    test_start = len(rows)-test_size
    val_start = test_start-validation_size
    cohort = []
    for index, row in enumerate(rows):
        part = 'history' if index < val_start else 'validation' if index < test_start else 'test'
        label = row['winner_label']
        if label not in (0, 1):
            label = None
        try:
            rec = decode_history_row(row)
            patch = rec['patch']
            end = rec['result_end_ms']
            blue = {bool(p.get('win')) for p in rec['participants'] if p.get('teamId')==100}
            red = {bool(p.get('win')) for p in rec['participants'] if p.get('teamId')==200}
            if len(blue)!=1 or len(red)!=1 or blue==red or label!=int(not next(iter(blue))):
                label=None
        except ContractError:
            patch, end = 'UNKNOWN', None
        cohort.append({'match_id': row['match_id'], 'game_creation_time': row['game_creation_time'],
                       'date_utc': iso(row['game_creation_time']), 'patch': patch,
                       'winner_label': label, 'result_end_ms': end, 'partition': part})
    return cohort


def extract_common_features(rows, cohort):
    features = {14: {}, 18: {}}
    errors = {14: {}, 18: {}}
    facts = []
    for row in rows:
        try:
            facts.append(decode_history_row(row))
        except ContractError:
            facts.append(None)
    metadata = {r['match_id']:r for r in cohort}
    time_keys = [r['game_creation_time'] for r in rows]
    import bisect
    evaluated = 0
    for index,row in enumerate(rows):
        mid=row['match_id']
        if metadata[mid]['partition']=='history':
            continue
        try:
            rec=facts[index]
            if rec is None:
                raise ContractError('Invalid Riot source payload')
            clean=rec['feature_json']
            if metadata[mid]['winner_label'] not in (0,1):
                raise ContractError('Training label missing or disagrees with Riot teamId perspective')
            start=row['game_creation_time']
            before=bisect.bisect_left(time_keys,start)
            window=facts[max(0,before-10000):before]
            history=build_target_history(window,clean,as_of_ms=start)
            for dim,version in ((14,'nnriot.players14.v1'),(18,'nnriot.players18.v1')):
                try:
                    enriched=history.enrich(clean,include_timeline=dim==18,as_of_ms=start)
                    enriched['priors_provenance']['history_window_limit']=10000
                    FeatureContract(version).validate_match(enriched)
                    features[dim][mid]=enriched
                except ContractError as exc:
                    errors[dim][mid]=f'{type(exc).__name__}: {exc}'
        except (ContractError,ValueError,TypeError,KeyError) as exc:
            for dim in (14,18):errors[dim][mid]=f'{type(exc).__name__}: {exc}'
        evaluated+=1
        if evaluated%250==0:print(f'Shared live-policy replay {evaluated} evaluation matches',flush=True)
    return features,errors,{'policy':'same build_target_history as DatabaseHistoryProvider',
                           'history_window_limit':10000,'prediction_cutoff':'game_creation_time',
                           'historical_training_labels_required':False,
                           'availability_assumption':'Riot results observable at result end; actual ingestion times unavailable'}


def independent_eligibility(item, test):
    bounds = item.get('training', {}).get('bounds_ms')
    reasons = list(item.get('limitations', []))
    if not bounds:
        reasons.append('Training, validation and earlier test time bounds are unknown')
    elif max(v['max_ms'] for v in bounds.values()) >= min(r['game_creation_time'] for r in test):
        reasons.append('Recent cohort is not strictly later than all recorded training/selection partitions')
    # Existing artifacts do not bind the historical feature-generation code and
    # acquisition timing to their checkpoint. Never infer eligibility from dates.
    reasons.append('No checkpoint-bound attestation of result-end ordered training priors and encoder provenance')
    return {'eligible': False, 'reasons': reasons, 'classification': 'exploratory'}


def run(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.replay:
        source = Path(args.replay).resolve()
        protocol = json.loads((source/'protocol.json').read_text(encoding='utf-8'))
        for filename, expected in protocol['frozen_files'].items():
            if sha256_file(source/filename) != expected:
                raise ValueError(f'Frozen replay SHA-256 mismatch: {filename}')
        bound_runtime = protocol.get('runtime_sources_sha256')
        if bound_runtime is not None and bound_runtime != evaluation_runtime_sources():
            raise ContractError('Frozen evaluation runtime source SHA-256 mismatch; use the recorded source version')
        if bound_runtime is None:
            print('Legacy frozen source has no runtime binding; recording current runtime for this exploratory run.', flush=True)
        rows = json.loads(gzip.decompress((source/'source_snapshot.json.gz').read_bytes()))
        cohort = [json.loads(line) for line in (source/'cohort.jsonl').read_text(encoding='utf-8').splitlines()]
        inventory = json.loads((source/'inventory.json').read_text(encoding='utf-8'))
        db_info = protocol['source_database']
    else:
        db_path = config.get_db_path(args.db, require_exists=True)
        print(f'Reading frozen source window from {db_path}', flush=True)
        rows = read_recent_rows(db_path, args.history_size+args.validation_size+args.test_size)
        cohort = freeze_cohort(rows, args.history_size, args.validation_size, args.test_size)
        inventory = inventory_candidates()
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(Path(db_path).as_uri()+'?mode=ro', uri=True)) as conn:
            schema = conn.execute('SELECT MAX(version) FROM schema_version').fetchone()[0]
        db_info = {'path': db_path, 'schema_version': schema, 'snapshot_policy': 'read-only single SELECT source window; frozen bytes stored',
                   'window_requested': args.history_size+args.validation_size+args.test_size}
    (out/'source_snapshot.json.gz').write_bytes(gzip.compress(json.dumps(rows, separators=(',', ':')).encode(), mtime=0))
    (out/'cohort.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in cohort), encoding='utf-8')
    write_json(out/'inventory.json', inventory)
    frozen = {name: sha256_file(out/name) for name in ('source_snapshot.json.gz', 'cohort.jsonl', 'inventory.json')}
    test = [r for r in cohort if r['partition'] == 'test']
    validation = [r for r in cohort if r['partition'] == 'validation']
    cutoff = min(r['game_creation_time'] for r in test)
    prior_rows = [r for r in cohort if r['partition'] != 'test' and r['winner_label'] in (0, 1)
                  and r['result_end_ms'] is not None and r['result_end_ms'] < cutoff]
    if not prior_rows:
        raise ValueError('No completed pre-test outcomes for Blue Side baseline')
    prior_red = sum(r['winner_label'] for r in prior_rows)/len(prior_rows)
    eligibility = {i['model_id']: independent_eligibility(i, test) for i in inventory}
    protocol = {'schema_version': 1, 'classification': 'exploratory; no validated winner or automatic promotion',
                'source_database': db_info, 'frozen_files': frozen,
                'inclusion': 'latest CLASSIC creation_time>0 window; invalid labels/roles/payloads retained and counted as feature failures',
                'partitions': {k: {'count': sum(r['partition']==k for r in cohort),
                                  'start_utc': min(r['date_utc'] for r in cohort if r['partition']==k),
                                  'end_utc': max(r['date_utc'] for r in cohort if r['partition']==k)} for k in ('history','validation','test')},
                'calibration_policy': 'Existing checkpoint-bound validation calibration only. No fitting, thresholds, blending or selection on recent test.',
                'blue_prior': {'red_probability': prior_red, 'blue_probability': 1-prior_red, 'completed_pre_test_count': len(prior_rows), 'cutoff_ms': cutoff},
                'eligibility': eligibility, 'seed': 20261009,
                'runtime_sources_sha256': evaluation_runtime_sources(),
                'runtime_environment': runtime_environment(),
                'pipeline_policy': '14D/18D exact source layouts, result-end ordered player priors, neutral lane tracker matching legacy training defaults; current static champion profiles, historical patch provenance unknown',
                'thresholds_frozen': [0,.02,.04,.06,.08,.10,.15,.20],
                'reference_decision': 'Keep candidate_deepsets_v3 configured; legacy, uncalibrated and not validated for actionable serving'}
    write_json(out/'protocol.json', protocol)  # Freeze before reading any test inference.
    features, feature_errors, history_report = extract_common_features(rows, cohort)
    predictions = {'coinflip50:raw': {}, 'blue_prior:raw': {}}
    failures = {}
    per_match = []
    for r in test:
        mid = r['match_id']
        if mid in features[14] and r['winner_label'] in (0, 1):
            predictions['coinflip50:raw'][mid] = .5
            predictions['blue_prior:raw'][mid] = prior_red
    import tensorflow as tf
    from core.equivariant_model import NNRIOT_CUSTOM_OBJECTS
    manifests = out/'manifests'; manifests.mkdir()
    for item in inventory:
        name = item['model_id']
        if args.candidates and name not in args.candidates.split(',') and item.get('kind') != 'tree':
            continue
        contract_version = item.get('feature_contract')
        if contract_version not in ('nnriot.players14.v1', 'nnriot.players18.v1', 'legacy.tree24.v1'):
            key = name + ':raw'
            reason = 'Unsupported serving contract or missing spatial encoder provenance'
            predictions[key] = {}
            failures[key] = Counter({reason: len(test)})
            per_match += [{'model': key, 'match_id': r['match_id'], 'winner_label': r['winner_label'],
                           'probability_team_b': None, 'status': 'contract_excluded', 'reason': reason} for r in test]
            continue
        print(f'Inference: {name}', flush=True)
        raw_key = name+':raw'; predictions[raw_key] = {}; failures[raw_key] = Counter()
        try:
            cp = config.PROJECT_ROOT/item['checkpoint']['path']
            if sha256_file(cp) != item['checkpoint']['sha256']:
                raise ContractError('Frozen inventory checkpoint SHA-256 changed')
            if item['kind'] == 'tree':
                import joblib
                from scripts.train_tree_baseline import extract_tabular_features
                bundle = joblib.load(cp)
                model, encoder = bundle['model'], bundle.get('encoder')
                usable = [r for r in test if r['match_id'] in features[14]]
                values_names = [extract_tabular_features(features[14][r['match_id']]) for r in usable]
                if not values_names or any(names != bundle['feature_names'] for _, names in values_names) or encoder is None:
                    raise ContractError('Existing tree baseline requires its exact legacy 24-column encoder')
                x = encoder.transform(np.asarray([v for v, _ in values_names], dtype=object))
                p = model.predict_proba(x)[:, list(model.classes_).index(1)]
                calibrated = None
            else:
                manifest = serving_manifest(item)
                path = manifests/(name+'.json'); write_json(path, manifest)
                checked = load_model_manifest(path, checkpoint_path=cp)
                contract = FeatureContract(contract_version)
                model = tf.keras.models.load_model(cp, custom_objects=NNRIOT_CUSTOM_OBJECTS, compile=False)
                contract.validate_model(model)
                pipeline = DraftFeaturePipeline(matchup_tracker=LaneMatchupTracker())
                macro = any(t.name.split(':')[0]=='macro_clash' for t in model.inputs)
                usable = [r for r in test if r['match_id'] in features[contract.dim]]
                tensors = [contract.extract(features[contract.dim][r['match_id']], pipeline, macro=macro) for r in usable]
                if not tensors:
                    raise ContractError('No feature-evaluable test matches')
                x = {k: np.concatenate([t[k] for t in tensors]) for k in tensors[0]}
                # Direct eager batches avoid tf.data thread pools; same inference graph as facade.
                parts = [np.asarray(model({k:v[s:s+128] for k,v in x.items()}, training=False)['winner']) for s in range(0,len(usable),128)]
                p = validate_probabilities(np.concatenate(parts), 'winner', rows=len(usable))[:,1]
                calibrated = apply_temperature_to_probabilities(p, checked['_temperature']) if '_temperature' in checked else None
            for r, prob in zip(usable, p):
                mid = r['match_id']; predictions[raw_key][mid] = float(prob)
                per_match.append({'model': raw_key, 'match_id': mid, 'winner_label': r['winner_label'], 'probability_team_b': float(prob), 'status':'ok'})
            if calibrated is not None:
                cal_key = name+':calibrated'; predictions[cal_key] = {r['match_id']:float(prob) for r,prob in zip(usable,calibrated)}
                failures[cal_key] = Counter()
                per_match += [{'model':cal_key,'match_id':r['match_id'],'winner_label':r['winner_label'],'probability_team_b':float(prob),'status':'ok'} for r,prob in zip(usable,calibrated)]
            del model
            tf.keras.backend.clear_session()
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            failures[raw_key][reason] += len(test)
            per_match += [{'model':raw_key,'match_id':r['match_id'],'winner_label':r['winner_label'],'probability_team_b':None,'status':'inference_error','reason':reason} for r in test]
            print(f'Excluded {name}: {reason}', flush=True)
    for key, probs in predictions.items():
        if not probs:
            continue
        failures.setdefault(key, Counter())
        for r in test:
            mid = r['match_id']
            if mid not in probs:
                dim = 18 if 'v4' in key else 14
                reason = feature_errors[dim].get(mid, 'invalid_label_or_features')
                failures[key][reason] += 1
                per_match.append({'model':key,'match_id':mid,'winner_label':r['winner_label'],'probability_team_b':None,'status':'feature_error','reason':reason})
            elif key in ('coinflip50:raw','blue_prior:raw'):
                per_match.append({'model':key,'match_id':mid,'winner_label':r['winner_label'],'probability_team_b':probs[mid],'status':'ok'})
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in per_match), encoding='utf-8')
    print('Calculating common metrics and paired bootstrap...', flush=True)
    metrics = compare_on_intersection(test, predictions, failures)
    contract_by_id = {item['model_id']: item.get('feature_contract') for item in inventory}
    for name, coverage in metrics['coverage'].items():
        version = contract_by_id.get(name.rsplit(':', 1)[0], 'nnriot.players14.v1')
        dim = 18 if version == 'nnriot.players18.v1' else 14
        supported = version in ('nnriot.players14.v1', 'nnriot.players18.v1', 'legacy.tree24.v1')
        coverage['feature_count'] = sum(r['match_id'] in features[dim] for r in test) if supported else 0
        coverage['feature_coverage'] = coverage['feature_count']/len(test)
    for name, score in metrics['metrics'].items():
        for row in score['risk_coverage']:
            row['coverage_initial_cohort'] = row['count']/len(test)
    metrics.update({'classification':protocol['classification'], 'feature_failures':{str(dim):dict(Counter(err.values())) for dim,err in feature_errors.items()},
                    'history_replay':history_report, 'cohort_sha256':frozen['cohort.jsonl'],
                    'test_period_utc':[test[0]['date_utc'],test[-1]['date_utc']],
                    'artifacts_sha256':{'predictions.jsonl':sha256_file(out/'predictions.jsonl')}})
    write_json(out/'metrics.json', metrics)
    report = ['# Common recent cohort — exploratory comparison', '', protocol['classification'], '',
              f"Test: {len(test)} initial matches, {metrics['intersection_count']} common evaluable matches; {test[0]['date_utc']} to {test[-1]['date_utc']}.",
              f"Cohort SHA-256: `{frozen['cohort.jsonl']}`.", '',
              '| Candidate / probability | Initial coverage | Accuracy | AUC | Log loss | Brier | ECE |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, score in metrics['metrics'].items():
        coverage = metrics['coverage'][name]['inference_coverage']
        auc = score['roc_auc']
        report.append(f"| {name} | {coverage:.1%} | {score['accuracy']:.4f} | {auc:.4f} | {score['log_loss']:.4f} | {score['brier']:.4f} | {score['calibration_error']:.4f} |")
    report += ['', 'Candidates with zero initial inference coverage:', '']
    for name, coverage in metrics['coverage'].items():
        if coverage['inference_count'] == 0:
            report.append(f"- `{name}`: 0/{len(test)}; " + '; '.join(coverage['exclusion_causes']))
    report += ['', 'The configured reference remains candidate_deepsets_v3. No model is declared superior or ready for actionable serving.', '',
               'Risk/coverage, equal conviction thresholds, Wilson intervals, and paired chronological block bootstrap intervals are in metrics.json. Predictions and failures remain visible on the initial cohort; no failed feature row is silently dropped.', '',
               'Training provenance: see protocol.json eligibility per candidate. Split dates alone do not prove training feature availability. v3 lacks a checkpoint-bound split and calibration. Spatial candidates lack encoder provenance. The stored tree is evaluated only through its original encoder; temporal overlap is reported.', '',
               'Calibration artifacts are existing validation fits, not fitted on this recent test. Different calibration cohorts and repaired result-end priors make these exploratory comparisons. Current static champion profiles lack historical patch provenance. Player overlap and dependencies across bootstrap blocks limit uncertainty estimates.', '',
               'Reproduce from immutable source:', '', f'`python -m scripts.compare_served_candidates --replay "{out}" --output-dir runs/served_model_comparison_replay`', '']
    (out/'report.md').write_text('\n'.join(report),encoding='utf-8')
    from core.evaluation_plots import write_evaluation_plots
    write_evaluation_plots(out, test, predictions, metrics)
    print(f"Complete: {out}; common matches={metrics['intersection_count']}",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--replay', help='Existing frozen run directory; all frozen hashes verified')
    p.add_argument('--history-size', type=int, default=10000)
    p.add_argument('--validation-size', type=int, default=500)
    p.add_argument('--test-size', type=int, default=2000)
    p.add_argument('--candidates', help='Optional comma-separated model ids; tree baseline always attempted')
    run(p.parse_args())


if __name__ == '__main__':
    main()
