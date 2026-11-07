"""Read-only live parity audit; never loads reserved test outcomes."""
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from core.aligned_artifacts import checked_path, read_train_validation, canonical
from core.aligned_features import extract_aligned_inputs
from core.calibration import sha256_file
from core.frozen_history import FrozenHistoryProvider
from core.pre_match_history import DatabaseHistoryProvider

root = ROOT / 'runs/aligned_dataset_20261009_verified'
x, records, meta = read_train_validation(root, 'nnriot.players14.v1')
combined = [(name, index, row) for name in ('train', 'validation')
            for index, row in enumerate(records[name])]
indices = np.linspace(0, len(combined) - 1, 20, dtype=int)
panel = [combined[int(index)] for index in indices]
selection = [{'partition': name, 'index': index, 'match_id': row['match_id'],
              'prediction_time_ms': row['game_creation_time']}
             for name, index, row in panel]
(root / 'live_panel.json').write_bytes(canonical(selection))
rows = json.loads(gzip.decompress(checked_path(root, meta, 'source_rows.json.gz').read_bytes()))
frozen = FrozenHistoryProvider(rows)
live = DatabaseHistoryProvider(meta['source_db'])
results = []
for name, index, row in panel:
    query = row['feature_json']
    a = frozen.enrich(query, include_timeline=False)
    b = live.enrich(query, include_timeline=False)
    priors_a = [p['priors'] for team in ('team_a', 'team_b') for p in a[team]]
    priors_b = [p['priors'] for team in ('team_a', 'team_b') for p in b[team]]
    if priors_a != priors_b:
        raise AssertionError(f"Live prior mismatch: {row['match_id']}")
    inputs = extract_aligned_inputs(b, 'nnriot.players14.v1')
    for key in inputs:
        if not np.array_equal(inputs[key][0], x[name][key][index]):
            raise AssertionError(f"Live tensor mismatch: {row['match_id']}/{key}")
    for enriched in (a, b):
        provenance = enriched['priors_provenance']
        assert provenance['as_of_ms'] == row['game_creation_time']
        assert provenance['history_window_limit'] == 10000
        end = provenance['latest_result_end_ms']
        assert end is None or end < row['game_creation_time']
    results.append(dict(match_id=row['match_id'], priors_equal=True, tensors_equal=True))
    print(f"Live parity {len(results)}/20: {row['match_id']}", flush=True)
report = dict(status='verified', source_db=meta['source_db'], panel_count=20,
              panel_sha256=sha256_file(root / 'live_panel.json'),
              audit_script_sha256=sha256_file(__file__), test_metrics_computed=False,
              results=results)
(root / 'live_parity.json').write_bytes(canonical(report))
