"""New binding from already SHA-verified frozen facts; no SQLite reads."""
import gzip
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.aligned_artifacts import checked_path, write_dataset
from core.calibration import sha256_file

parser = argparse.ArgumentParser()
parser.add_argument('--from-dir', default='runs/aligned_dataset_20261009')
parser.add_argument('--output-dir', default='runs/aligned_dataset_20261009_verified')
args = parser.parse_args()
old = ROOT / args.from_dir
new = ROOT / args.output_dir
meta = json.loads((old / 'dataset.json').read_text(encoding='utf-8'))
rows = json.loads(gzip.decompress(checked_path(old, meta, 'source_rows.json.gz').read_bytes()))
splits = {part: json.loads(checked_path(old, meta,
          ('sealed_test' if part == 'test' else part) + '.json').read_text(encoding='utf-8'))
          for part in ('train', 'validation', 'test')}
excluded = json.loads(checked_path(old, meta, 'exclusions.json').read_text(encoding='utf-8'))
selection = {**meta['selection'], 'origin_dataset_manifest_sha256': sha256_file(old / 'dataset.json')}
out = write_dataset(new, rows, splits, source_db=meta['source_db'],
                    selection=selection, exclusions=excluded)
assert out['files']['source_rows.json.gz']['sha256'] == meta['files']['source_rows.json.gz']['sha256']
assert out['split_sha256'] == meta['split_sha256']
assert out['dataset_sha256'] == meta['dataset_sha256']
for name in ('train14.npz', 'validation14.npz', 'sealed_test14.npz'):
    assert out['files'][name]['sha256'] == meta['files'][name]['sha256']
print(json.dumps({v: {k: a[k] for k in ('status', 'eligible', 'total', 'coverage')}
                 for v, a in out['availability'].items()}), flush=True)
