"""Diagnostic wrapper; CLI arguments and data preparation stay unchanged."""
import faulthandler
import logging
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(
        root / '.superpowers/sdd/2026-10-09-training-serving-parity-and-frozen-dataset/freeze-progress.log',
        mode='w', encoding='utf-8')])
trace = (root / '.superpowers/sdd/2026-10-09-training-serving-parity-and-frozen-dataset/freeze-stack.log').open('w')
faulthandler.dump_traceback_later(60, repeat=True, file=trace)
try:
    runpy.run_module('scripts.prepare_aligned_dataset', run_name='__main__')
finally:
    faulthandler.cancel_dump_traceback_later()
    trace.close()
