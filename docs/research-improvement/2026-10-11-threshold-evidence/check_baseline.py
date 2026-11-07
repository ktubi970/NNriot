"""Run pytest with the original selective module, without reverting workspace files.

Pass the test node IDs to compare after a full-suite failure. Test fixtures still
isolate DBs/checkpoints; set TEMP/TMP to the workspace temp directory as documented.
"""
from pathlib import Path
import importlib.machinery
import importlib.util
import sys

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[2]))
import core
import pytest

name = "core.selective_classification"
loader = importlib.machinery.SourceFileLoader(name, str(OUT / "selective_classification.py.before"))
spec = importlib.util.spec_from_loader(name, loader)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
loader.exec_module(module)
core.selective_classification = module
print("Using original selective_classification snapshot; workspace files unchanged.", flush=True)
raise SystemExit(pytest.main(sys.argv[1:]))
