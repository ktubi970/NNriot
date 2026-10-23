"""Keep the uncommitted execution evidence with the delivered run."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.superpowers/sdd/2026-10-09-training-serving-parity-and-frozen-dataset'
RUN = ROOT / 'runs/aligned_dataset_20261009_reviewed'
execution = RUN / 'execution'
shutil.copytree(WORK, execution, dirs_exist_ok=True)
for name in ('audit_aligned_live.py', 'refreeze_aligned.py',
             'record_aligned_reception.py', 'preserve_aligned_execution.py'):
    shutil.copyfile(ROOT / 'scratch' / name, execution / name)
ledger = (WORK / 'progress.md').read_text(encoding='utf-8')
decisions = [line for line in ledger.splitlines()
             if 'Ruling:' in line or 'ruling refinement:' in line
             or 'Ruling refinement for' in line]
minors = [line for line in ledger.splitlines() if 'minor (deferred)' in line]
result = next(line for line in reversed((WORK / 'final-tests.log').read_text(
    encoding='utf-8', errors='replace').splitlines()) if 'passed' in line and 'failed' not in line)
text = f'''# Exécution Native — dataset aligné

Les six tâches fonctionnelles sont réalisées sur `codex/training-serving-parity`.
Le journal, les snapshots de départ et les diffs attribuables sont conservés
dans [execution/](../../runs/aligned_dataset_20261009_reviewed/execution/).
Base et HEAD : `3bb93c7` ; aucun commit, merge ou push mélangeant les dépendances
préexistantes. Le répertoire de travail et le journal ignoré sont conservés.

Réception : [aligned_dataset_verification.md](aligned_dataset_verification.md).
Suite finale : **{result}**, exit 0. Replay : 148 458 comparaisons exactes.
SQLite : les mêmes 20 IDs fixés avant comparaison, priors et tenseurs exacts.
Source et tenseurs 14D identiques aux snapshots précédents ; nouveaux hashes
de runtime dans le manifeste final, snapshots antérieurs préservés.

Une seule [revue indépendante](../../runs/aligned_dataset_20261009_reviewed/execution/final-review.md)
a été réalisée par gpt-6-astra avec contexte neuf. Son Important sur les
timelines sans observations a été reproduit par sept cas RED, corrigé dans
une seule passe et vérifié GREEN (36 tests), puis par la suite finale.
Pas de seconde revue. Aucun Critical ou Minor étayé n'a été rapporté.

## Décisions consignées, dans leur ordre

{chr(10).join('- ' + line for line in decisions)}

## Minor différés

{chr(10).join('- ' + line for line in minors) if minors else 'Aucun.'}

Les cinq points que le reviewer a laissés hors jugement disposent chacun
d'une décision explicite ci-dessus. Leurs limites ne sont pas des résultats
prédictifs certifiés. Les commandes et hashes du run figurent dans le rapport
de réception ; le [journal complet](../../runs/aligned_dataset_20261009_reviewed/execution/progress.md)
conserve les observations RED/GREEN et les tentatives interrompues.
'''
(ROOT / 'docs/model_serving/aligned_dataset_execution.md').write_text(text, encoding='utf-8')
hashes = {str(p.relative_to(execution)).replace('\\', '/'):
          hashlib.sha256(p.read_bytes()).hexdigest()
          for p in execution.rglob('*') if p.is_file() and p.name != 'evidence_sha256.json'}
(execution / 'evidence_sha256.json').write_text(json.dumps(hashes, sort_keys=True, indent=2) + '\n', encoding='utf-8')
print(f'Execution evidence saved: {len(hashes)} files, {len(decisions)} decisions.')
