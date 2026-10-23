"""Record feature/data reception evidence without computing test performance."""
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.superpowers/sdd/2026-10-09-training-serving-parity-and-frozen-dataset'
RUN = ROOT / 'runs/aligned_dataset_20261009_reviewed'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()

meta = read(RUN / 'dataset.json')
replay = read(RUN / 'replay_verification.json')
live = read(RUN / 'live_parity.json')
assert replay['status'] == live['status'] == 'verified'
assert replay['test_metrics_computed'] is live['test_metrics_computed'] is False
shutil.copyfile(WORK / 'inventory.json', RUN / 'source_inventory.json')
shutil.copyfile(ROOT / 'scratch/audit_aligned_live.py', RUN / 'live_audit.py')
assert sha(RUN / 'live_audit.py') == live['audit_script_sha256']
shutil.copyfile(ROOT / 'scratch/trace_aligned_freeze.py', RUN / 'freeze_driver.py')
shutil.copyfile(ROOT / 'scratch/refreeze_aligned.py', RUN / 'refreeze_driver.py')
records = {p: read(RUN / (('sealed_test' if p == 'test' else p) + '.json'))
           for p in ('train', 'validation', 'test')}
exclusions = read(RUN / 'exclusions.json')
causes = Counter(r['reason'] for r in exclusions)
rows = json.loads(gzip.decompress((RUN / 'source_rows.json.gz').read_bytes()))
part_rows = []
population_rows = []
for name, part in records.items():
    part_rows.append(f"| {name} | {len(part):,} | {utc(part[0]['game_creation_time'])} | {utc(part[-1]['game_creation_time'])} |")
    for key in ('queue_id', 'region', 'patch'):
        values = Counter(str(r.get(key)) for r in part)
        population_rows.append(f"| {name} | {key} | " + ', '.join(f'{v}: {n:,}' for v, n in sorted(values.items())) + ' |')
availability_rows = []
for contract, a in meta['availability'].items():
    reasons = Counter(e['reason'] for e in a['errors'])
    availability_rows.append(f"| {contract} | {a['status']} | {a['eligible']:,}/{a['total']:,} | {100*a['coverage']:.1f}% | " + '; '.join(f'{k} ({v:,})' for k, v in reasons.items()) + ' |')
paths = [*meta['files'], 'dataset.json', 'live_panel.json', 'live_parity.json',
         'live_audit.py', 'replay_verification.json', 'source_inventory.json']
paths.append('freeze_driver.py')
paths.append('refreeze_driver.py')
hashes = {name: sha(RUN / name) for name in paths}
for name, binding in meta['files'].items():
    assert hashes[name] == binding['sha256']
reference = read(ROOT / 'docs/model_serving/reference.manifest.json')
assert reference['serving_validation']['status'] == 'unverified'
assert sha(ROOT / reference['checkpoint']['path']) == reference['checkpoint']['sha256']
old_protocol_sha = sha(ROOT / 'runs/served_model_comparison_20261009_complete/protocol.json')
assert old_protocol_sha == 'face72e773913ff63af3305fdb1ae5abbfbf0220369199e53cab585e2b0d6355'
assert 'Frozen evaluation runtime source SHA-256 mismatch' in (
    WORK / 'old-runtime-refusal.log').read_text(encoding='utf-8', errors='replace')
report = dict(status='verified_features', timestamp_utc=datetime.now(timezone.utc).isoformat(),
    source_count=len(rows), partitions={k: len(v) for k, v in records.items()},
    exclusions_by_reason=dict(causes), artifacts_sha256=hashes,
    runtime_sources_verified=True, tensor_comparisons=replay['tensor_comparisons'],
    live_panel_count=live['panel_count'], test_metrics_computed=False,
    serving_validation_status='unverified', reference_checkpoint_sha256=reference['checkpoint']['sha256'],
    historical_protocol_replay=dict(status='rejected_runtime_mismatch', exit_code=1,
                                    unchanged_protocol_sha256=old_protocol_sha))
(RUN / 'artifact_checks.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
test_log = (WORK / 'final-tests.log').read_text(encoding='utf-8', errors='replace')
result = next(line for line in reversed(test_log.splitlines()) if 'passed' in line and 'failed' not in line)
assert not re.search(r'\b[1-9]\d* (failed|errors?)\b', result)
command = '.\\.venv312\\Scripts\\python.exe -m pytest tests/test_frozen_history.py tests/test_aligned_partitions.py tests/test_aligned_artifacts.py tests/test_aligned_training.py tests/test_aligned_parity.py tests/test_serving_contract.py tests/test_pre_match_history_serving.py tests/test_serving_http.py tests/test_predict_api.py tests/test_config_cli.py tests/test_ordered_training_integration.py tests/test_train_deepsets_v4_timeline.py tests/test_equivariant_model.py tests/test_deepsets_layers.py tests/test_timeline_metrics.py tests/test_point_in_time_timeline.py -q'
text = f'''# Réception du dataset aligné entraînement/service

Réception du 9 octobre 2026. Le dataset **14D est prêt pour une future expérience**, avec parité de features vérifiée. Le contrat 18D est bloqué par l'absence de timelines dans la source récente. Aucun entraînement réel, métrique prédictive du test réservé ou changement de modèle servi n'a été effectué dans ce chantier.

## Source et sélection

Source SQLite en lecture seule : `{meta['source_db']}`. Inventaire enregistré dans [source_inventory.json](../../runs/aligned_dataset_20261009_reviewed/source_inventory.json). Le snapshot contient **{len(rows):,} créations CLASSIC**, de {utc(rows[0]['game_creation_time'])} à {utc(rows[-1]['game_creation_time'])}. Les 10 000 premières lignes constituent le contexte historique et ne sont pas des cibles. Les créations ex æquo à la dernière ligne de warmup sont également exclues. Population cible : queues 420/440, cinq rôles valides par équipe, vainqueur Riot cohérent avec les labels ML multitâches.

Le cutoff exploratoire antérieur est **{utc(meta['selection']['after_ms'])}**. Le SHA-256 du cohort antérieur est `{meta['selection']['after_cohort_sha256']}` ; ses {len(meta['selection']['excluded_test_ids']):,} IDs test sont exclus des cibles. Le nouveau test commence strictement après ce cutoff.

| Partition | Cibles | Première création UTC | Dernière création UTC |
|---|---:|---|---|
{chr(10).join(part_rows)}

Les créations identiques restent groupées. Le train est purgé de toute partie finissant à partir du début de validation ; la validation de toute partie finissant à partir du début du test. Les minima de 3 000 cibles validation et test ont été conservés. Causes enregistrées dans [exclusions.json](../../runs/aligned_dataset_20261009_reviewed/exclusions.json) :

| Cause | Lignes |
|---|---:|
{chr(10).join(f'| {k} | {v:,} |' for k, v in sorted(causes.items()))}

## Population, sans lecture de métriques prédictives

| Partition | Attribut connu avant match | Effectifs |
|---|---|---|
{chr(10).join(population_rows)}

## Contrats et audits

| Contrat | Statut | Éligibles/total | Couverture | Causes |
|---|---|---:|---:|---|
{chr(10).join(availability_rows)}

Les tensors sont float32 : équipes `(N, 5, 14)` et macro `(N, 15)`. Le replay complet a vérifié **{replay['tensor_comparisons']:,} comparaisons exactes** par `np.array_equal`, à partir des faits figés, sans relire la base. Il contrôle également formes, dtype, valeurs finies, chemins internes et SHA-256. Le test réservé est rejoué pour vérifier ses features ; aucune accuracy, log-loss ou sélection d'hyperparamètres n'en est extraite. Voir [replay_verification.json](../../runs/aligned_dataset_20261009_reviewed/replay_verification.json).

Un panel déterministe de **{live['panel_count']} cibles train/validation**, fixé avant comparaison avec des positions régulièrement espacées incluant les extrêmes, a été rejoué au même cutoff via `DatabaseHistoryProvider`. Priors et tenseurs sont exactement égaux sur les 20 cibles. Provenance vérifiée : même création, fenêtre globale de 10 000 créations antérieures, dernier résultat strictement antérieur au cutoff. Voir [live_panel.json](../../runs/aligned_dataset_20261009_reviewed/live_panel.json), [live_parity.json](../../runs/aligned_dataset_20261009_reviewed/live_parity.json) et [script d'audit](../../runs/aligned_dataset_20261009_reviewed/live_audit.py).

Les tests des trainers ont branché les tenseurs alignés jusqu'au premier batch et contrôlé la clôture/calibration sur validation. L'ouverture des faits sources et des fichiers du test réservé est interdite par les tests du loader. Le trainer v4 refuse la disponibilité 18D bloquée avant de créer un modèle. Les chemins legacy restent exploratoires.

## Commandes exécutées et vérifications

```powershell
.\\.venv312\\Scripts\\python.exe -m scripts.prepare_aligned_dataset inventory --db training_data.db --db runs/training_data_23h_20260922_191434.db --db runs/db_merge_20260924/consolidated.db
.\\.venv312\\Scripts\\python.exe -u scratch/trace_aligned_freeze.py freeze --db runs/db_merge_20260924/consolidated.db --after-cohort runs/served_model_comparison_20261009_complete/cohort.jsonl --output-dir runs/aligned_dataset_20261009 --source-limit 61000
.\\.venv312\\Scripts\\python.exe -u scratch/refreeze_aligned.py
.\\.venv312\\Scripts\\python.exe -u scratch/refreeze_aligned.py --from-dir runs/aligned_dataset_20261009_verified --output-dir runs/aligned_dataset_20261009_reviewed
.\\.venv312\\Scripts\\python.exe -m scripts.prepare_aligned_dataset verify --dataset-dir runs/aligned_dataset_20261009_reviewed
.\\.venv312\\Scripts\\python.exe scratch/audit_aligned_live.py --dataset-dir runs/aligned_dataset_20261009_reviewed
.\\.venv312\\Scripts\\python.exe -m scripts.formalize_served_model
$env:TEMP='D:\\projet\\NNriot\\scratch\\aligned-tests'
$env:TMP=$env:TEMP
{command}
```

Résultat de la suite ciblée : **{result}**, exit 0. Ceci couvre les 16 suites listées, sans prétendre que tous les tests du dépôt ont été exécutés. Les avertissements Keras/NumPy déjà présents sont conservés dans le journal.

Le gel initial sous `runs/aligned_dataset_20261009` a exécuté la sous-commande `freeze` via le wrapper diagnostique `scratch/trace_aligned_freeze.py`, avec les mêmes paramètres ; il ajoute uniquement une trace de pile périodique et un journal de progression. Les premières lectures ont été interrompues avant création d'un dataset, pour corriger le coût de la sélection SQLite. La fenêtre finale utilise les index couvrant les rowids et dates, résout les IDs ex æquo à sa frontière et lit les payloads retenus en ordre de rowid, dans une seule transaction en lecture seule. Les temporaires SQLite sont en mémoire ; aucune écriture ni création d'index sur la source.

Après suppression des lectures ML inutiles dans `DatabaseHistoryProvider`, un premier refreeze a créé `runs/aligned_dataset_20261009_verified`. La revue indépendante finale a ensuite reproduit un défaut sur des timelines partielles : des mesures absentes étaient remplacées par zéro puis comptées comme observations 18D. La correction exige maintenant dix participantFrames contenant les quatre mesures numériques finies requises, ainsi que les listes d'événements observées. Une liste vide ou une mesure numérique égale à zéro reste une observation valide. Les timelines incomplètes ne peuvent plus rendre 18D disponible ; les résultats Riot restent utilisables en 14D. Le chemin legacy permissif est conservé explicitement et ne reçoit aucune certification nouvelle. Sept cas de régression ont été observés RED, puis la vérification ciblée a obtenu 36 passes GREEN, incluant le writer et SQLite. Aucune seconde revue n'a été lancée.

La deuxième commande `refreeze_aligned.py` a créé le répertoire final `runs/aligned_dataset_20261009_reviewed` à partir des faits et records vérifiés du snapshot précédent, sans relire SQLite. SHA des faits, identifiants/ordre de partitions et fingerprint de données sont identiques ; les trois NPZ 14D gardent exactement les mêmes SHA-256 (assertions exécutées). Le manifeste final lie le runtime corrigé et le SHA du manifeste précédent. Les deux snapshots précédents restent archivés, avec leurs audits d'origine. Replay complet et panel SQLite ont été relancés pour le runtime final. Pour préparer une nouvelle expérience, exécuter directement `python -m scripts.prepare_aligned_dataset freeze` avec les arguments de la commande diagnostique ci-dessus et un répertoire neuf ; aucun wrapper n'est nécessaire au fonctionnement du CLI.

Les 22 entrées d'inventaire et manifestes ont été régénérées avec les sources de features actuelles. Le checkpoint de référence reste lié à `{reference['checkpoint']['sha256']}` et le statut de service reste `unverified`. Les calibrations et checkpoints existants n'ont pas été réécrits. Les anciens protocoles de comparaison gardent leurs hashes d'origine ; leurs replays doivent refuser le nouveau runtime.

Ce refus a également été vérifié sur le protocole réel par `python -m scripts.compare_served_candidates --replay runs/served_model_comparison_20261009_complete --output-dir scratch/aligned-tests/legacy-runtime-refusal` : `ContractError: Frozen evaluation runtime source SHA-256 mismatch`, exit 1 attendu, avant toute comparaison de modèles. Aucun fichier de prédictions ou métriques n'a été écrit dans ce répertoire scratch. SHA du protocole identique avant/après : `{old_protocol_sha}`.

## Empreintes des artefacts

[dataset.json](../../runs/aligned_dataset_20261009_reviewed/dataset.json) lie partitions, tenseurs, faits sources, disponibilité, règles et hashes de code. [artifact_checks.json](../../runs/aligned_dataset_20261009_reviewed/artifact_checks.json) enregistre les contrôles ainsi que les preuves SQLite supplémentaires.

| Artefact | SHA-256 |
|---|---|
{chr(10).join(f'| `{k}` | `{v}` |' for k, v in hashes.items())}

## Limites et étape suivante

Les temps d'acquisition historiques ne sont pas stockés : disponibilité présumée à la fin Riot du résultat. Les profils statiques des champions sont hashés, mais leur disponibilité historique par patch n'est pas certifiée. Le test est réservé à la future expérience de ce chantier ; cela ne prouve pas que toute la base a toujours été inconnue lors de recherches antérieures. Les fichiers de faits contiennent les résultats Riot : la réservation est une discipline d'expérience.

La suite logique est un protocole d'expérience 14D utilisant train/validation pour entraîner, calibrer et choisir, puis une seule évaluation finale réservée. Les commandes de pilots sont documentées dans [pre_match_training.md](../pre_match_training.md) et n'ont pas été exécutées ici. Le 18D récent exige une acquisition de timelines utilisable également au service, dans un chantier séparé ; ni remplissage de zéros ni fusion implicite avec une autre base.
'''
(ROOT / 'docs/model_serving/aligned_dataset_verification.md').write_text(text, encoding='utf-8')
print(json.dumps({k: report[k] for k in ('status', 'source_count', 'partitions', 'exclusions_by_reason', 'tensor_comparisons', 'live_panel_count')}, ensure_ascii=False))
