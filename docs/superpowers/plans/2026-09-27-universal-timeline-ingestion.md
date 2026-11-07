# Ingestion universelle des timelines — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Garantir que chaque match ingéré reçoit une décision durable de collecte des coordonnées, quel que soit son producteur, avec reprise et quotas partagés.

**Architecture:** Outbox transactionnelle dans la base opérationnelle v13, consommée par un worker sans réseau dans les transactions. Adaptateur pour le journal historique existant et coordinateur de quotas commun à tous les appels Riot.

**Tech Stack:** Python 3.12, SQLite/WAL, requests, pytest, PowerShell ; aucune nouvelle dépendance produit nécessaire.

**Spec:** `docs/superpowers/specs/2026-09-27-universal-timelines-and-spatial-history-design.md`, sections communes et A.

## Global Constraints

- Tous les fichiers de données, caches, bases temporaires, rapports et modèles candidats restent sous `D:/projet/NNriot/runs/` ; aucun cache de données sur C:.
- Chaque accès reçoit un `db_path` explicite et une identité de base vérifiée ; aucune écriture ne dépend du répertoire courant.
- Aucun test ne contacte Riot ; les clés ne figurent ni dans les journaux, ni dans les bases de coordination, ni dans les rapports.
- La publication des bases suit `docs/database_merge.md` : sortie préparée, contrôle, arrêt des connexions, checkpoint WAL, copie de retour arrière et bascule explicite.
- La campagne actuelle et les modifications locales de l'utilisateur ne sont pas modifiées par cette planification.

## Review Focus

1. Réingestion d'un match déjà présent sans timeline ou avec ancien NULL : job unique, aucun oubli — tâches 2/5.
2. Processus tué après téléchargement ou lors d'une réservation expirée : aucune perte ou écriture d'un ancien propriétaire — tâche 4.
3. Deux processus utilisant la même clé pour des méthodes différentes, avec dernier 429 ou saut d'horloge : aucun contournement de quota — tâche 3.
4. Fusion de bases v12/v13 et conflits de brut : cible prioritaire, aucune réservation importée active, données exclues archivées — tâche 5.
5. Relance contre un autre chemin de base ou un ancien manifeste : refus explicite ; file épuisée avec erreurs distincte de collecte complète — tâches 2/4/6.

## Préparation et ordre

Le worktree de collecte existant est utilisé par un processus. Créer, à l'exécution, un autre checkout géré en partant de son HEAD vérifié (`3829d86` à la rédaction), sans modifier ni archiver celui qui tourne. Inspecter les artefacts avant création. Préserver le checkout principal et ses changements dans `player_priors`, `selective_*`, tests et expériences. Ne pas utiliser `git add .`.

Runtime de validation : `D:/projet/NNriot/.venv312/Scripts/python.exe`. Dans le checkout choisi, définir `$py` à ce chemin. Les chemins relatifs ci-dessous désignent ce checkout, jamais un emplacement implicite de données.

Avant chaque invocation pytest, utiliser un dossier neuf sous D: pour ses fixtures et ses processus enfants :

```powershell
$testRoot = Join-Path 'D:/projet/NNriot/runs' ('timeline-tests-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
$env:PYTEST_ADDOPTS = '--basetemp=' + (Join-Path $testRoot 'pytest')
$env:TEMP = $testRoot
$env:TMP = $testRoot
$env:SQLITE_TMPDIR = $testRoot
```

Ces variables sont limitées au shell de validation ; préserver/restaurer leurs valeurs précédentes à sa sortie.

Ordre : 1 → 2 → 3 → 4 → 5 → 6. Ne pas démarrer de second collecteur réel avant le handoff final. Les tests de concurrence utilisent des processus locaux simulant HTTP, pas la clé réelle.

## Structure des fichiers

| Fichier | Responsabilité |
|---|---|
| `core/timeline_policy.py` | Décision queue/carte/identité, métadonnées canonisées |
| `core/timeline_quality.py` | Qualification de la timeline brute partagée entre backfill/live |
| `core/timeline_queue.py` | Outbox, réservations, transitions et réconciliation |
| `core/timeline_worker.py` | Une boucle de consommation indépendante du support de jobs |
| `core/riot_rate_coordination.py` | Réservations et cooldowns API entre processus |
| `core/database.py`, `core/repository.py` | Migration, identité, écritures atomiques |
| `core/riot_api.py`, `core/config.py` | Appels HTTP coordonnés et configuration explicite |
| `scripts/run_timeline_worker.py`, `scripts/reconcile_timeline_queue.py` | Consommation live et rattrapage hors réseau |
| `scripts/merge_databases.py`, `scripts/migrate_timeline_outbox.py` | Fusion et migration sur sortie préparée |
| Producteurs existants, routes d'historique | Sauvegarde commune sans téléchargements timeline ad hoc |

## Task 1: Extraire les contrats réutilisables du backfill

**Files:** créer `core/timeline_policy.py`, `core/timeline_quality.py`, `tests/test_timeline_policy.py` ; modifier `core/timeline_collection.py` ; préserver `tests/test_timeline_corpus.py`, `tests/test_riot_timeline_results.py`.

**Interfaces:** `decide_timeline(match_id: str, raw: object) -> dict` retourne `state`, `reason`, `policy_version='soloq_v1'`, `candidate: dict | None`. Le candidat contient début/fin, patch, plateforme, IDs/PUUID et équipes physiques dérivées du match brut. `inspect_timeline(match_id: str, payload: object, expected: dict | None=None) -> dict` conserve le contrat existant.

- [ ] Écrire les tests queue 420/carte 11, autre queue, JSON incomplet, identités dupliquées et fin inconnue. Réutiliser les fabriques de `tests/test_timeline_corpus.py` en les déplaçant dans `tests/timeline_fixtures.py`, puis adapter leurs imports sans modifier les payloads.

```python
from tests.timeline_fixtures import match_payload
from core.timeline_policy import decide_timeline

def test_other_queue_has_explicit_decision():
    d = decide_timeline('EUW1_1', match_payload('EUW1_1', queue=440))
    assert (d['state'], d['reason']) == ('out_of_scope', 'queue')
    assert d['policy_version'] == 'soloq_v1'
```

- [ ] Lancer `& $py -m pytest tests/test_timeline_policy.py -q` ; constater l'échec avant création de l'interface.
- [ ] Extraire `_candidate` et `inspect_timeline` sans changer la qualification du corpus historique. Traduire seulement la décision live : rejet de queue/carte → `out_of_scope`, incohérence du brut → `invalid`, plateforme inconnue → `unsupported`, candidat admissible → `pending`. Garder un adaptateur `_candidate` pour l'ancien manifeste et les tests qui s'y réfèrent.
- [ ] Vérifier `test_timeline_policy`, `test_timeline_corpus`, `test_riot_timeline_results` ; le manifeste historique ne change ni de hash ni de périmètre.
- [ ] Commit limité aux contrats et tests : `refactor: share timeline policy and quality contracts`.

## Task 2: Migrer vers v13 et rendre match/demande atomiques

**Files:** modifier `core/database.py`, `core/repository.py`, `core/config.py` ; créer `core/timeline_queue.py`, `scripts/migrate_timeline_outbox.py`, `tests/test_database_v13_timeline_outbox.py`, `tests/test_timeline_outbox.py`.

**Interfaces:** `enqueue_timeline_decision(conn, match_id: str, raw: object, now_ms: int) -> dict` utilise la transaction appelante. `get_database_identity(db_path: str) -> str`. `migrate_timeline_database(source: str, output: str, report: str) -> dict` refuse les sorties existantes et n'écrit jamais la source. Les signatures publiques `save_match` / `save_matches_batch` restent compatibles.

- [ ] Écrire le test d'interruption dans la transaction. Les fixtures utilisent une vraie SQLite temporaire.

```python
def test_enqueue_failure_rolls_back_match(tmp_path, monkeypatch):
    from core import database, timeline_queue
    from tests.timeline_fixtures import match_payload
    db = str(tmp_path / 'v13.db')
    database.init_db(db)
    def fail(*args, **kwargs):
        raise RuntimeError('injected enqueue failure')
    monkeypatch.setattr(timeline_queue, 'enqueue_timeline_decision', fail)
    import pytest
    with pytest.raises(RuntimeError):
        database.save_match('EUW1_1', match_payload('EUW1_1'), db_path=db)
    assert not database.match_exists('EUW1_1', db_path=db)
```

- [ ] Constater l'échec, puis ajouter le schéma exact de l'outbox défini dans la spec et une table `database_metadata(key PRIMARY KEY,value)`. Activer les FK sur les connexions concernées et indexer les échéances. Sur une base neuve, `init_db` crée v13 ; une base existante v12 exige une migration explicite, sans upgrade surprise au démarrage du service.
- [ ] Appeler `enqueue_timeline_decision` dans la transaction des deux fonctions de sauvegarde, après détermination du brut canonique enregistré. Un doublon n'utilise pas le nouveau payload contradictoire pour fabriquer son candidat. Un match existant sans job obtient sa décision ; un job terminé ne perd pas son historique.
- [ ] Ajouter les cas : rollback du batch, doublons dans le batch, ancien NULL, autre queue, sauvegarde via repository et appel direct, deux bases portant le même match ID, mauvais identifiant attendu. Pour un brut valide déjà stocké, qualifier et créer `saved` sans réseau.
- [ ] Créer la commande de migration : backup cohérent vers sortie neuve sur D:, migration, nouvelle identité, contrôle SQLite/FK, rapports de comptes. L'outbox des anciens matchs est remplie par la tâche 5 pour borner la migration.
- [ ] Adapter explicitement la lecture/copie historique pour accepter les schémas 12 et 13 sans migration de source. Ajouter le test d'un ancien journal v1 avec sa base v12 après passage de `SCHEMA_VERSION` à 13 ; le worker historique ne doit pas appeler une initialisation qui impose v13.
- [ ] Vérifier `& $py -m pytest tests/test_database_v13_timeline_outbox.py tests/test_timeline_outbox.py tests/test_collection_database_selection.py -q` et committer `feat: persist timeline decisions with match transactions`.

## Task 3: Partager les quotas entre tous les processus Riot

**Files:** créer `core/riot_rate_coordination.py`, `tests/test_riot_shared_quota.py` ; modifier `core/riot_api.py`, `core/config.py`, `tests/test_riot_timeline_results.py`.

**Interfaces:** `SharedRiotQuota(db_path: str, key_scope: str)` ; `acquire(host: str, method: str) -> str` retourne un ID de réservation ; `observe(reservation_id: str, status: int | None, headers: dict) -> None`. `key_scope` est une empreinte SHA-256 de la clé ; la clé brute n'est pas passée au store. `NNRIOT_RIOT_RATE_DB` désigne le fichier partagé obligatoire lors d'une collecte réelle coordonnée.

- [ ] Ajouter des tests avec horloge injectée et deux processus : requête détails puis timeline sous quota application 1/10 s ; un troisième 429 doit reporter le premier appel de l'autre processus. Les méthodes gardent des compteurs distincts mais partagent le plafond application.

```python
def test_last_429_blocks_other_method(tmp_path):
    from core.riot_rate_coordination import SharedRiotQuota
    q = SharedRiotQuota(str(tmp_path / 'quota.db'), key_scope='fake-key-digest')
    token = q.acquire('europe', 'timeline')
    q.observe(token, 429, {'Retry-After': '120'})
    # La fixture multiprocess de ce fichier avance une horloge contrôlée ;
    # elle vérifie que la requête « details » ne part pas avant t=120.
    with q.inspect_state() as state:
        assert state.cooldown_seconds_remaining() >= 119
```

Le hook de lecture `inspect_state()` est à implémenter comme contexte read-only retournant une vue avec `cooldown_seconds_remaining()`, sans clé ni données joueur. Ajouter en complément le test multiprocess réellement bloquant avec fenêtres courtes, délai maximal de cinq secondes et synchronisation par Event, pas de longue attente arbitraire.

- [ ] Constater RED. Implémenter réservation atomique `BEGIN IMMEDIATE`, purge bornée des anciennes réservations, quotas minimums observés, application conservative du compteur serveur et cooldown partagé. Réservation conservée dans le budget en cas de timeout réseau. Aucun lock SQLite pendant le sleep. Sur verrou inaccessible : attente bornée puis arrêt récupérable, jamais de requête libre.
- [ ] Adapter `_make_request` et `get_match_timeline_result` avant **chaque** tentative. Les routes compte/joueur/détails participent au même quota application. Conserver le fallback conservateur de Retry-After et tracer les tentatives avant éventuelle exception auth.
- [ ] Tester rotation de clé, hôtes différents, en-têtes absents/malformés, reboot/reprise de réservations et horloge revenant en arrière. Une rupture d'horloge impose une fenêtre conservatrice ; elle ne remet pas les compteurs à zéro.
- [ ] Vérifier tests HTTP existants et nouveaux tests multiprocess, puis commit `feat: coordinate Riot quotas across collection processes`.

## Task 4: Consommer les deux supports de jobs et raccorder tous les producteurs

**Files:** créer `core/timeline_worker.py`, `scripts/run_timeline_worker.py`, `tests/test_timeline_worker.py`, `tests/test_ingestion_timeline_paths.py` ; modifier `core/timeline_queue.py`, `core/timeline_collection.py`, `scripts/data_collector.py`, `scripts/background_collector.py`, `scripts/crawler_1m.py`, `app/routes/history.py`.

**Interfaces:** `JobStore.claim(owner: str, now_ms: int) -> dict | None`, `renew(job: dict, owner: str, now_ms: int) -> bool`, `finish(job: dict, owner: str, outcome: dict) -> None`, `recover(now_ms: int) -> int`. Deux implémentations : `OperationalTimelineStore(db_path, database_identity)` et `ManifestTimelineStore(db_path,state_path,manifest_path)`. `run_timeline_worker(store, *, max_jobs: int, quota_db: str, min_free_gb: float=10) -> dict` traite un budget positif ; le mode service boucle par lots bornés.

- [ ] Écrire le test d'intégration de chaque entrée réelle avec HTTP simulé : collecte seed, top-leagues, PUUID, batch/smurfs, background, crawler, routes history/batch, LoLPros et imports. Après sauvegarde, exactement une décision dans la base demandée et **zéro** appel timeline synchrone du producteur.
- [ ] Tester un worker tué après réception avant commit et un propriétaire devenu obsolète. Le store live valide le brut et `saved` dans la même transaction ; le store historique conserve sa réconciliation existante brut puis journal.

```python
def test_stale_lease_cannot_finish(live_store, valid_outcome):
    first = live_store.claim('worker-a', now_ms=1000)
    live_store.recover(now_ms=302000)
    second = live_store.claim('worker-b', now_ms=302001)
    import pytest
    with pytest.raises(RuntimeError, match='lease'):
        live_store.finish(first, 'worker-a', valid_outcome)
    live_store.finish(second, 'worker-b', valid_outcome)
```

Les fixtures `live_store` et `valid_outcome` sont définies dans ce fichier à partir de `database.init_db`, `save_match`, des fabriques de la tâche 1 et de `inspect_timeline`. Horloges et HTTP restent injectés.

- [ ] Constater RED puis implémenter réservations de 300 s avec heartbeat pendant les attentes de quota. Une réponse appartenant à un ancien owner est rejetée. Vérifier l'identité de la base à l'ouverture, sa capacité avant réseau et écriture, et la priorité des nouveaux jobs sans affamer les retries dus.
- [ ] Retirer les quatre téléchargements conditionnels de `data_collector` et celui du background. Le drapeau historique ne bloque plus la création des décisions. La consommation dépend du worker explicite. Les wrappers d'import héritent du même comportement sans code réseau ajouté.
- [ ] Propager partout `db_path` ; le crawler et la route directe sont couverts par leur sauvegarde centrale. Étendre les sorties de statut existantes avec compteurs queued/saved/usable/partial/retry/invalid et âge du plus vieux job, sans exposer de clé ou PUUID.
- [ ] Vérifier l'ensemble des tests ingestion et l'adaptateur du backfill, puis commit `feat: route all match ingestion through durable timeline jobs`.

## Task 5: Rattraper les anciennes données et adapter la fusion

**Files:** créer `scripts/reconcile_timeline_queue.py`, `tests/test_timeline_reconciliation.py` ; modifier `scripts/merge_databases.py`, `tests/test_merge_databases.py`, `docs/database_merge.md`.

**Interfaces:** `reconcile_timeline_queue(db_path: str, *, after_rowid: int=0, batch_size: int=1000) -> dict` retourne `scanned`, `created`, `reconciled`, `last_rowid`, `finished`. Curseur durable dans une table de maintenance de la même base. Aucune requête Riot.

- [ ] Ajouter des fixtures v12/v13 mêlant match sans timeline, ancien NULL, brut valide, brut corrompu, autre queue et identifiant conflictuel. Vérifier exactement une décision par match canonique après deux passes.

```python
def test_null_is_pending_but_received_payload_is_preserved(reconciliation_db):
    from scripts.reconcile_timeline_queue import reconcile_timeline_queue
    first = reconcile_timeline_queue(reconciliation_db, batch_size=2)
    assert first['scanned'] == 2
    second = reconcile_timeline_queue(reconciliation_db, batch_size=2)
    assert second['created'] == 0
```

La fixture crée exactement deux matchs, un ancien NULL et un payload valide ; ajouter des assertions SQL `pending`/`saved` et hash du payload intact, pas seulement les compteurs.

- [ ] Constater RED puis scanner par rowid avec borne supérieure capturée au lancement. Reprendre le curseur d'une exécution interrompue ; un nouveau lancement complet repart de zéro avec nouvelle borne. Des insertions simultanées sont couvertes par l'outbox atomique et ne rendent pas le scan infini.
- [ ] Adapter les allowlists de tables et contrôles de version du merge. Sortie v13 : jobs cible conservés avec leases libérés lors de la maintenance, jobs des matchs importés dérivés du brut canonique ; aucun transfert aveugle de retry/lease de la source.
- [ ] Corriger la fusion d'une timeline source valide lorsque la cible possède un ancien NULL, avec validation du match et conservation des conflits dans l'archive. Payload cible valide prioritaire ; payload cible corrompu archivé avant toute réparation explicitement demandée.
- [ ] Vérifier compteurs, intégrité, FK et archives pour v12→v13 et v13→v13. Documenter migration, publication et rollback. Commit `feat: reconcile missing timelines and preserve jobs during database merge`.

## Task 6: Valider, préparer la maintenance et effectuer le handoff contrôlé

**Files:** modifier `docs/timeline_collection.md` ; créer rapport d'exécution sous `D:/projet/NNriot/runs/universal_timeline_ingestion_<date>/` au moment de l'exécution, avec chemins uniques et provenance des commits.

- [ ] Exécuter les tests ciblés puis la suite complète du checkout intégré. Une revue indépendante examine les transactions, les deux processus HTTP et les anciennes données. Résoudre les constats importants avant usage réel.
- [ ] Faire un dry-run de migration/fusion sur D:, mesurer taille et durée, publier les compteurs par voie d'entrée. Ne pas migrer en place la base v12 encore ouverte par un processus.
- [ ] Préparer le dossier de maintenance : PID/processus réellement concernés, checkpoint source, copie rollback, bases v13 préparées, chemin commun des quotas et commande de reprise historique. Arrêter proprement les anciens processus à une frontière sûre et vérifier la libération des connexions avant bascule.
- [ ] Publier selon `docs/database_merge.md`, puis démarrer les appels Riot coordonnés et reprendre l'ancien corpus avec ses fichiers inchangés. La copie historique peut rester v12 via son adaptateur ; une nouvelle base opérationnelle v13 reçoit les nouvelles ingestions.
- [ ] Pilote réel : 20 ingestions admissibles réparties entre les entrées disponibles, vérifier 20 décisions uniques ; budget worker maximum 200 jobs. Vérifier quotas, auth, profondeur de file, qualité et reprise, sans promettre 200 réponses exploitables.
- [ ] Livrer un rapport séparant `queue_drained`, `all_eligible_saved`, reçus, exploitables, partiels, indisponibles/retry et exclus ; vérifier les anciens NULL et la conservation des données. Aucun entraînement spatial n'est déclenché par ce plan A.

## Critère d'acceptation

Toutes les voies testées créent une décision dans la bonne base, aucune fenêtre de crash ne perd un job, deux processus respectent le quota commun, le corpus historique reprend sans altération de son manifeste et la migration est réversible. L'absence de timeline est une information traçable, jamais un succès implicite.
