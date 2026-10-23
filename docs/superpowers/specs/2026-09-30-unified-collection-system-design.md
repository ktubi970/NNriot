# Unified Collection System Design

> **Document de conception d'architecture**  
> **Date :** 2026-09-30  
> **Statut :** Validé — Prêt pour plan d'implémentation  
> **Conformité :** Conforme aux conventions `AGENTS.md` (Domain Boundaries, Facade Pattern, Repositories).

---

## 1. Contexte & Diagnostic Critique

L'audit architectural a révélé un état hautement fragmenté et techniquement dégradé de la collecte de données au sein de NNriot :

1. **Violation majeure des frontières de couches (`Pattern Boundaries`)** :
   - Les routes Web Flask (`app/routes/api.py`, `app/routes/history.py`, `app/routes/lolpros.py`) importent directement des fonctions métier et des variables d'état globales depuis `scripts/data_collector.py`.
   - `scripts/` ne doit héberger que des points d'entrée CLI d'orchestration. Toute la logique d'ingestion et de gestion de file d'attente doit être localisée dans la couche domaine `core/`.
2. **Duplication et divergence des moteurs de collecte** :
   - `scripts/data_collector.py` : gère 4 modes disparates (`collect_training_data`, `collect_top_leagues_data`, `collect_by_puuid`, `collect_batch_with_smurfs`), utilise un état global mutable non réentrant (`batch_status`, `batch_lock`), et extrait des métriques avec fuite post-game (`extract_match_features` incluant KDA et golds finaux du match prédit).
   - `scripts/background_collector.py` : boucle `while True` dépilant de la table SQLite `crawl_queue`, mais utilisant des seeds figées, contournant la couche Repository par des accès SQL directs, et avec une file indexée par nom/tag qui gaspille ~50% du quota API en résolutions préliminaires de PUUID.
   - `scripts/crawler_1m.py` : crawler autonome haute cadence contournant complètement SQLite `crawl_queue` au profit d'un fichier JSON `crawler_frontier.json` sur disque, utilisant cette fois `extract_pre_match_features` (sans fuite), mais ignorant totalement les timelines.
   - `app/routes/history.py` : duplique l'ingestion de matches en ligne directement à l'intérieur du endpoint HTTP `api_player_history`.
3. **Incohérence des Timelines & Embeddings Spatiaux** :
   - Traitement hétérogène des timelines (téléchargement conditionnel dans certains scripts via `NNRIOT_FETCH_TIMELINES`, totalement ignoré dans `crawler_1m`).

---

## 2. Objectifs d'Architecture

1. **Centralisation dans `core/collection/`** :
   - Fournir une couche de service unifiée (`CollectionService`) agissant en Façade pour toutes les entrées du système (CLI, tâches de fond, endpoints Web).
2. **Élimination stricte des fuites de données** :
   - Utilisation systématique et exclusive de `json_utils.extract_pre_match_features` et `feature_labels.extract_labels` pour toutes les données d'entraînement.
3. **File de crawl unifiée dans SQLite (Migration v14)** :
   - Migrer la table `crawl_queue` pour indexer en premier ordre `(puuid, region)`.
   - Snowballing direct : les 10 participants d'un match sont enregistrés directement par PUUID sans appel réseau Riot `get_puuid`.
4. **Collecte systématique des Timelines et Embeddings Spatiaux** :
   - Chaque match ingéré télécharge sa timeline correspondante et extrait ses trajectoires spatiales 32D via `core/spatial_rnn.py`, avec gestion gracieuse des erreurs (les 404 Riot ne bloquent pas l'enregistrement du match).
5. **Thread-Safety & Suivi d'Avancement** :
   - Remplacement de l'état global volatile `data_collector.batch_status` par `CollectionTracker`, un tracker thread-safe réentrant et inspectable par l'API Web.
6. **Rétro-compatibilité 100% assurée** :
   - Préservation des interfaces de `scripts/data_collector.py`, `scripts/background_collector.py`, `scripts/crawler_1m.py` via des adaptateurs déléguant au service `core/collection/`.
   - Maintien au vert de la suite de tests existante (`tests/test_collection_database_selection.py`, etc.).

---

## 3. Schéma de Données & Migration v14

Dans `core/database.py`, nous incrémentons `SCHEMA_VERSION = 14` et ajoutons la migration suivante :

```sql
-- Migration 13 -> 14 : Unification de crawl_queue sur PUUID et région
CREATE TABLE IF NOT EXISTS crawl_queue_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    puuid TEXT UNIQUE,
    region TEXT NOT NULL,
    game_name TEXT,
    tag_line TEXT,
    priority INTEGER DEFAULT 0,
    is_processed INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_crawl_v2_status_prio 
    ON crawl_queue_v2(is_processed, priority DESC, id ASC);
CREATE INDEX IF NOT EXISTS idx_crawl_v2_puuid 
    ON crawl_queue_v2(puuid);

-- Migration des données existantes
INSERT OR IGNORE INTO crawl_queue_v2 (game_name, tag_line, region, priority, is_processed, created_at)
SELECT game_name, tag_line, 'EUW', priority, is_processed, created_at FROM crawl_queue;

DROP TABLE crawl_queue;
ALTER TABLE crawl_queue_v2 RENAME TO crawl_queue;
```

---

## 4. Architecture des Modules (`core/collection/`)

Le nouveau package `core/collection/` s'organise en responsabilités découplées :

```
core/collection/
├── __init__.py           # Exports publics (CollectionService, CrawlQueueRepository, MatchIngestor, resolve_region)
├── regions.py            # Centralisation de resolve_region(tag) et _TAG_TO_REGION
├── models.py             # Dataclasses typées (CrawlItem, IngestionResult, MatchQualityFilter)
├── queue.py              # CrawlQueueRepository (opérations SQL sur crawl_queue v14)
├── ingestor.py           # MatchIngestor (filtrage, extraction sans fuite, timelines, écriture repos)
├── tracker.py            # CollectionTracker (suivi de progression thread-safe & rolling logs)
└── service.py            # CollectionService (Façade haut-niveau pour CLI, Web et workers)
```

### 4.1. Modèles (`models.py`)

```python
@dataclass
class CrawlItem:
    puuid: Optional[str]
    region: str
    game_name: Optional[str] = None
    tag_line: Optional[str] = None
    priority: int = 0
    id: Optional[int] = None

@dataclass
class IngestionResult:
    matches_saved: int = 0
    records_saved: int = 0
    timelines_saved: int = 0
    embeddings_computed: int = 0
    skipped: int = 0
    errors: int = 0
    discovered_puuids: list[tuple[str, str]] = field(default_factory=list)

@dataclass(frozen=True)
class MatchQualityFilter:
    allowed_game_modes: frozenset[str] = frozenset({"CLASSIC"})
    allowed_queue_ids: frozenset[int] = frozenset({420})
    min_duration_seconds: int = 300
```

### 4.2. File d'attente (`queue.py` — `CrawlQueueRepository`)
- Encapsule les requêtes SQLite vers `crawl_queue`.
- Méthode `enqueue_puuids(items: list[tuple[str, str, int]]) -> int` :
  `INSERT INTO crawl_queue (puuid, region, priority) VALUES (?, ?, ?) ON CONFLICT(puuid) DO UPDATE SET priority = MAX(crawl_queue.priority, excluded.priority) WHERE crawl_queue.is_processed = 0`.
- Méthode `pop_next(limit: int = 1) -> list[CrawlItem]`.
- Méthode `mark_processed(puuid: str)`.
- Méthode `get_stats() -> dict` (`total`, `pending`, `processed`).

### 4.3. Ingestion des matches (`ingestor.py` — `MatchIngestor`)
- Découpe le flux d'ingestion en étapes strictes :
  1. Filtrage des matches déjà existants via `MatchRepository.match_exists`.
  2. Récupération parallèle des détails via `RiotAPI.get_match_details_batch`.
  3. Filtrage qualité : mode `CLASSIC`, queue `420`, durée $\ge 300\text{s}$, vainqueur explicite.
  4. Extraction sans fuite : `json_utils.extract_pre_match_features` + `feature_labels.extract_labels`.
  5. Récupération systématique des timelines via `RiotAPI.get_match_timelines_batch`.
  6. Calcul des trajectoires spatiales 32D via `spatial_rnn.extract_and_store_timeline_embeddings`.
  7. Écriture atomique par lot via `MatchRepository.save_matches_batch` et `TrainingRepository.save_training_records_batch`.
  8. Extraction des participants (`puuid`, `region`) pour le snowballing.

### 4.4. Tracker de statut (`tracker.py` — `CollectionTracker`)
- Protégé par `threading.Lock`.
- Stocke les compteurs globaux, la vitesse instantanée, l'ETA et un buffer circulaire de logs récents.
- Compatible avec le format attendu par `/api/batch/status` (`status`, `total`, `current`, `player`, `logs`).

### 4.5. Façade de Service (`service.py` — `CollectionService`)
- Initialisation liée à `db_path` (avec support complet du multi-base et des tests).
- Méthodes principales :
  - `collect_player_matches(puuid, region, count, start_time)`
  - `collect_seed_players(seeds, matches_per_player, days_back)`
  - `collect_by_puuid_accounts(accounts, matches_per_player)`
  - `collect_top_leagues(regions, players_per_region, matches_per_player, days_back)`
  - `collect_batch_with_smurfs(player_list, sources, count)`
  - `run_crawler(target_matches, regions, days_back, batch_size, stop_event)`
    - Intègre le troncage périodique du WAL SQLite (`PRAGMA wal_checkpoint(TRUNCATE)`).
    - Vérifie la limite de taille disque (`NNRIOT_MAX_DB_SIZE_GB`).
    - Gère l'arrêt gracieux sur `SIGINT`/`SIGTERM`.

---

## 5. Nouveau Point d'Entrée CLI (`scripts/crawler.py`)

Un script CLI unifié offre une interface complète :
- `--target N` : Définit le quota de nouveaux matches pour le crawl haute cadence.
- `--regions R1,R2` : Régions ciblées (défaut : `EUW,KR,NA`).
- `--seeds "Nom#Tag,..."` : Ingestion rapide pour une liste de joueurs cibles.
- `--top` : Crawl des ligues Challenger/GM.
- `--status` : Affiche l'état de la base de données et de la file `crawl_queue`.
- `--db PATH` : Permet de cibler une base SQLite spécifique.

---

## 6. Adaptateurs de Rétro-compatibilité

Pour éviter toute régression dans le code existant ou les suites de tests :
- `scripts/data_collector.py` :
  - Réexporte `resolve_region`, `_build_training_record`, `DEFAULT_REGIONS`, etc.
  - Expose `batch_status` et `batch_lock` liés à l'instance globale de `CollectionTracker`.
  - Implémente `collect_training_data`, `collect_top_leagues_data`, `collect_by_puuid`, `collect_batch_with_smurfs` en déléguant à `CollectionService`.
- `scripts/background_collector.py` :
  - Délégué à `CollectionService.run_crawler()`.
- `scripts/crawler_1m.py` :
  - Délégué à `CollectionService.run_crawler()`.

---

## 7. Assainissement des Routes Flask

- `app/routes/history.py` :
  - Remplacement de l'ingestion inline par `CollectionService.collect_player_matches`.
  - Endpoints `/api/history/collect`, `/api/history/collect_batch`, `/api/batch/collect` délèguent à `CollectionService`.
  - Endpoint `/api/batch/status` interroge `CollectionService.get_default_tracker().to_dict()`.
- `app/routes/api.py` :
  - Import de `resolve_region` depuis `core.collection`.
  - Tâches auto-collect exécutées via `CollectionService.collect_seed_players`.
- `app/routes/lolpros.py` :
  - Importation des joueurs pro déléguée à `CollectionService.collect_by_puuid_accounts`.

---

## 8. Plan de Tests et de Validation

1. **Tests unitaires des nouveaux composants** :
   - `tests/test_crawl_queue_repository.py` : test des insertions par PUUID, priorités, pop et déduplication `ON CONFLICT`.
   - `tests/test_match_ingestor.py` : test du filtrage qualité, de l'extraction sans fuite et de l'ingestion systématique des timelines.
   - `tests/test_collection_service.py` : test des différents scénarios (seeds, top leagues, accounts, crawler).
2. **Tests de non-régression** :
   - Exécution de `tests/test_collection_database_selection.py` (vérification que tous les adaptateurs écrivent bien dans la base cible sélectionnée).
   - Exécution de `tests/test_collector_game_mode_guard.py` et `tests/test_timeline_collection.py`.
   - Exécution des tests d'intégration API (`tests/test_predict_api.py`, `tests/test_task_runner.py`).
