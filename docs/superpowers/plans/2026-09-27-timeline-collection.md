# Collecte des timelines Riot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Constituer un corpus reprenable de timelines brutes, contenant les positions et événements historiques des joueurs, pour de futurs modèles de prédiction avant match.

**Architecture:** Étendre le collecteur existant, avec une collecte séquentielle, des résultats HTTP explicites et un journal durable. Travailler sur une copie SQLite cohérente de la base consolidée ; conserver les timelines dans la table existante et le journal de collecte dans une base séparée. La sélection des matchs et l'évaluation de qualité sont indépendantes de l'apprentissage.

**Tech Stack:** Python du projet (`.venv312`), requests, SQLite, gzip/base64, pytest ; aucune nouvelle dépendance nécessaire.

**Spec:** Cadrage utilisateur et décisions de collecte dans ce document, section « Besoin confirmé ». Plan proposé le 27 septembre 2026 ; son écriture ne constitue pas un lancement de collecte.

## Besoin confirmé

- Priorité : maximiser la qualité prédictive avant match en apprenant des habitudes des joueurs et de leur complémentarité.
- L'utilisateur indique que la clé API est disponible. Utiliser la configuration existante, sans la demander ni l'inclure dans les commandes ou rapports.
- La demande présente porte sur le plan de collecte. La construction du modèle séquentiel et son entraînement feront l'objet d'un travail distinct.
- Collecter d'abord les parties classées solo/duo déjà présentes dans la base, puis mesurer la profondeur des historiques avant d'envisager des requêtes supplémentaires par joueur.

## Constats vérifiés

Lecture seule effectuée le 27 septembre 2026 :

| Base | Matchs | Timelines |
|---|---:|---:|
| `training_data.db` | 116 966 | 0 |
| `runs/training_data_23h_20260922_191434.db` | 83 938 | 0 |
| `runs/db_merge_20260924/consolidated.db` | 188 688 | 0 |

Les bases se recouvrent. La dernière est explicitement référencée par `runs/candidate_selective_prod/model.metrics.json`. Ses 188 688 matchs sont marqués CLASSIC ; les 500 plus récents échantillonnés ont `queueId=420`. Cela ne démontre pas que tous les matchs sont classés solo/duo. La base occupe environ 4,18 Go décimaux ; environ 65,66 Go étaient libres sur le volume lors de l'audit. Recalculer l'espace au lancement.

Le nombre de joueurs disposant de 5, 10 ou 20 matchs antérieurs n'a pas été établi : l'audit SQL borné en temps a interrompu ce calcul. Ne pas présenter ces profondeurs comme acquises.

Le code conserve le JSON brut compressé (`core/database.py`, `save_match_timelines_batch`). Les lacunes observées concernent notamment :

- `get_match_timelines_batch` absorbe `RiotAuthError` ; le backfill peut alors enregistrer des NULL pour une clé refusée.
- Les timeouts, 429 et 5xx épuisés sont également confondus avec une absence de timeline.
- Les NULL sont ensuite exclus des reprises ; la limite de matchs peut être dépassée ; une exception répétée peut boucler sur le même lot.
- Le limiteur partage 18 requêtes/s et 95/120 s dans un processus ; les en-têtes sont enregistrés mais ne pilotent pas actuellement le débit.
- `NNRIOT_FETCH_TIMELINES` est désactivé par défaut. Les collecteurs de nouveaux matchs ne suffisent donc pas à compléter l'historique existant.

## Global Constraints

- SQLite opérationnel : conserver le schéma v12 et la table `match_timelines` existante. Le nouveau journal possède son propre schéma versionné ; aucun remplacement de base active dans ce plan.
- Accès SQLite centralisé dans `core/database.py`, transactions brèves, WAL et fermeture des connexions. Suivre `docs/database_merge.md` pour copie cohérente, intégrité et publication éventuelle.
- Source : `D:/projet/NNriot/runs/db_merge_20260924/consolidated.db`, ouverte en lecture seule pour la préparation.
- Destination : `D:/projet/NNriot/runs/timeline_collection_20260927/collection.db`. Refuser d'écraser un fichier existant ; une reprise utilise les fichiers déjà créés et leur manifeste.
- Conserver champions, rôles, équipes, PUUID, timestamps, positions et événements bruts. Aucun tri destructif, changement de coordonnées ou remplissage artificiel de positions manquantes.
- Futur entraînement : un historique n'est admissible que si le match historique s'est terminé avant le début du match cible. La date de collecte ne remplace pas la date du match.
- Les conventions d'anti-symétrie et de permutation du projet restent inchangées ; cette collecte ne change aucun tenseur ou modèle.
- Un seul processus de collecte utilisant cette clé pendant le run. Ne pas interrompre silencieusement un autre travail : détecter un conflit avant lancement et le signaler.
- Un worker HTTP pour cette version. Les quotas restent activés ; rejeter un lancement réel avec `RIOT_DISABLE_RATE_LIMIT=1`.
- Journaliser les résultats et les statuts HTTP, jamais la clé ou les en-têtes d'authentification.

## Review Focus

1. Clé refusée au milieu d'un run : arrêt explicite, succès déjà écrits conservés et matchs restants reprenables — tâche 2.
2. Interruption entre l'écriture d'une timeline et la mise à jour du journal : réconciliation sans nouveau téléchargement — tâche 3.
3. Ancien NULL ou JSON ne correspondant pas au match demandé : requalification explicite, aucun faux succès — tâches 3 et 4.
4. Match CLASSIC hors solo/duo ou historique clairsemé : sélection par queueId, couverture temporelle mesurée sans extrapolation — tâches 1 et 4.
5. Quota plus bas que prévu ou disque presque plein : attente ou arrêt reprenable, sans boucle serrée ni perte des résultats — tâches 2 et 3.

## Fichiers et responsabilités

| Fichier | Responsabilité |
|---|---|
| `core/riot_api.py` | Résultat typé de timeline, statut des échecs, respect des quotas et fermeture des sessions |
| `core/database.py` | Copie SQLite, lecture des candidats, écriture des timelines et accès au journal séparé |
| `core/timeline_collection.py` (nouveau) | Validation des données, règles de reprise, calcul du rapport de couverture |
| `scripts/prepare_timeline_collection.py` (nouveau) | Copie initiale, inventaire et manifeste immuable |
| `scripts/backfill_timelines.py` | CLI de collecte bornée et reprenable utilisant les composants précédents |
| `tests/test_riot_api_timeline.py`, `tests/test_riot_api_hardening.py` | HTTP, authentification et quotas simulés |
| `tests/test_backfill_timelines.py` | Reprise, budget, journal et interruption |
| `tests/test_timeline_collection.py` | Sélection, positions, identités et couverture |
| `docs/timeline_collection.md` (nouveau) | Procédure d'exploitation et lecture des rapports |

## Task 1: Préparer la copie et un inventaire solo/duo reproductible

**Files:** `core/database.py`, nouveau `scripts/prepare_timeline_collection.py`, `tests/test_timeline_collection.py`.

**Interfaces:**

```python
def prepare_timeline_collection(source: str, output_dir: str, queue_id: int = 420) -> dict:
    """Create collection.db, state.db, manifest.json and candidates.jsonl; refuse overwrite."""
```

- [ ] Ajouter les tests suivants sur de petites bases temporaires : un match 420 accepté, un 440 et un 400 exclus, JSON invalide compté à part, source inchangée, destination existante refusée, deux lectures donnant le même ordre.
- [ ] Exécuter `python -m pytest tests/test_timeline_collection.py -q` et constater les échecs liés à l'interface absente.
- [ ] Implémenter la copie avec l'API SQLite backup, et non une copie du fichier principal susceptible d'ignorer le WAL :

```python
from pathlib import Path
from contextlib import closing
import sqlite3

destination_dir = Path(output_dir).resolve()
destination_dir.mkdir(parents=True, exist_ok=False)
destination = destination_dir / "collection.db"
source_uri = Path(source).resolve().as_uri() + "?mode=ro"
with closing(sqlite3.connect(source_uri, uri=True)) as src:
    src.execute("PRAGMA query_only=ON")
    with closing(sqlite3.connect(destination)) as dst:
        src.backup(dst, pages=2048)
        assert dst.execute("PRAGMA quick_check").fetchone()[0] == "ok"
```

La fonction enveloppante refuse source=destination et toute destination existante. En production, remplacer l'assertion de contrôle par une erreur explicite conservée dans le rapport. Vérifier aussi la version et les nombres de matchs/timelines sur la copie. Un échec de préparation conserve le dossier partiel identifié comme incomplet ; il ne devient pas une collecte reprenable sans manifeste final validé.

- [ ] Parcourir les matchs de la copie une seule fois, avec `fetchmany(500)`, en ordre `game_creation_time DESC, match_id ASC`. Décoder `raw_json` selon le format existant et retenir `queueId == 420`, `mapId == 11`, dix participants identifiables, timestamp positif. Classer les autres cas dans le rapport, sans les supprimer de la copie.
- [ ] Écrire `candidates.jsonl` : `match_id`, `game_start_ms`, `game_end_ms`, `patch`, `platform`, `regional_host`, `puuids`. Préférer `gameEndTimestamp` ; sinon calculer une fin avec la durée uniquement après validation de son unité selon la version des données. Un temps de fin inconnu demeure inconnu.
- [ ] Écrire le manifeste : chemins absolus, date UTC, queue, nombres, intervalle temporel, régions, SHA-256 de `candidates.jsonl` et identifiant du jeu de collecte. Ce manifeste fige la sélection et l'ordre ; la collecte ne reconstruit pas implicitement une sélection différente.
- [ ] Rejouer les tests et vérifier que le schéma opérationnel demeure v12. Commit ciblé des fichiers de code et tests, jamais des bases ou de la clé.

## Task 2: Distinguer les résultats HTTP et respecter les quotas

**Files:** `core/riot_api.py`, `tests/test_riot_api_timeline.py`, `tests/test_riot_api_hardening.py`.

**Interfaces:** conserver les méthodes existantes pour leurs consommateurs ; ajouter un chemin explicite utilisé par le backfill :

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class TimelineFetchResult:
    match_id: str
    status: Literal["ok", "not_found", "retryable", "invalid"]
    payload: dict | None
    http_status: int | None
    retry_after_s: float | None
    reason: str | None

# Méthode de RiotAPI ; RiotAuthError est propagée, jamais convertie en résultat absent.
def get_match_timeline_result(self, match_id: str) -> TimelineFetchResult:
    ...
```

La signature ci-dessus est un contrat, pas un corps d'implémentation à recopier. Le chemin HTTP doit conserver le dernier statut et la dernière cause après ses trois essais ; le `None` historique ne permet pas cette classification. Étendre `_make_request` par un mode explicite retournant ou levant ces informations, en conservant son comportement par défaut pour les autres endpoints.

- [ ] Ajouter des tests avec réponses simulées : 200 JSON, 404, 503 épuisé, timeout épuisé, JSON illisible, 401 et 403 non récupérables. Exemple de test de propagation sur le batch existant :

```python
def test_timeline_batch_propagates_auth_error(monkeypatch):
    from core.riot_api import RiotAPI, RiotAuthError
    import pytest
    monkeypatch.setenv("RIOT_API_KEY", "test-only")
    def fail(self, match_id):
        raise RiotAuthError("HTTP 403")
    monkeypatch.setattr(RiotAPI, "get_match_timeline", fail)
    with RiotAPI(region="EUW") as api:
        with pytest.raises(RiotAuthError):
            api.get_match_timelines_batch(["EUW1_1"], max_workers=1)
```

- [ ] Exécuter les deux fichiers de tests pour constater les régressions attendues, puis implémenter : 401/403 arrêt ; 404 statut indisponible à recontrôler ; 429/5xx/réseau statut réessayable ; erreur JSON statut invalide ; succès dict conservé intégralement. Ne pas utiliser la vérité booléenne de `requests.Response` pour distinguer absence de réponse et statut HTTP d'erreur.
- [ ] Conserver une cadence conservatrice plafonnée à 95 requêtes/120 s et 18/s ; la réduire si les limites application/méthode observées imposent moins. Comptabiliser chaque tentative HTTP. Sur 429, respecter `Retry-After` dans le scheduler séquentiel ; valeur manquante ou illisible : attente conservatrice de 120 s. Utiliser une horloge monotone pour les fenêtres.
- [ ] Tester les quotas avec horloge et attente simulées : aucune attente réelle dans pytest ; 429 suivi de succès, quota méthode inférieur, en-têtes absents et invalides. Fermer les clients en `finally`. Vérifier l'absence de clé dans les logs.
- [ ] Lancer `python -m pytest tests/test_riot_api_timeline.py tests/test_riot_api_hardening.py -q`, puis commit ciblé.

## Task 3: Rendre la collecte reprenable et strictement bornée

**Files:** `core/database.py`, `core/timeline_collection.py`, `scripts/backfill_timelines.py`, `tests/test_backfill_timelines.py`.

**Interfaces:**

```python
# Dans core/database.py ; toutes les lectures/écritures SQLite passent par ce module.
def init_timeline_collection_state(state_path: str, manifest_sha256: str) -> None: ...
def record_timeline_attempt(state_path: str, row: dict) -> None: ...
def list_due_timeline_matches(state_path: str, now_ms: int, limit: int) -> list[dict]: ...

# Dans core/timeline_collection.py.
def collect_timelines(db_path: str, state_path: str, manifest_path: str,
                      limit: int, report_path: str, min_free_gb: float = 10.0,
                      retry_errors: bool = False) -> dict: ...
```

Les corps de ces fonctions sont à implémenter selon les transitions ci-dessous. Journal séparé `state.db`, schéma v1 :

```sql
CREATE TABLE collection_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE timeline_jobs (
    match_id TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK(state IN
      ('pending','in_progress','saved','retryable','not_found','invalid','unsupported')),
    attempts INTEGER NOT NULL DEFAULT 0,
    not_found_checks INTEGER NOT NULL DEFAULT 0,
    last_http_status INTEGER,
    next_attempt_ms INTEGER,
    last_error TEXT,
    updated_at_ms INTEGER NOT NULL
);
CREATE INDEX idx_timeline_due ON timeline_jobs(state, next_attempt_ms, ordinal);
```

- [ ] Écrire les tests : limite exacte de cinq matchs, limite négative/nulle rejetée ; succès+timeout+404 ; auth refusée après un succès ; crash avant/après écriture ; reprise d'un ancien NULL ; espace disque sous le seuil ; journal appartenant à un autre manifeste refusé.
- [ ] Adapter les anciennes attentes qui acceptent un dépassement ou un NULL ambigu. Exécuter `python -m pytest tests/test_backfill_timelines.py -q` pour vérifier les échecs.
- [ ] Implémenter une sélection figée des prochains IDs dus. `limit` compte les IDs distincts pris en charge pendant ce run, y compris routage non reconnu ; les retries HTTP sont comptés séparément. Un même ID n'est traité qu'une fois par run, avec au plus trois essais HTTP internes. Le run se termine quand aucun travail n'est dû, en indiquant la prochaine échéance.
- [ ] Appliquer les transitions :

| Résultat | Action durable |
|---|---|
| Succès valide | Écrire le brut dans `collection.db`, puis marquer `saved` |
| Timeout/5xx/429 épuisé | `retryable`, prochaine tentative au plus tôt dans 5 min et après Retry-After |
| 404 | `not_found`, recontrôle au plus tôt dans 24 h ; après trois contrôles, laisser visible comme indisponible, sans boucle automatique |
| 401/403 | Arrêter le run ; remettre le job en attente et conserver la progression |
| Préfixe non reconnu | `unsupported`, aucun appel API ; anomalie de routage visible |
| JSON illisible ou mauvais matchId | `invalid`, raison explicite ; aucun succès enregistré |

Les erreurs réessayables atteignant cinq runs restent visibles avec prochaine échéance vide et nécessitent une relance explicite des erreurs. Elles ne deviennent jamais une preuve d'absence. `attempts` compte les prises en charge par run ; `not_found_checks` compte seulement les réponses 404 finales. Les essais HTTP internes restent des compteurs de rapport distincts. Stocker l'heure UTC pour la reprise après redémarrage.

- [ ] À la reprise, pour chaque job `in_progress` et chaque ancien NULL, consulter le payload opérationnel : succès déjà valide -> `saved` ; absence/NULL -> `pending`. Écrire le brut avant le statut assure la récupération du crash entre les deux bases. Ajouter une écriture conditionnelle qui remplace seulement un NULL, jamais une timeline valide.
- [ ] Vérifier l'espace libre avant chaque requête et chaque écriture : arrêter proprement sous 10 Go décimaux, avec code de sortie non nul et rapport conservé. Respecter également `NNRIOT_MAX_DB_SIZE_GB`. Poser un verrou de run sur le journal pour refuser deux collecteurs concurrents.
- [ ] Ajouter les options CLI `--state`, `--manifest`, `--report`, `--min-free-gb`, `--retry-errors`, avec `--chunk 1` obligatoire pour ce parcours. `--retry-errors` remet en attente les erreurs épuisées, les invalides et les 404 déjà contrôlés trois fois, sans effacer leurs compteurs historiques ; les routes non reconnues restent bloquées jusqu'à correction de leur mapping. Chaque ID demeure limité à une prise en charge dans le run. Refuser un chemin de rapport déjà utilisé par un ancien run ; pendant le run, actualiser son rapport via fichier temporaire puis renommage. Gérer Ctrl+C en fermant clients et transactions.
- [ ] Rejouer les tests et committer les seuls fichiers de la tâche.

## Task 4: Mesurer les coordonnées et les historiques réellement utilisables

**Files:** `core/timeline_collection.py`, `tests/test_timeline_collection.py`, `docs/timeline_collection.md`.

**Interfaces:**

```python
def inspect_timeline(match_id: str, payload: dict) -> dict:
    """Return identity_ok, frames, participant_frames, positions_valid,
    position_coverage, frame_interval_ms, observed_gaps_ms, last_timestamp_ms,
    participants_mapped, events_with_position and warnings."""

def build_collection_report(db_path: str, state_path: str, manifest_path: str) -> dict:
    """Aggregate collection states, quality, sizes, speed and historical coverage."""
```

- [ ] Tester une timeline complète, un joueur manquant, des frames vides, un intervalle variable, des timestamps décroissants, une coordonnée non finie, un matchId incorrect et un événement sans position. Exemple de propriété à vérifier avec un payload minimal :

```python
def test_position_coverage_counts_missing_players():
    from core.timeline_collection import inspect_timeline
    payload = {
        "metadata": {"matchId": "EUW1_1"},
        "info": {"frameInterval": 60000, "frames": [{
            "timestamp": 60000,
            "participantFrames": {str(i): {"position": {"x": 100, "y": 200}}
                                  for i in range(1, 10)},
            "events": []}]}}
    report = inspect_timeline("EUW1_1", payload)
    assert report["positions_valid"] == 9
    assert report["position_coverage"] == 0.9
```

- [ ] Calculer la couverture par rapport à dix joueurs par frame, et vérifier le mapping participantId/PUUID contre le match brut. Conserver une timeline partielle reçue avec le bon matchId ; la rapporter comme partielle, sans inventer les positions ni la compter dans les données complètes.
- [ ] Définir une timeline exploitable pour le pilote : identité cohérente, dix joueurs reliés, au moins deux frames horodatées, ordre temporel non décroissant, positions numériques finies pour au moins 95 % des couples frame/joueur attendus. Consigner séparément fin tronquée et trous ; une fin tronquée n'est pas complète. Le seuil sert au contrôle initial, pas à une suppression du corpus brut.
- [ ] Le rapport doit séparer sélectionné, tenté, reçu, exploitable, partiel, absent, en attente de retry et invalide. Ajouter débit réel, nombre d'appels/retries/429, octets compressés moyens et p95, croissance SQLite/WAL, couverture par région/patch/date et échéance du prochain retry.
- [ ] Mesurer pour les matchs cibles la proportion de joueurs avec au moins 1/5/10/20 timelines antérieures admissibles, ainsi que la proportion de matchs dont les dix joueurs ont chacun au moins 5 puis 10 historiques. Utiliser les temps de fin, pas uniquement l'ordre des débuts. Inclure les joueurs inconnus dans le dénominateur ; publier séparément les fins inconnues.
- [ ] Écrire la procédure de relance et les statuts dans la documentation ; exécuter les tests et committer.

## Task 5: Exécuter le pilote et décider de l'extension sur des mesures

**Files:** artefacts locaux sous `runs/timeline_collection_20260927/` et `docs/timeline_collection.md`.

Les commandes ci-dessous décrivent les interfaces à livrer par les tâches 1–4. Elles ne sont pas toutes disponibles dans le code actuel.

- [ ] Lancer les vérifications ciblées après implémentation :

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_riot_api_timeline.py tests/test_riot_api_hardening.py tests/test_backfill_timelines.py tests/test_timeline_collection.py tests/test_database_v11_migration.py tests/test_collection_database_selection.py -q
```

- [ ] Préparer la copie et son inventaire :

```powershell
.\.venv312\Scripts\python.exe scripts/prepare_timeline_collection.py `
  --source runs/db_merge_20260924/consolidated.db `
  --output runs/timeline_collection_20260927 --queue-id 420
```

- [ ] Pilote : au plus 200 matchs sélectionnés, un worker. Le premier appel valide réellement l'accès à la route ; une clé déclarée disponible n'est pas une mesure du quota ou de la disponibilité des anciennes parties.

```powershell
.\.venv312\Scripts\python.exe scripts/backfill_timelines.py `
  --db runs/timeline_collection_20260927/collection.db `
  --state runs/timeline_collection_20260927/state.db `
  --manifest runs/timeline_collection_20260927/manifest.json `
  --report runs/timeline_collection_20260927/pilot-200.json `
  --limit 200 --chunk 1 --min-free-gb 10
```

- [ ] Vérifier le rapport : zéro contamination par mauvais ID, reprise testée, au moins 95 % de timelines exploitables parmi les réponses 200, pas d'échec auth non traité, pas de retry immédiat en boucle. Un taux élevé de 404 sur des données anciennes déclenche un diagnostic d'âge/routage, pas une promesse de couverture totale.
- [ ] Si ces contrôles passent, traiter jusqu'à 4 800 nouveaux IDs avec les mêmes fichiers et `--report runs/timeline_collection_20260927/pilot-5000.json`. Reporter à la fois les chiffres du run et les totaux cumulés : les retries ne doivent pas faire passer artificiellement le corpus au seuil de 5 000. Si un run consomme une partie de son budget en retries, poursuivre les runs bornés jusqu'à 5 000 IDs distincts tentés, et toujours donner séparément le nombre de timelines reçues/exploitables.
- [ ] Recalculer durée et disque nécessaires à partir des mesures. Si la projection respecte le plafond de base et la réserve de 10 Go, poursuivre les candidats solo/duo restants par runs d'au plus 5 000 IDs, du plus récent au plus ancien. Chaque run écrit un rapport numéroté distinct.
- [ ] À chaque lot, mesurer la profondeur d'historique. Si elle reste faible, présenter une extension ciblée par joueur, avec son budget d'appels, avant d'ajouter de nouvelles parties au manifeste. Le nombre de timelines seul n'établit pas l'utilité pour la prédiction avant match.
- [ ] Rapport final : couverture atteinte et manquante, raisons, contrôle SQLite, espace consommé, reproductibilité de la reprise et chemins exacts. Ne pas déclarer la collecte complète tant qu'il reste des jobs dus/réessayables ; distinguer collecte terminée et ressources indisponibles.

## Budget indicatif

Les quotas effectifs de la clé n'ont pas été testés dans cette phase de planification. À titre de scénario conservateur, le limiteur actuel à 95 requêtes par 120 secondes donne environ 2 850 appels/heure. Sans retries, latence supplémentaire ou autre consommateur :

| Nombre de requêtes | Ordre de grandeur au débit conservateur |
|---:|---:|
| 200 | 4 à 5 minutes |
| 5 000 | environ 1 h 45 |
| 188 688 | environ 66 heures |

Ces durées sont des estimations de débit, pas des engagements ; le dernier chiffre est un plafond de corpus brut, pas le nombre solo/duo vérifié. Un quota méthode inférieur rallonge le temps. Le pilote remplacera ces estimations par le débit observé.

Stockage : `taille_cible ≈ taille_copie + N × octets_moyens_stockés`, avec marge pour index/WAL et variation de taille. Utiliser aussi le p95 pour une projection conservatrice ; ne pas inventer une taille moyenne avant le pilote.

La collecte des futures parties devra ensuite passer par le même mécanisme de journalisation. Activer seulement le drapeau historique ne fournit pas à lui seul les garanties de reprise définies ici. Cette intégration continue vient après le backfill et sera dimensionnée selon les quotas réellement disponibles.

## Sources et vérification du plan

- [Riot Developer Portal — quotas et statuts HTTP](https://developer.riotgames.com/docs/portal) : limites application/méthode/service, respect de Retry-After ; un 404 ne prouve pas une absence permanente. Les plafonds publiés pour une clé personnelle ne permettent pas d'identifier le type de la clé de ce projet.
- [Exemple RiotTuxedo — Match-V5 timeline](https://gist.github.com/RiotTuxedo/ab04ac027546f711ce820faed394295b) : positions des joueurs, événements et frameInterval ; la fréquence réelle sera mesurée sur le pilote.
- Code inspecté : `core/database.py`, `core/riot_api.py`, `core/config.py`, `scripts/backfill_timelines.py`, `scripts/data_collector.py`, tests de timeline/auth/backfill, `docs/database_merge.md`.

Auto-revue effectuée avant remise : signatures et options rapprochées entre tâches, reprise du crash entre les deux bases explicitée, refus d'écraser un résultat prévu, compteurs de 404 séparés des autres échecs et distinction « reçu / exploitable / antérieur » conservée. Les signatures contenant des points de suspension sont des contrats de fonctions à créer ; les transitions et tests attendus sont définis dans les tâches correspondantes. Le lancement réel et les tests du futur code ne sont pas accomplis par la rédaction de ce plan.
