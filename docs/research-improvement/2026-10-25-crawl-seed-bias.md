# Biais de survivant de la collecte : mesure, drapeau et neutralisation (25 octobre 2026)

## Résumé

- **Le biais de survivant est confirmé, et dépasse largement les seeds du crawler.** Toutes les collectes de NNriot sont parties de joueurs choisis pour leur rang. Les parties récupérées au moment de ce choix sur-représentent leurs victoires :
  - **Seeds ladder** (150, le 11 octobre) : 54,3 % de victoires avant leur entrée dans la file, 49,6 % après. Dans le train, l'équipe d'une seed gagne 56,2 % (10 895 matchs), comme mesuré le 24 octobre.
  - **Anchors legacy** (table `players` : collectes top-ladder et pros, enregistrées en mai) : avant leur dernière sauvegarde, l'équipe qui est seule à en porter un gagne **63,6 %** (17 049 matchs du train). **100 % des matchs d'avril–mai du train** contiennent un tel anchor.
  - **Crawl postérieur au match** : sans aucun anchor, l'équipe qui compte plus de joueurs crawlés *après* la partie gagne 56,5 % (28 567 matchs du train). Quand l'écart vient de joueurs crawlés *avant*, elle gagne 49,1 % (1 315). C'est une fuite de disponibilité : l'historique d'un joueur n'existe dans la base que s'il a été crawlé ensuite.
- **Drapeau disponible.** Le module `core/collection/provenance.py` et le fichier compagnon `runs/crawl_provenance_20261025/crawl_provenance.csv` (123 600 matchs train + validation) le fournissent. Le schéma SQLite et le dataset aligné ne sont pas modifiés.
- **Neutralisation à l'entraînement.**
  - L'exclusion dégrade le modèle : exclure les seeds coûte +0,00085 de log-loss en dev [+0,00029 ; +0,00133].
  - L'IPW (`ipw_anchor_and_crawled`) est neutre en accuracy : −0,04 pt en dev [−0,49 ; +0,44], +0,20 pt en validation équilibrée [−0,68 ; +1,05]. Il divise par deux le penchant du modèle vers l'équipe la plus crawlée.
  - **Aucun gain d'accuracy n'est à attendre de ce côté** : le modèle du 24 octobre n'exploitait déjà que peu l'artefact.
- **Recommandation.** Pondérer par IPW le mélange et les arbres, ne rien exclure et rapporter systématiquement les sous-ensembles équilibrés. Corriger ensuite à la source : enregistrer l'heure d'ingestion et utiliser les seeds pour découvrir la frontière, sans en faire des cibles. Sur les sous-ensembles équilibrés, le modèle gelé fait 59,72 % en dev et 59,37 % en validation. L'écart dev → validation du 24 octobre tenait surtout aux matchs déséquilibrés.

## Ce que la collecte sélectionne

Trois chemins de collecte ont alimenté `training_data.db`. Tous partent de joueurs choisis pour leur rang.

1. **Crawler continu (`CollectionService.run_crawler`, octobre).** Le 11 octobre 2026 à 09:25 UTC, 150 seeds (50 par région, EUW/KR/NA) issus de `get_top_players` entrent dans `crawl_queue` en priorité 10. Aucune seed de priorité 5 n'existe : la file ne s'est jamais épuisée. La boule de neige (priorité 0) a traité 21 613 joueurs entre le 11 et le 18 octobre ; 27 647 restent en attente.
2. **Collectes « legacy » (mars–août).** `collect_top_leagues`, les imports de pros (lolpros, Liquipedia) et les lots manuels n'écrivent pas dans `crawl_queue` mais dans `players` (`save_player`). La table compte 6 924 joueurs, dont 6 800 qui ne sont pas des seeds ladder, enregistrés pour la plupart en mai 2026. `last_updated` y vaut la **dernière** sauvegarde.
3. **Crawler de septembre (`scripts/crawler_1m.py`).** Il est amorcé sur Challenger, puis Grandmaster, et ne garde ses états que dans `crawler_frontier.json` (22 153 joueurs visités). La liste de ses seeds n'a pas été conservée.

Un joueur choisi pour son rang l'est *parce qu'il* a gagné récemment. Les parties de son historique récupérées au moment du choix sur-représentent donc ses victoires. Après son entrée, il retombe à 50 %.

## Mesures (lecture seule)

Script : `scripts/diagnose_crawl_seed_bias.py`. Sorties : `runs/crawl_provenance_20261025/` (`report.json`, `crawl_provenance.csv`, `registry.json`, `manifest.json`). La partie base porte sur les parties CLASSIC d'au moins 300 s créées strictement avant la première création du test scellé (17 octobre 2026, 11:43:50 UTC), soit 135 275 matchs 5 contre 5. La partie dataset lit uniquement `train.json` et `validation.json` ; `sealed_test*` n'est jamais ouvert. Intervalles : Wilson à 95 %.

### Taux de victoire par joueur (toutes parties de la base)

| Catégorie | Parties | Victoires | Wilson 95 % |
|---|---:|---:|---|
| Seed ladder, partie **avant** son entrée dans la file | 23 511 | 54,33 % | [53,69 ; 54,97] |
| Seed ladder, partie **après** son entrée | 2 335 | 49,64 % | [47,61 ; 51,66] |
| Anchor legacy, avant sa dernière sauvegarde | 301 838 | 52,96 % | [52,78 ; 53,13] |
| Anchor legacy, après | 134 588 | 50,39 % | [50,12 ; 50,65] |
| Crawlé en boule de neige | 468 897 | 50,11 % | [49,97 ; 50,26] |
| En file, pas encore traité | 75 840 | 47,09 % | [46,74 ; 47,45] |
| Jamais en file | 345 741 | 47,46 % | [47,29 ; 47,63] |

Ces chiffres diffèrent un peu de ceux du 24 octobre (53,8 / 50,7 / 49,0 %), qui ne séparaient ni l'avant de l'après-entrée, ni les anchors legacy, et ne filtraient pas les remakes. Le constat est le même et plus net. **Les seeds ne gagnent plus après leur entrée.** L'excès de victoires concerne les parties récupérées au moment de leur sélection.

### Équipe porteuse de l'artefact (matchs où une seule équipe le porte)

| Artefact | Base | Train aligné | Validation |
|---|---|---|---|
| Seed ladder (toute date) | 55,92 % (12 983) | 56,22 % (10 895) | 44,2 % (77) |
| Seed avant son entrée | 56,30 % (12 290) | 56,63 % (10 285) | — (0) |
| Seed uniquement après son entrée | 49,21 % (693) | 49,34 % (610) | 44,2 % (77) |
| **Anchor quelconque avant entrée/sauvegarde** | **63,02 % (20 025)** | **63,56 % (17 049)** | — (0) |
| Plus de crawlés, aucune seed | 55,18 % (60 053) | 55,54 % (52 908) | 53,11 % (1 883) |
| Plus de crawlés, aucun anchor avant entrée | 56,43 % (29 407) | 56,74 % (27 254) | 53,18 % (1 888) |
| … dont crawlés **après** le match | 55,93 % (31 422) | 56,48 % (28 567) | 50,70 % (2 558) |
| … dont crawlés **avant** le match | 51,35 % (2 440) | 49,05 % (1 315) | 54,19 % (1 039) |

Le chiffre du 24 octobre est retrouvé : dans le train, l'équipe d'une seed gagne 56,2 %. Il ne mesure pourtant que la plus petite part de l'artefact.

- **Anchors legacy.** Avant leur sauvegarde, l'équipe qui porte seule un anchor gagne 60,7 % des matchs d'avril et 66,4 % de ceux de mai. **100 % des matchs d'avril et de mai du train** (56 734 matchs, 48 % du train) contiennent au moins un anchor d'avant sa sauvegarde. Sur l'ensemble du train, 17 049 matchs n'en portent que d'un côté, dont 13 571 en avril–mai.
- **Crawl postérieur au match.** L'équipe qui compte plus de joueurs crawlés gagne nettement plus, même sans aucun anchor. L'effet vient des joueurs crawlés *après* la partie : 56,5 % dans le train, contre 49,1 % pour ceux crawlés avant. Il croît avec l'écart : en septembre, 54,2 %, 58,7 % puis 64,2 % pour 1, 2 et 3 crawlés d'écart ou plus. Il disparaît presque en validation (50,7 %), où le crawl a suivi la partie d'un à deux jours au plus.
- **Crawler de septembre** (`sept_frontier_check.py`). Sur les matchs sans anchor, l'équipe qui compte le plus de joueurs visités par ce crawler gagne 57,3 % en septembre (19 709 matchs). Elle gagne 49,6 % en octobre (12 463) et 53,5 % en validation (2 360).

**Interprétation.** Le statut « crawlé » est décidé *après* le match. Il dépend de la fréquence à laquelle le joueur réapparaît ensuite dans des historiques récupérés, et donc probablement de son envie de rejouer après une victoire. Or l'historique d'un joueur n'entre dans la base que s'il est crawlé. Les features d'historique, point-in-time par horodatage, ne le sont donc pas **par disponibilité**. Le manifeste du dataset le signalait déjà (« ingestion timestamps unknown »). C'est une hypothèse, mais trois observations la soutiennent : l'opposition avant/après, la relation dose–effet et la disparition de l'effet quand le délai est court.

**Côté Red.** Red gagne 53,47 % du train, mais 51,41 % sur le sous-ensemble strictement équilibré (29 030 matchs) et 51,26 % en validation. Une partie de l'avantage de côté appris sur le train est donc un artefact de collecte.

## Drapeau par match

Module : `core/collection/provenance.py`, couche données, sans dépendance au modèle. Fichier compagnon : `runs/crawl_provenance_20261025/crawl_provenance.csv`, une ligne par match train/validation du dataset aligné, clé `match_id`. Le schéma SQLite et le dataset aligné ne sont pas modifiés. Le manifeste lie le fichier à `dataset_sha256` et `split_sha256`, et vérifie les hachages de `train.json` et `validation.json`. `registry.json` fige le registre utilisé (seeds, anchors, horaires de crawl), au cas où `crawl_queue` serait purgée ou fusionnée.

| Colonne | Sens |
|---|---|
| `seeds_a/_b`, `pre_entry_seeds_a/_b` | seeds ladder (`crawl_queue.priority >= 5`), et celles dont l'entrée est postérieure à la création du match |
| `legacy_anchors_a/_b`, `pre_entry_anchors_a/_b` | lignes `players` hors seeds, et celles dont la dernière sauvegarde est postérieure au match |
| `crawled_a/_b`, `crawled_after_a/_b` | joueurs traités dans `crawl_queue`, et ceux traités à partir de la création du match |
| `has_ladder_seed`, `has_pre_entry_seed`, `has_pre_entry_anchor` | drapeaux de match |
| `crawled_imbalance`, `crawled_after_imbalance` | B − A, même orientation que `winner = 1` |
| `balanced` | aucune seed et autant de crawlés des deux côtés (définition du 24 octobre) |

Le sous-ensemble `strict_balanced` (aucun anchor avant entrée et autant de crawlés) se déduit des colonnes.

Ces comptes décrivent la provenance d'un match et **ne doivent jamais servir d'entrée de modèle**. Ils servent à pondérer ou exclure à l'entraînement et à découper les rapports. Les tests (`tests/core/test_crawl_provenance.py`, `tests/test_diagnose_crawl_seed_bias.py`) couvrent plusieurs propriétés :

- la frontière avant/après entrée ;
- le signe du déséquilibre ;
- l'invariance par échange d'équipes (le drapeau échangé est le miroir exact) et par permutation intra-équipe ;
- l'ouverture en lecture seule de la base ;
- l'absence de lecture du test scellé ;
- la coupure temporelle.

## Neutralisation à l'entraînement

Script : `docs/research-improvement/crawl_seed_bias_20261025/neutralize.py`. Il reprend à l'identique la configuration gelée le 24 octobre :

- table `feat_v4m.npz`, SHA-256 vérifié ;
- LR avec C = 0,001 ;
- LightGBM « B » avec échange d'équipes ;
- mélange 0,4 LR + 0,6 LGB.

Seuls changent les matchs d'entraînement ou leurs poids. Le dev (12 000 derniers matchs du train) et la validation ne sont jamais modifiés. La référence reproduit exactement les chiffres gelés : dev 59,62 % / 0,66364, validation 58,30 % / 0,67204, LightGBM à 2 126 itérations.

- **Exclusion** : les matchs portant le drapeau sont retirés du fit et du train complet.
- **IPW** (pondération inverse de la probabilité de sélection) : dans chaque strate où un seul côté porte l'artefact, les victoires et défaites de ce côté sont repondérées par 0,5/p et 0,5/(1−p). L'artefact gagne alors exactement la moitié de la strate, dont le poids total est conservé.
  - Strates `ipw_anchor_and_crawled` : côté porteur d'un anchor d'avant entrée ; puis, parmi les autres matchs, écart signé de crawlés (plafonné à ±3).
  - Taux corrigés dans le fit : anchor côté Red 65,95 % (11 703 matchs), côté Blue 58,32 % (5 346) ; crawlés de 53,4 à 65,7 % selon l'écart.

Le dev et la validation ne contiennent **aucun** match avec un anchor d'avant entrée. Le dev porte en revanche un fort artefact de crawl postérieur : l'équipe la plus crawlée y gagne 59,1 %, contre 53,2 % en validation. Le « penchant » mesure ce que le modèle en a appris : c'est la probabilité moyenne qu'il prédit pour l'équipe la plus crawlée.

### Mélange 0,4 LR + 0,6 LGB (modèle retenu le 24 octobre)

| Variante | Matchs de train | Dev : acc. / log-loss | Dev équilibré | Validation | Validation équilibrée | Penchant val. (observé 53,2 %) |
|---|---:|---|---|---|---|---:|
| Référence gelée | 118 600 | 59,62 % / 0,66364 | 59,72 % / 0,66340 | 58,30 % / 0,67204 | 59,37 % / 0,66948 | 52,35 % |
| Exclusion des seeds | 104 053 | 59,46 % / 0,66449 | 59,48 % / 0,66448 | 58,10 % / 0,67221 | 59,03 % / 0,67038 | 52,30 % |
| IPW anchors + crawlés | 118 600 pondérés | 59,58 % / 0,66345 | 59,72 % / 0,66310 | 58,28 % / 0,67228 | **59,57 % / 0,66906** | **51,11 %** |

Différences appariées avec la référence (bootstrap par blocs de 100 matchs consécutifs, 2 000 tirages, IC 95 %) :

| Variante | Sous-ensemble | Δ accuracy | Δ log-loss |
|---|---|---|---|
| Exclusion des seeds | dev | −0,16 pt [−0,51 ; +0,22] | **+0,00085 [+0,00029 ; +0,00133]** |
| | dev équilibré | −0,25 pt [−0,66 ; +0,18] | **+0,00107 [+0,00041 ; +0,00164]** |
| | validation équilibrée | −0,33 pt [−1,08 ; +0,41] | +0,00089 [−0,00006 ; +0,00193] |
| IPW anchors + crawlés | dev | −0,04 pt [−0,49 ; +0,44] | −0,00019 [−0,00105 ; +0,00063] |
| | dev équilibré | +0,00 pt [−0,45 ; +0,48] | −0,00031 [−0,00121 ; +0,00053] |
| | validation équilibrée | +0,20 pt [−0,68 ; +1,05] | −0,00042 [−0,00189 ; +0,00109] |

### LR seule, sept variantes (`runs/crawl_seed_bias_20261025_lr/`)

| Variante | Retirés du train | Dev équilibré | Validation | Δ validation (IC 95 %) |
|---|---:|---|---|---|
| Référence | 0 | 59,70 % / 0,66508 | 58,22 % / 0,67381 | — |
| Exclusion des seeds | 14 547 | 59,77 % / 0,66549 | 58,20 % / 0,67360 | −0,02 pt [−0,46 ; +0,44] |
| Exclusion des seeds avant entrée | 13 464 | 59,61 % / 0,66550 | 58,02 % / 0,67357 | −0,20 pt [−0,60 ; +0,22] |
| Exclusion des anchors avant entrée | 62 316 | 59,26 % / 0,66604 | 57,44 % / 0,67514 | **−0,78 pt [−1,52 ; −0,04]** |
| IPW seeds avant entrée | 0 | 59,56 % / 0,66504 | 58,10 % / 0,67361 | −0,12 pt [−0,42 ; +0,16] |
| IPW anchors avant entrée | 0 | 59,50 % / 0,66467 | 57,54 % / 0,67387 | **−0,68 pt [−1,20 ; −0,10]** |
| IPW anchors + crawlés | 0 | 59,54 % / 0,66476 | 57,62 % / 0,67392 | −0,60 pt [−1,26 ; +0,10] |

Lecture :

- **Le modèle exploite peu l'artefact.** Il donne 52,3 % à l'équipe la plus crawlée, là où elle gagne 59,1 % en dev et 53,2 % en validation. Le contrôle du 24 octobre l'indiquait déjà.
- **L'exclusion ne paie pas.** Retirer 12 % du train (seeds) dégrade la log-loss de façon significative en dev, sous-ensemble équilibré compris. Retirer les 53 % de matchs à anchor (tout avril–mai) coûte près d'un point.
- **L'IPW est neutre en accuracy** pour le mélange et LightGBM. Il améliore un peu la log-loss équilibrée, sans que ce soit significatif, et réduit de moitié le penchant (52,35 → 51,11 % en validation).
  - LightGBM pondéré prend 2 688 itérations au lieu de 2 126.
  - Seule, la LR perd 0,6 à 0,7 pt en validation avec l'IPW. En dev, l'écart reste dans le bruit (−0,1 à −0,2 pt).
- **L'écart dev → validation du 24 octobre est surtout dû aux matchs déséquilibrés.** Sur les sous-ensembles équilibrés, le modèle gelé fait 59,72 % en dev et 59,37 % en validation, soit 0,35 pt d'écart au lieu de 1,32. En validation, les 40 % de matchs déséquilibrés plafonnent à 56,7 %.
- **Lecture répétée de la validation.** Elle avait déjà été lue le 24 octobre ; elle est relue ici pour trois variantes du mélange et sept en LR, sans aucune sélection sur elle. Les décisions reposent sur le dev.

## Recommandation

1. **Ne pas exclure les matchs de seeds ni d'anchors** à l'entraînement. Pour le mélange, l'exclusion des seeds dégrade la log-loss du dev de +0,00085 [+0,00029 ; +0,00133], sans gain d'accuracy. L'exclusion des anchors coûte −0,78 pt en validation (LR).
2. **Pondérer par IPW (`ipw_anchor_and_crawled`)** les modèles tabulaires d'arbres et le mélange. Le coût est nul en accuracy (dev −0,04 pt [−0,49 ; +0,44]), la log-loss équilibrée gagne un peu (−0,0003 à −0,0004, non significatif), et le penchant vers la collecte est divisé par deux. C'est une mesure d'hygiène, pas un levier : **le gain attendu en accuracy est nul**. Ne pas l'appliquer à la LR seule, qui perd 0,6 pt en validation.
3. **Rapporter systématiquement**, pour toute évaluation : l'ensemble, `balanced`, `strict_balanced` et le penchant crawlé. Le dev chronologique du train est lui-même contaminé : l'équipe la plus crawlée y gagne 59,1 %. Son chiffre global n'est donc pas une référence sans artefact.
4. **Corriger à la source** (section suivante). Le plus gros artefact, le crawl postérieur au match, ne peut pas être neutralisé proprement après coup. Sans heure d'ingestion, on ne sait pas quelles parties de l'historique étaient disponibles avant le match.

## Variante côté collecte (points 1 et 2 implémentés, voir « Mise en œuvre côté collecte »)

1. **Enregistrer la disponibilité.** Ajouter `matches.ingested_at`, et idéalement `fetched_by_puuid`. Construire ensuite l'historique d'un match uniquement à partir des lignes ingérées avant son début. C'est la seule correction à la source du biais de crawl postérieur. Elle demande une migration v16, la mise à jour de la politique de `scripts/merge_databases.py` (qui fixe `SCHEMA_VERSION`) et le protocole de `docs/database_merge.md`. En attendant, l'heure d'ingestion d'un match de l'ère `crawl_queue` peut être approchée par le plus petit `processed_at` de ses participants crawlés après lui.
2. **Seeds : découverte seulement.** Ne pas stocker comme cibles ni comme historique les parties d'une seed antérieures à son entrée : soit `start_time = created_at` pour les seeds, soit un marquage à l'ingestion. La seed sert alors à découvrir la frontière, et ses parties ultérieures arrivent par la boule de neige ; elles gagnent 49,6 %, sans biais mesurable. Même règle pour `collect_top_leagues` et les imports de pros.
3. **Amorcer hors du sommet du ladder.** Tirer les seeds au hasard dans `league-exp-v4` par palier et division. La population collectée se rapprocherait alors de celle servie, et la sélection sur les victoires récentes deviendrait faible.
4. **Tracer la provenance des collectes legacy.** `players.last_updated` est écrasé à chaque sauvegarde : la date de première sélection est perdue. Une table de provenance (source, date) éviterait d'en être réduit à l'approximation actuelle.

## Limites

- **Seeds de septembre invisibles.** Les anchors de septembre ne sont repérés que par la liste des joueurs visités, sans date d'entrée. Une partie de leur artefact reste dans « plus de crawlés ».
- **`last_updated` approximatif.** Pour les anchors legacy, c'est la dernière sauvegarde : « avant sauvegarde » englobe aussi des parties récupérées lors de collectes antérieures du même joueur.
- **`priority >= 5` ambigu.** `enqueue_riot_id` utilise aussi une priorité 10 par défaut pour les ajouts manuels. Dans la base actuelle, les 150 lignes de priorité 10 sont toutes créées en 3 s le 11 octobre, à raison de 50 par région : ce sont bien des seeds ladder.
- **Mécanisme du crawl postérieur.** Il reste une hypothèse. La validation (n = 1 039) montre même un effet résiduel pour les joueurs crawlés *avant* le match (54,2 %), absent du train (49,1 %).

## Reproduction

Avec `.venv312`, depuis la racine du dépôt :

1. `python scripts/diagnose_crawl_seed_bias.py --db training_data.db --dataset-dir runs/aligned_dataset_20261023_prospective --output-dir runs/crawl_provenance_20261025` (environ 5 minutes sur le HDD, lecture seule).
2. `python docs/research-improvement/crawl_seed_bias_20261025/sept_frontier_check.py crawler_frontier.json runs/aligned_dataset_20261023_prospective runs/crawl_provenance_20261025/crawl_provenance.csv`
3. `python docs/research-improvement/crawl_seed_bias_20261025/neutralize.py <feat_v4m.npz> runs/crawl_provenance_20261025/crawl_provenance.csv runs/crawl_seed_bias_20261025 --variants baseline,drop_seed,ipw_anchor_and_crawled`, puis la même commande avec `--models lr` et le dossier `runs/crawl_seed_bias_20261025_lr` pour les sept variantes.
   - Le script refuse toute table de features dont le SHA-256 diffère de celui gelé dans `runs/comfort_features_20261024/lock.json`. Pour régénérer cette table, suivre la section « Reproduction » de `2026-10-24-comfort-features.md`.
   - Durée : 20 à 30 minutes par variante LightGBM sur 10 threads.
   - Preuves agrégées versionnées, sans PUUID, à côté des scripts : `diagnostic_report.json`, `diagnostic_manifest.json`, `neutralize_results.json`, `neutralize_results_lr.json` et `neutralize_inputs.json`. Les hachages des sources enregistrés correspondent aux fichiers commités. La référence du run principal a été calculée par une version du script qui n'écrivait pas encore `inputs.json` ; sa logique est par ailleurs identique.
   - Le run principal a lu une version antérieure du CSV compagnon, sans les colonnes `crawled_after*`. Cette copie est conservée dans `runs/crawl_seed_bias_20261025/input_crawl_provenance.csv` (SHA-256 `f30f77d0…`). Ses valeurs sont identiques sur toutes les colonnes communes, vérifié ligne à ligne.

## Mise en œuvre côté collecte (25 octobre 2026)

Branche `claude/collection-ingest-time`, depuis `216132d`. `training_data.db` n'a été ni migré ni lu : les tests tournent sur des bases temporaires. **Aucun gain d'accuracy n'est mesuré ici** ; la correction ne vaut que pour les matchs collectés après la migration.

- **Schéma v16.**
  - `matches.ingested_at` : ms UTC du premier insert, NULL pour les lignes antérieures.
  - `matches.fetched_by_puuid` : le participant dont l'historique a renvoyé le match, NULL s'il n'y a pas joué.
  - Les écrivains de `core/database.py` les remplissent, et une nouvelle récupération ne les déplace pas.
  - `scripts/merge_databases.py` passe en v16. Le target gagne. Un match importé garde la provenance de la source si elle est cohérente ; sinon elle devient NULL et la ligne d'origine est archivée (`normalized_metadata`). Détails dans `docs/database_merge.md`.
- **Seeds : découverte seulement.** La règle est décidée à l'ingestion (`MatchIngestor.is_discovery_only`), par **participation** et non selon l'anchor qui a récupéré le match. Un match est écarté si l'un de ses participants est une seed (`crawl_queue.priority >= 5`) entrée après sa création, ou s'il précède l'heure de sélection donnée par l'appelant.
  - Pourquoi la participation : la base contient 23 511 participations de seeds avant entrée, bien plus que les 3 750 matchs au plus que renvoient les listes des 150 seeds avec la taille de lot par défaut (25 ; la valeur du 11 octobre n'est pas enregistrée). L'essentiel est arrivé par d'autres chemins : historiques d'autres joueurs et collectes antérieures.
  - Ces matchs ne sont écrits ni dans `matches`, ni dans `training_dataset`, ni dans `match_timelines`, mais leurs participants alimentent la file. Aucun lecteur (historique, cibles, analytics, fusion) n'a donc besoin de filtre.
- **`collect_top_leagues`** applique la même règle, avec l'heure de lecture du ladder. Ses anchors sont enregistrés comme seeds (priorité 10) puis marqués crawlés, et leurs participants sont mis en file. La fonction ne stocke donc plus de match : elle découvre la frontière.
- **Filtre de disponibilité (optionnel).** `core/availability_history.py` fournit deux fournisseurs d'historique, l'un sur la base, l'autre sur des lignes figées. Ils ne gardent que les lignes ingérées strictement avant le début de la cible ; les lignes à NULL sont exclues. Le filtre est opt-in et n'est pas branché au serving. Le module ne modifie ni `core/pre_match_history.py` ni `core/frozen_history.py`, car leur SHA-256 est lié aux manifestes de modèles et aux datasets figés.

Limites :

- **Dépendance à `crawl_queue`.** Purger les seeds de la file y laisserait revenir leurs matchs d'avant entrée.
- **Anchor déjà en file.** Un anchor de `collect_top_leagues` déjà présent dans `crawl_queue` garde son entrée : seule sa propre liste est écartée.
- **Hors périmètre.** Les imports de pros (`collect_by_puuid_accounts`), `collect_seed_players` et les lots manuels ne sont pas couverts.
- **Re-téléchargement.** Un match écarté qui réapparaît est retéléchargé une fois par processus : le cache n'est qu'en mémoire.
- **Données existantes.** Toutes leurs lignes ont `ingested_at` à NULL : le filtre de disponibilité y laisse les joueurs sans historique.
- **Migration.** Le premier démarrage de l'app sur `training_data.db`, ou la première écriture d'un collecteur, appliquera la v16 (deux `ALTER TABLE ADD COLUMN`).
