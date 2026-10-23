# Dataset figé et parité entraînement/service NNriot

Date : 2026-10-09. Statut : proposition issue de la discussion, à relire avant exécution.

## Objectif et périmètre

Produire un dataset reproductible pour de nouveaux entraînements et démontrer que ses tenseurs 14D/18D correspondent exactement aux features produites par le service au même instant. Cette étape livre le replay, les partitions, les artefacts, les adaptateurs des deux trainers et leur vérification. Elle prépare une expérience ; elle ne démontre aucun gain prédictif.

La campagne longue, les seeds multiples, une nouvelle baseline tabulaire, l'ouverture du test final, la sélection du modèle et sa promotion feront l'objet d'un plan d'expérience distinct. La collecte de nouvelles timelines constitue également un chantier distinct si l'audit de disponibilité l'exige.

## État constaté dans le code et les données

- Le service et la comparaison actuelle partagent `core/pre_match_history.py` : résultats terminés strictement avant la création du match cible, fenêtre globale de 10 000 matchs CLASSIC précédents, puis filtrage des joueurs pertinents.
- v3 possède déjà un replay ordonné des confrontations de lane. `core/features/ordered_matchup.py` précise que ce replay ne reconstruit ni ne certifie les priors joueurs. Ce parcours utilise également un tracker de matchups appris, alors que le service actuel exige des matchups neutres.
- v4 charge les priors via `load_enriched_records`, qui actualise le tracker dans l'ordre de création. Ses tenseurs sont ensuite extraits directement par `DraftFeaturePipeline`.
- Les deux trainers évaluent automatiquement leur partition test à la fin. Leur nouveau parcours aligné doit réserver cette évaluation pour la future expérience.
- La comparaison déjà réalisée sur 2 000 matchs est exploratoire. Ses matchs et sa période servent à définir les exclusions du futur test ; ce cohort ne constitue plus un test final à découvrir.

Inventaire en lecture seule effectué le 9 octobre 2026 :

| Source | Matchs, tous modes | Dernière création, UTC ms | Lignes de timelines |
| --- | ---: | ---: | ---: |
| `training_data.db` | 116 966 | 1789645132620 | 116 966 |
| `runs/training_data_23h_20260922_191434.db` | 83 938 | 1790258787690 | 0 |
| `runs/db_merge_20260924/consolidated.db` | 188 688 | 1790258787690 | 0 |

Ces nombres décrivent des lignes stockées, pas leur validité pour l'entraînement. L'absence de timelines récentes interdit de promettre un dataset commun 14D/18D prêt aujourd'hui. La source consolidée est une candidate pour 14D ; sa sélection définitive dépend de l'audit. Aucun assemblage implicite avec les timelines d'une autre base : une telle source devrait aussi être disponible au service et ferait évoluer son contrat d'acquisition.

## Contraintes globales

- Répertoire de travail : `D:\projet\NNriot` ; commandes Python : `.\.venv312\Scripts\python.exe` ; Python 3.12.
- Aucune nouvelle dépendance ; utiliser SQLite, NumPy, TensorFlow/Keras et les modules déjà installés.
- Bases opérationnelles en lecture seule : URI `mode=ro`, `PRAGMA query_only=ON` ; aucune migration, fusion, collecte ou mise à jour `is_trained`.
- Préserver tous les changements locaux préexistants et `archive/frontend-2026-10-09/` ; aucun reset, nettoyage global ou commit de changements étrangers à la tâche.
- Dimensions exactes : contrats `nnriot.players14.v1` et `nnriot.players18.v1` ; entrées float32 `(N, 5, 14)` ou `(N, 5, 18)`, macro float32 `(N, 15)` ; aucun padding ni troncature.
- Rôles canoniques : `TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY` ; chaque équipe a cinq joueurs ; chaque identité connue apparaît une seule fois.
- Cutoff de features : `game_creation_time` en millisecondes UTC ; utiliser uniquement les résultats avec `result_end_ms < game_creation_time` ; timestamps d'acquisition inconnus, hypothèse explicitement déclarée.
- Historique : fenêtre globale de 10 000 créations CLASSIC strictement antérieures ; les labels ML historiques ne conditionnent pas les priors ; matchups neutres au service et dans le nouveau parcours d'entraînement.
- Population cible : queues Riot `420` et `440`, mode `CLASSIC`, rôles et labels valides ; l'historique conserve tous les CLASSIC, comme le service.
- Partitions communes définies avant les métriques : 3 000 cibles récentes pour test, 3 000 précédentes pour validation, créations ex æquo regroupées ; purger train avec fin >= début validation et validation avec fin >= début test.
- Le test réservé commence strictement après la dernière création du cohort exploratoire antérieur ; exclure ses identifiants test des cibles du nouveau dataset et enregistrer cette règle.
- Chaque fichier figé possède un SHA-256 ; le replay exige les mêmes hashes de sources de features ; toute incompatibilité échoue explicitement.
- Le parcours aligné charge et entraîne uniquement train/validation ; aucune lecture des labels test ni métrique test par les trainers ; aucun chargement de poids anciens.
- 18D exige une source de timelines historiques utilisable dans la fenêtre réelle ; absence de source = `FeatureUnavailableError`, avec causes et couverture, jamais une disponibilité inventée.
- Préserver `draft_advantage` antisymétrique et miroir 50/50 ; conserver `DirectionalSideBias` séparé dans `winner` ; aucun changement d'architecture dans ce chantier.
- Aucun changement automatique de checkpoint configuré, de calibration de production ou de statut `validated_for_serving`.

## Comportement attendu

### Replay partagé

Un provider figé sélectionne exactement la même fenêtre de créations que `DatabaseHistoryProvider`. Il utilise `decode_history_row`, `build_target_history`, `ResultOrderedHistory.enrich` et `FeatureContract.extract`. Les données du match cible sont assainies avec `to_serving_features`. Les priors joueurs et les tenseurs doivent être identiques bit à bit au chemin SQLite, y compris sans index `match_participants` complet. La canonisation des équipes et la neutralité des matchups restent celles du contrat actuel.

Les cibles n'utilisent aucun de leurs champs de combat ou labels pour créer leurs propres features. Un résultat de validation ou d'un test précédent peut entrer dans l'historique d'une cible ultérieure seulement après sa fin : c'est le replay séquentiel du service. Ces observations ne modifient jamais les poids du modèle, la calibration ou le choix des hyperparamètres.

### Snapshot, population et partitions

Geler une unique lecture SQLite cohérente. Conserver les faits historiques compactés, le `queueId`, les labels multitâches décodés et les causes d'exclusion. Réserver les 10 000 premières lignes du snapshot au contexte historique ; elles ne sont pas des cibles. Les doublons d'identifiant et les contradictions entre vainqueur Riot et label ML sont des erreurs/exclusions explicites.

Les fichiers `train.json`, `validation.json` et `sealed_test.json` conservent leurs identifiants, créations, fins et labels. Les fichiers de tenseurs sont distincts par partition et contrat. Un dataset peut être `ready_14d` avec `blocked_18d`, sans changer les partitions pour dissimuler les absences. Un trainer 18D refuse ce dataset. Lorsque 18D est disponible, les identifiants et leur ordre sont exactement ceux de 14D.

La réservation du test est une discipline d'expérience, pas une barrière de sécurité : les faits Riot archivés contiennent les résultats. Le code d'entraînement ne doit ni décoder `sealed_test.json` ni extraire de métriques test. Les limitations concernant les expériences antérieures et la disponibilité rétrospective restent visibles dans le manifeste.

### Traçabilité et reproductibilité

Le manifeste lie le snapshot de faits, les règles de population, les exclusions antérieures, les IDs et bornes de partitions, les hashes du code de features, les tenseurs et la disponibilité par contrat. La vérification n'interprète jamais une simple dimension ou une date de fichier comme preuve de provenance.

Une préparation CLI explicite produit un répertoire neuf. Une commande de vérification rejoue les mêmes faits sans relire la base et compare les tenseurs avec `np.array_equal`. Une vérification SQLite sur un petit panel figé démontre également la parité de priors et de tenseurs avec le provider en service.

### Branchement des trainers

Ajouter `dataset_dir` et `--dataset-dir` aux trainers v3/v4. Dans ce mode, rejeter les anciens caches/manifests/artefacts de fins fournis simultanément, charger seulement train/validation, conserver les boucles d'optimisation et les têtes existantes, fitter la température sur validation avec l'API stricte de calibration, et écrire un lien de provenance vers le dataset. Les chemins historiques restent utilisables pour leurs usages exploratoires ; ils ne deviennent pas certifiés par cette modification.

## Conditions de réception

1. Les tests de parties qui se chevauchent, d'égalité de timestamps, d'index incomplet, de permutations/swap, de labels corrompus, d'absence de timelines et d'artefacts modifiés passent.
2. Un snapshot réel 14D, ses partitions, ses exclusions et ses hashes sont livrés et rejouables ; une insuffisance de cibles après la période déjà explorée est signalée sans réduire silencieusement les minima.
3. Un rapport 18D donne une couverture et des causes réelles. Avec les bases récentes inventoriées, `blocked_18d` est le résultat attendu jusqu'à l'acquisition adéquate des données.
4. Les adaptateurs entraînent uniquement sur les tenseurs du dataset et leur smoke test ne consulte pas le test réservé.
5. Les contrats/manifests du service sont remis en cohérence avec les hashes de code modifiés, en conservant le statut de validation existant. Les anciens résultats exploratoires restent archivés avec leur protocole original.
6. La documentation permet de préparer, vérifier et utiliser le dataset avec des commandes exactes, et distingue la prochaine expérience des garanties de cette étape.
