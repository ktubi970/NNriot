# Reprise de la recherche sélective — 11 octobre 2026

## Cadre, historique et état initial

Reprise du [cycle sélectif 1 clos le 27 septembre](2026-09-27-selective-boundaries.md),
sans renumérotation ni remise à zéro de ses sources. L'étude accuracy A1–A3 du
[journal distinct](2026-09-27-accuracy.md) reste inchangée.
Budget de cette invocation : trois cycles au maximum, au plus une amélioration
par cycle. Plan initial : S2, puis S3 de consolidation. À la clôture S2, la règle
d'arrêt prévaut : S1 historique + S2 satisfont le minimum cumulatif, et l'objectif
est atteint. S3 n'est donc pas ouvert. Pas d'entraînement de candidat pour cette
recherche, promotion de modèle ni tâche planifiée ; les tests exécutent leurs
entraînements courts isolés habituels. Stagnation initiale : 0.

Racine conservée : classification sélective pré-match par conviction `abs(P-.5)`.
Objectif observable : la recherche automatique trouve la couverture empirique
maximale admissible parmi les masques réalisables avec la règle de service,
sans manquer une cible à cause de l'échantillonnage des seuils.
Ce n'est pas un objectif de gain d'accuracy du réseau ni de garantie prospective.

HEAD : `3bb93c7ccb4e5e201f953d2686b468ff1413b825`.
Nombreux changements préexistants conservés ; les deux fichiers visés étaient
propres à l'ouverture. Leur copie initiale et les empreintes des fichiers suivis
déjà modifiés sont dans [les preuves](2026-10-11-threshold-evidence/).
Pas de nouvelle dépendance, changement d'API, dimensions, SQLite ou checkpoint.
Les têtes draft/side-aware ne sont pas concernées ; la symétrie testée ici est
la transformation des probabilités `p -> 1-p`, pas celle du winner avec biais de côté.

Graphify : son exécutable échoue (`Failed to canonicalize script path`). Lecture
du graphe puis parcours NetworkX avec son interpréteur existant : vocabulaire
pertinent `calibration`, `threshold`, voisinage de `calibration_error()`
(`analytics/interpretable.py:L57`) vers `evaluate_probabilities():L73`.
Le graphe ne couvre pas les modules sélectifs actuels ; code et usages vérifiés
directement par `rg`. Aucun appel de `find_threshold_for_accuracy` hors tests
et documentation dans `core/` et `scripts/`.

Premier pytest interrompu avant collecte : le TEMP du bac à sable empêche
l'ouverture SQLite. Reprise avec TEMP et TMP dans
`D:/projet/NNriot/scratch/research-threshold-temp`, sans toucher aux données réelles.

## Cycle S2 — recherche et plan avant édition

État : recherche effectuée, référence de tests en cours, code non modifié.

Frontière : distance 1, de la sélection par conviction à l'optimisation empirique
du compromis risque/couverture. Lien : mêmes ensembles sélectionnés ordonnés par
score, formulation sous contrainte de risque (Franc et al.). Recherche de contrôle
distance 0 : définition des masques et méthode SGR (Geifman et El-Yaniv).

Besoin local mesuré : la grille de 100 quantiles et 101 seuils uniformes ne couvre
pas tous les masques. Sur 1 000 scores serrés, elle retourne `target_met=False`,
accuracy .9, count 990, pour cible .9007 ; le seuil .100003 retient 998 observations,
accuracy .9008016032064128. Résultats dans `counterexample-before.json`.

Question ciblée : peut-on visiter tous les ensembles réalisables avec la même
quantification six décimales et les égalités du service, à coût O(n log n),
sans supposer que l'accuracy empirique est monotone ? Hypothèse réfutable par
énumération exhaustive indépendante et exemples adverses.

Recherche Firecrawl réellement exécutée :

- `selective classification optimal bounded selective risk coverage empirical threshold sorted confidence scores Geifman Franc` (4 résumés).
- Lecture ciblée `arxiv:2101.12523` : introduction, contributions, §2.3,
  problèmes 4–5, théorèmes 4–6 (version indexée v1 de 2021).
- Lecture ciblée `arxiv:1705.08500` : algorithme 1, lemme 3.1, théorème 3.2, §4.
- Lecture ciblée `arxiv:2407.01032` : introduction, figure 2 et §3 (version v2).

Sources dédoublonnées et limites :

1. Franc, Prusa, Voracek, *Optimal Strategies for Reject Option Classifiers*,
   [prépublication 2021](https://arxiv.org/abs/2101.12523), publiée JMLR 24(11), 2023.
   Résultat des auteurs : caractérisation optimale avec risque conditionnel connu,
   randomisation possible aux égalités. Transfert local : maximiser la couverture
   sur notre famille **déterministe**, avec les égalités groupées ; pas de
   prétention d'optimalité de la conviction NNriot ou de connaissance du risque vrai.
2. Geifman, El-Yaniv, *Selective Classification for Deep Neural Networks*,
   [NeurIPS 2017](https://arxiv.org/abs/1705.08500).
   Résultat des auteurs : tri des scores et certification SGR par borne binomiale
   et correction des comparaisons sous iid. Notre correction est une recherche
   empirique exhaustive ; elle n'implémente ni SGR ni sa garantie.
3. Traub, Bungert, Lüth, Baumgartner, Maier-Hein, Maier-Hein, Jaeger (2024),
   *Overcoming Common Flaws in the Evaluation of Selective Classification Systems*,
   [arXiv:2407.01032](https://arxiv.org/abs/2407.01032).
   Sections lues distinguent un seuil opérationnel des métriques agrégées et
   illustrent les limites d'AURC. Ajouter AUGRC ne corrigerait pas le défaut local ;
   candidat écarté sans implémentation. Les résumés de 2512.12844, 2410.15361 et
   2010.07853 ont été repérés, sans lecture du texte ni conclusion transférée.

Sélection : une seule correction, recherche exacte des masques déterministes.
SGR, nouveau score, nouvel indicateur et seuil de production : différés/écartés
car ils ne sont pas nécessaires à cet objectif et requièrent une autre validation.

Plan :

1. Écrire des tests de régression pour cible manquée, couverture maximale,
   scores arrondis ex aequo, accuracy non monotone, contraintes, fallback et
   inversion des probabilités ; observer les échecs pertinents avant patch.
2. Modifier uniquement `core/selective_classification.py` : quantifier une fois
   avec le même arrondi que le service, trier et compter les suffixes par groupe.
   Choisir la première couverture admissible atteignant la cible ; sinon conserver
   le fallback meilleure accuracy, départagé par couverture. Évaluer le seuil
   choisi avec l'évaluateur existant pour conserver le format et Wilson.
   Le chemin `candidate_thresholds` explicite garde son contrat.
3. Tests existants sélectifs/calibration, oracle exhaustif, comparaison temporelle
   sur données synthétiques fixées (aucun réajustement sur cohorte test réelle),
   suite Python et revue indépendante. Corriger les affirmations d'optimalité
   de la docstring de cette fonction, pas les résultats numériques de Wilson.
4. Critères : tous les contre-exemples corrigés, équivalence des métriques au
   masque réellement servi, aucune régression imputable ; mémoire linéaire et
   mesure du coût sur 45 000 observations. Préserver les modifications initiales.
5. Retour arrière si non concluant : restaurer seulement ces deux fichiers
   depuis leurs copies `.before` (s'ils n'ont pas changé entre-temps), garder
   les preuves et la raison. Pas de restauration globale du dépôt.

Prochaine frontière candidate : optimisation empirique → validité après sélection
de seuil, via comparaisons multiples. Le journal historique établit déjà un lien
direct entre racine et certification (Geifman, algorithme 1) : la distance minimale
reste **1**, même si la découverte par ce plan serait de longueur 2.

### Vérifications S2 avant clôture

- Référence existante : **28 passed in 1.03s**, `targeted-before.txt`.
- Régressions avant patch : **8 failed, 38 passed**, `red.txt` ; quatre échecs
  fonctionnels de cible/couverture, deux changements de représentant canonique
  et deux défauts de validation des tailles. Les tests ne revendiquent pas huit
  défauts algorithmiques indépendants.
- Après patch, les quatre fichiers sélectifs/calibration : **107 passed in 75.77s**,
  code retour 0, `focused-after.txt`. Régressions des tailles : erreur cohérente
  `ValueError` avant le calcul, y compris lorsque un seul tableau est vide.
- Revue indépendante : aucun défaut actionnable. 160 jeux comparés à l'oracle
  exhaustif ; candidats explicites comparés à HEAD ; arrondi inchangé sur
  55 455 scores × 7 seuils. Contrôle sur probabilités `.5` exact séparé du swap
  car la convention de prédiction à l'égalité est une limite préexistante.
- Benchmark reproductible `benchmark.py` / `benchmark.json` : 45 000 observations
  synthétiques, seed 20261011, trois mesures en ordre alterné. Avant :
  11.386 / 9.086 / 8.935 s ; après : .1511 / .1637 / .0862 s.
  Médianes 9.0858 s et .1511 s (~60× sur cette machine et cette charge).
  Le benchmark tournait avec la validation ciblée ; ces mesures ne sont ni
  un gain de latence du service, ni une prédiction du coût sur toute machine.
- Le contre-exemple donne désormais `target_met=True`, threshold .100003,
  count 998 et accuracy .9008016032064128, avec le même masque en évaluation.
- `git diff --check` sur les deux fichiers : succès (avertissements LF/CRLF).
- Contrôle SHA256 des 54 fichiers suivis préalablement modifiés : 53 identiques,
  `frontend/index.html` a changé pendant la session hors de nos éditions.
  Aucun outil d'édition de cette invocation ne l'a ciblé ; état courant conservé,
  sans restauration d'une modification concurrente. Détails dans
  `preservation-check.json`. Les fichiers préexistants non suivis n'ont pas été
  édités ; pas de prétention d'audit SHA exhaustif de leurs artefacts.

Limites d'exécution : les deux premières suites larges lancées dans le bac à sable
ont été interrompues après absence de progression visible ; leur résultat est
inconclusif, pas « passé ». Un diagnostic borné de l'import TensorFlow s'est
arrêté à 30 s dans la lecture des modules. Les mêmes tests ciblés exécutés avec
autorisation hors bac à sable ont terminé. La suite complète après patch est
en cours dans ce même environnement, journal `full-after.txt`.

État : acceptation finale S2 en attente de cette suite. Pas de nouveau cycle
commencé tant que la validation n'est pas close.

Commandes reproductibles (PowerShell, depuis la racine) :

```powershell
$env:TEMP = 'D:/projet/NNriot/scratch/research-threshold-temp'
$env:TMP = $env:TEMP
# Créer ce dossier si nécessaire ; les fixtures confinent DB et modèles de test.
.venv312/Scripts/python.exe -m pytest tests/test_selective_classification.py -q --tb=short
.venv312/Scripts/python.exe -m pytest tests/test_selective_classification.py tests/test_selective_infer.py tests/test_training_metrics_calibration.py tests/test_model_calibration.py -q --tb=short
.venv312/Scripts/python.exe docs/research-improvement/2026-10-11-threshold-evidence/benchmark.py
.venv312/Scripts/python.exe -m pytest -q --tb=short
git diff --check -- core/selective_classification.py tests/test_selective_classification.py
```

Les copies `.before` permettent de rejouer le benchmark de référence sans
remplacer les fichiers de travail. Les métriques de ce benchmark sur labels
aléatoires ne sont pas des métriques de performance du modèle LoL.

### Attribution des échecs de la suite globale

Rejeu ciblé pendant la suite globale, via `check_baseline.py` (charge le snapshot
du module original en mémoire sans remplacer les fichiers) :

```powershell
.venv312/Scripts/python.exe docs/research-improvement/2026-10-11-threshold-evidence/check_baseline.py tests/analytics/test_extract.py tests/test_merge_databases.py -q --tb=short
```

Résultat : **32 failed, 12 passed in 50.23s**, code retour 1,
`baseline-data-tests.txt`. Sept cas `test_snapshot_matches_original_aggregate_exactly`
attendent le schéma 12 alors que la fixture est au schéma 14 ; 24 cas de fusion
échouent directement ou indirectement sur `only schema version 12 is supported` ;
un cas `test_alias_target_wins_queue_conflicts_and_full_archive_manifest` insère
cinq valeurs dans une table `crawl_queue` à neuf colonnes. Les identifiants complets
sont conservés dans ce journal de test. Ces échecs sont reproduits sans le correctif.
Comparaison avec la liste finale de la suite globale en attente.

### Résultat complet et décision finale — S2 clos, accepté

Suite Python complète effectivement terminée : **1009 passed, 36 failed,
2043 warnings in 842.60s**, code retour **1**, `full-after.txt`.
Ne pas présenter ce dépôt comme entièrement vert.

Les quatre échecs en plus des 32 précédents ont été rejoués avec le module
original (`baseline-other-tests.txt`, **4 failed in 40.32s**) :

- `tests/test_calibrated_serving.py::test_endpoint_feature_parity_and_calibration` :
  HTTP 503 au lieu de 200.
- `tests/test_database_v11_migration.py::test_schema_version_is_at_least_12` :
  constante 14 alors que le test exige exactement 12.
- `tests/test_model_calibration_guard.py::test_final_predict_rejects_uncalibrated_model` :
  checkpoint de fixture absent au lieu de l'erreur de calibration attendue.
- `tests/test_pre_match_training.py::test_equivariant_deepsets_training_run` :
  diagnostic `anti_symmetry_guarantee=False`, contre True attendu. Aucun
  correctif des têtes de modèle ou du biais de côté dans cette invocation.

Comparaison automatisée des identifiants dans `failure-attribution.json` :
**36 échecs globaux = 36 échecs avec le module original**, aucune différence.
La référence restaurée uniquement en mémoire isole l'effet du module changé ;
elle ne prétend pas reconstituer un ancien dépôt entier sans les travaux en cours.
Le rejeu des quatre tests :

```powershell
.venv312/Scripts/python.exe docs/research-improvement/2026-10-11-threshold-evidence/check_baseline.py tests/test_calibrated_serving.py::test_endpoint_feature_parity_and_calibration tests/test_database_v11_migration.py::test_schema_version_is_at_least_12 tests/test_model_calibration_guard.py::test_final_predict_rejects_uncalibrated_model tests/test_pre_match_training.py::test_equivariant_deepsets_training_run -q --tb=short
```

Décision **acceptée** : les contre-exemples sont corrigés, les 107 tests ciblés
passent, les invariants numériques sont contrôlés, et aucun échec de la suite
globale n'est imputable au changement dans ce comparatif. La suite reste rouge
sur les 36 cas préexistants. Aucun essai de production rejeté à restaurer.

Bilan cumulatif : seul le module de sélection et ses tests changent ; format de
retour, candidats explicites, prédicat servi, probabilités, dimensions, SQLite,
architecture et checkpoints préexistants sont conservés. Les longueurs incohérentes
sont maintenant refusées systématiquement, détail explicite du contrat.
Les résultats portent sur l'exactitude de la recherche empirique et son coût ;
aucun gain d'accuracy LoL, de calibration ou de risque futur n'est établi.

**Arrêt de session : objectif mesurable atteint**, minimum de deux cycles clos
respecté par reprise S1 → S2. Un seul nouveau cycle exécuté dans cette invocation,
deux places non utilisées sur trois, stagnation **0**. Distance maximale atteinte
**1**. S3 initialement envisagé n'a été ni recherché, ni exécuté, ni compté.
Les deux cycles de session annoncés dans le plan initial ne sont donc pas
présentés comme réalisés. La littérature n'est pas épuisée.

Point de reprise : la prochaine demande peut traiter la certification après
sélection (sources déjà identifiées dans S1 : *Learn then Test*, arXiv:2110.01052,
DOI 10.1214/24-AOAS1998), avec une cohorte chronologique réellement indépendante
et hypothèses de dépendance explicites. Le lien direct SGR → contrôle statistique
est déjà documenté, donc distance minimale 1. Les échecs de contrat SQLite et
de fixtures de service constituent aussi un chantier distinct mesuré ; aucun
changement de schéma ne doit être improvisé depuis cette étude.
