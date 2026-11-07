# Intégration des statistiques de matchup ordonnées

Statut : conception préparée à la demande de l'utilisateur, sans implémentation.
Le prompt de lancement associé autorise l'exécution du plan et de cette conception.

## Objectif et définition d'« intégré »

Remplacer le calcul contaminé des matchup dans les quatre scripts réels
d'entraînement/évaluation. Le nouvel extracteur doit devenir le chemin normal
des représentations concernées, avec provenance vérifiable et tests exécutant
les consommateurs. Ajouter un module inutilisé ou conserver seulement un
benchmark ne satisfait pas cet objectif.

L'intégration porte sur **la préparation des données, l'entraînement et
l'évaluation**. Elle ne signifie pas qu'un nouveau modèle est activé dans
l'application. `PredictionService` ne restaure actuellement aucun état de
matchup : une activation ultérieure exigera un contrat d'inférence et un état
historique compatible. Aucun checkpoint actif, route Flask ou configuration
de sélection de modèle ne doit être remplacé ici.

## Preuves déjà disponibles

- Prototype : `docs/research-improvement/ordered_matchup_features.py`.
- Provenance des fins : `accuracy_matchup_experiment.py::result_end_times` dans
  le même dossier ; 31 500 fins explicites valides sur la référence.
- Tests : `tests/test_ordered_matchup_experiment.py` et
  `tests/test_accuracy_matchup_metadata.py` ; 45 tests combinés réussis en A3.
- Journal : `docs/research-improvement/2026-09-27-accuracy.md`, cycle A3.
- Mesures : `runs/accuracy_matchup_20260927/metrics.json` et `decision.json`.
- Accuracy test HGB : greedy 53,0519 %, ordered 57,5556 %, neutre 57,3481 %.
  Le gain face au contrôle neutre n'est pas établi. Le réseau historique reste
  à 58,8296 %. Ces résultats de mai ne prouvent aucun gain d'un réseau réentraîné.

## Choix de conception

1. **Extracteur partagé et métadonnées séparées — retenu.** Promouvoir le helper
   validé dans `core/features/ordered_matchup.py`. Lire les fins dans la couche
   données, conserver des fichiers JSON associés aux nouveaux runs et brancher
   les quatre consommateurs. Préserve les caches pickle et schémas existants.
2. Ajouter des dates dans tous les records/caches : écarté pour ce changement,
   car cela impose une évolution de format sans résoudre la provenance des
   caches déjà produits.
3. Neutraliser les matchup par défaut : écarté comme substitution silencieuse.
   Le contrôle neutre reste une ablation de recherche, pas le correctif demandé.

### Temporalité

À la création d'une cible, le tracker reçoit seulement les résultats du train
dont la fin est **strictement antérieure**. La fin doit être strictement après
la création de son propre match. Original et swap sont extraits au même état.
Validation et test n'alimentent jamais ce tracker. L'ordre des sorties reste
celui des entrées ; un nouvel appel repart d'un état local vide.

Les timestamps sont des entiers Unix en millisecondes. Utiliser uniquement
`info.gameStartTimestamp` et `info.gameEndTimestamp` explicites, avec contrôle
de cohérence comme dans le lecteur expérimental. Ne pas estimer une fin par
création + durée. La fin reste un proxy optimiste : l'ingestion API est inconnue.
Les priors joueurs attachés ne sont pas reconstruits dans ce chantier.

### Données manquantes et caches anciens

Les consommateurs acceptent `--result-end-times PATH`. Sinon, ils utilisent
leur base explicitement fournie ; v2 reçoit donc aussi `--db-path`.
L'évaluateur et le rejeu enrichi du script général acceptent `--splits-cache`
pour utiliser exactement les priors déjà attachés, sans reconstruction implicite.
`--missing-result-end {error,exclude}` vaut `error` par défaut. Le mode `exclude`
est une dégradation explicite, journalisée et enregistrée, qui exclut seulement
les événements sans fin valide. Si aucun résultat n'est utilisable, arrêter
même sous `exclude` pour éviter un entraînement neutre involontaire.

Le helper bas niveau conserve son comportement testé d'exclusion ; le résolveur
partagé applique la politique stricte avant tout entraînement. Les anciens
caches sont lus sans réécriture. Une source absente, un cache/provenance
incompatible ou un désaccord base/cache provoquent une erreur exploitable.

### Contrats et compatibilité

Les nouveaux runs écrivent `model.features.json` et
`model.result_end_times.json` ; aucun ancien artefact n'est modifié. Le contrat
identifie `ordered_matchup_v1`, représentation, dimensions, poids du tracker,
hash des données enrichies, hash des splits, hash des fins, exclusions et
hypothèse de disponibilité. Le hash historique de `core.experiment` reste
inchangé ; il ne couvre pas les priors ni les dates de fin. Ajouter un hash
distinct des records enrichis complets et le SHA du pickle s'il est utilisé.

Les nouveaux manifestes de v2/v3 utilisent `build_manifest` ; leurs manifestes
actuels écrits à la main sont incomplets pour `read_manifest`. Ne pas réparer
les anciens fichiers en place ni affaiblir le validateur pour les accepter.

L'évaluation reconnaît les entrées neuronales par nom, y compris `macro_clash`.
Le chemin historique à 8 dimensions garde son extracteur et ses probabilités.
Le chemin à 14 dimensions exige un contrat compatible ; sans celui-ci, arrêter
avec un message de réentraînement requis. Aucun mode greedy de compatibilité
en production n'est ajouté. Le poids NN vient du contrat (5 pour v2/v3, 15 pour
la branche correspondante du script général), celui des arbres reste 15.
La calibration d'un nouveau checkpoint est liée au hash de son contrat.

## Contraintes globales

- Préserver les modifications déjà présentes dans le workspace.
- Aucune nouvelle dépendance, migration SQLite ou modification des anciens caches.
- SQLite en lecture seule ; ne pas utiliser une connexion qui active WAL/commit pour cette lecture.
- Préserver les dimensions 8, 14, 232 et 15 des représentations concernées.
- Préserver l'anti-symétrie des probabilités et l'invariance aux permutations de joueurs.
- Ne jamais recalculer les swaps d'anciennes lignes avec le tracker final.
- Aucun label validation/test dans le tracker, aucune sélection sur les scores test.
- Aucun écrasement de checkpoint, rapport, manifeste ou calibration existant.
- Ne pas revendiquer un gain du modèle actif sur la base du benchmark tabulaire.
- Aucun changement à PredictionService, aux routes HTTP ou au modèle activé dans ce plan.

## Validation et limites du chantier

Les tests causaux sont repris sur le code de `core`, puis complétés par des
tests d'intégration des quatre consommateurs, des métadonnées et de la
compatibilité. Des entraînements réels courts à une époque vérifient les
chemins neuronaux ; ils ne mesurent pas leur qualité prédictive.

Rejouer une comparaison tabulaire figée via le code intégré, dans un nouveau
run, sans réglage sur le test de mai. Comparer les matrices et prédictions à
A3 sous les versions de bibliothèques identiques. Une différence doit être
expliquée, pas masquée par une modification des scores attendus.

Deux réparations adjacentes sont explicitement dans le périmètre si elles
bloquent les tests : `raw_val_probs` non défini dans v2 ; mauvais contrat de
`split_records` dans v3. Le chargement général des vieux checkpoints, la
causalité des priors joueurs, le suivi de patch et l'inférence servie restent
des chantiers distincts. Ne pas lancer une nouvelle boucle bibliographique.
