# Comparaison prospective DeepSets / baselines tabulaires — protocole fixé avant lecture des données

Rédigé le 23 octobre 2026, avant la lecture des cibles et avant tout entraînement. Le protocole machine est `runs/prospective_20261023/protocol.json` ; ce document en donne la justification.

## Questions

1. L'architecture DeepSets apporte-t-elle quelque chose par rapport à une régression logistique ou à LightGBM entraînés **sur exactement les mêmes tenseurs 14D/18D et le même découpage** ? Le pilote du 11 octobre ne comparait DeepSets qu'à une fréquence constante ; l'arbre historique (encodeur 24 colonnes figé) n'est pas une baseline équitable.
2. Le gain 18D − 14D observé dans le pilote (−0,0048 nats, IC [−0,0083 ; +0,0010], une seule seed, 1 344 matchs) se confirme-t-il avec davantage de données, trois seeds et un test plus grand ?
3. Quel est le niveau réel du signal sur un test prospectif jamais évalué ?

## Données

- Source : `training_data.db`, en lecture seule, via une copie sur SSD (le disque D: est un HDD dont les lignes récentes sont fragmentées). La copie (base + WAL) est prise sans aucun écrivain actif ; le script vérifie que la copie et l'original ont les mêmes comptes, la même dernière création et le même `MAX(rowid)` avant et après la préparation.
- Toutes les lignes CLASSIC (≤ 150 000), les 10 000 plus anciennes servant uniquement d'historique. Cibles : files classées 420/440. Éligibilité commune : l'extraction 18D doit réussir, et le filtre s'applique aux deux dimensions avant le découpage (même règle que l'audit du 11 octobre).
- **Test** : les 10 000 dernières cibles éligibles, toutes créées après la fin du dernier test déjà évalué (pilote, 11 octobre 2026 10:22 UTC). Environ 31 000 matchs postérieurs à cette date ont été collectés les 16–18 octobre et n'ont jamais été évalués.
- **Validation** : les 5 000 cibles terminées juste avant le test. **Train** : le reste, avec les purges par fin de partie habituelles.
- Les matchs des anciens tests (déjà évalués) ne peuvent entrer ni en validation ni en test. Ils peuvent servir à l'entraînement : ils ne constituent plus un test indépendant, et les utiliser pour apprendre ne contamine pas le nouveau test, qui leur est postérieur.

## Modèles

- **DeepSets** : 14D (`train_deepsets_v3.py`) et 18D (`train_deepsets_v4_timeline.py --patience 0`), seeds 42/43/44, cinq époques, batch 64, learning rate 0,0008, PCGrad : exactement les hyperparamètres du pilote, sans réglage. Le checkpoint retenu est celui de plus faible log-loss de validation, puis la température est ajustée sur la validation. Nouveauté : le biais de côté est initialisé au taux Blue mesuré sur le train (et non plus à 52,5 % Blue). L'ensemble par dimension est la moyenne des trois probabilités calibrées, sans aucun ajustement supplémentaire.
- **Régression logistique** (`core/tabular_baselines.py`) : différences Red − Blue par rôle des features numériques, moyennes des deux équipes, macro clash et sac signé de champions (+1 Red, −1 Blue). Les features sont standardisées, et C est choisi sur la validation dans {0,001 … 1}.
- **LightGBM** : mêmes features, paramètres fixes et nombre d'arbres déterminé par early stopping sur la validation.
- Les deux baselines utilisent la seed 42 et reçoivent la même calibration par température sur la validation.
- **Constante** : P(Red) du train.

## Critères

- **Principaux** : différence appariée de log-loss calibrée sur le test, DeepSets 18D (ensemble) − LightGBM 18D et DeepSets 18D (ensemble) − logistique 18D, par bootstrap apparié en blocs chronologiques mobiles (blocs de 100 matchs, 2 000 répétitions, seed 20261023).
- **Règle de décision fixée d'avance** : DeepSets n'est réputé apporter une valeur que si les deux intervalles à 95 % sont entièrement sous zéro. Sinon, la baseline plus simple est préférée.
- **Secondaires** : chaque modèle calibré contre la constante ; 18D − 14D dans chaque famille ; dispersion entre seeds ; AUC, Brier, exactitude et risque sélectif aux seuils fixes existants.

## Écart déclaré au protocole

Un essai de fumée de la chaîne complète (préparation, une époque GPU, baselines) a été lancé sur les 15 000 dernières lignes, soit la période qui contient le futur test. La logistique y a choisi C = 0,001, borne inférieure de la grille initiale. La grille a donc été élargie à {0,0001 ; 0,0003} avant tout ajustement réel, et le module signale désormais un choix en bordure (`boundary_hit`). C'est la seule modification décidée après cet essai. Les métriques du smoke (log-loss de validation d'environ 0,69) ont été vues ; elles ne servent à aucune sélection et le smoke n'est pas un résultat.

## Discipline du test

Tous les checkpoints, calibrations et artefacts de baselines, ainsi que les sources, sont hachés dans `lock.json` avant la première lecture du test. Le test est ensuite évalué une seule fois. Aucun hyperparamètre, seuil ni modèle n'est modifié après sa lecture, et aucun modèle n'est promu. Les intervalles restent exploratoires : des joueurs sont partagés entre blocs, et la variance d'entraînement n'est couverte que par trois seeds.

## Limites connues avant exécution

- Le test devait couvrir environ trois jours (16–18 octobre) ; il en couvre en réalité 21 heures (voir la cohorte observée ci-dessous). Ses matchs sont tous issus de deux campagnes d'enrichissement (`days_back=14`) ; la population de joueurs est celle du crawler, pas un échantillon du ladder.

## Cohorte observée après préparation (sans métrique)

La préparation a lu 146 241 lignes CLASSIC (du 17 juin 2024 au 18 octobre 2026) et retenu 133 600 cibles, avec une couverture 18D de 100 % après filtrage. Le découpage obtenu est le suivant :

- train : 118 600 matchs, du 12 avril au 16 octobre 2026 ;
- validation : 5 000 matchs, du 16 octobre 14:44 au 17 octobre 11:24 UTC ;
- test : 10 000 matchs, du 17 octobre 11:43 au 18 octobre 09:02 UTC.

Le test couvre donc environ 21 heures, et non trois jours : c'est une seule tranche temporelle, ce qui accentue la dépendance entre blocs. 3 844 matchs d'anciens tests figurent dans le train. Les exclusions se répartissent ainsi : 2 051 pour une file non classée, 86 pour des rôles invalides, 81 pour un historique timeline 18D indisponible et 423 purges temporelles. La lecture s'est faite sur une copie SSD de la base, dont l'état logique (comptes, `MAX(rowid)`, dernière création) a été vérifié identique à l'original avant et après.
- La disponibilité des résultats est supposée effective dès la fin de partie ; les temps d'ingestion réels sont inconnus.
- Les profils statiques de champions n'ont pas de provenance historique par patch.
