# Audit de disponibilité 18D — 11 octobre 2026

**Audit terminé. Dans `training_data.db`, 97 894 des 97 916 matchs cibles admissibles passent réellement l'extraction 18D, soit 99,978 %. Les 1 344 matchs ajoutés après la fin du test réservé précédent passent tous.** Ce résultat établit la disponibilité des features ; il ne mesure aucun gain prédictif et ne rend pas automatiquement prêt l'ancien dataset figé.

La collecte est restée arrêtée. Cet audit a uniquement lu la base et écrit des rapports locaux. Aucun entraînement, nouveau bundle de features, calcul de performance sur le test réservé ou changement du modèle servi n'a été effectué.

| Contrôle | Résultat |
|---|---:|
| Matchs dans la base | 118 310 |
| Sources CLASSIC avec date entière positive | 110 049 |
| Timelines acceptées par l'extracteur strict | 109 891 |
| Timelines absentes ou vides | 158 |
| Timelines présentes rejetées | 0 |
| Sources réservées à l'amorçage historique | 10 000 |
| Cibles exclues : file non prise en charge | 2 051 |
| Cibles exclues : rôles non canoniques | 82 |
| Cibles admissibles après sélection | 97 916 |
| Extraction 18D réussie | 97 894 / 97 916 |
| Extraction 18D échouée | 22 / 97 916 |

Les 22 échecs ont tous la même cause : aucun match antérieur d'aucun des dix joueurs dans la fenêtre historique retenue. Ils ne résultent pas d'une timeline manquante sur un historique existant. Aucun candidat de récupération de timeline existante n'a été identifié pour les résoudre. Relancer aveuglément le script de backfill ne corrigerait donc pas ces 22 cas. Leur exclusion éventuelle d'une expérience doit être déclarée et appliquée aux deux variantes 14D et 18D.

Une cible acceptée ne signifie pas que ses dix joueurs ont chacun un historique timeline. Les joueurs sans observation reçoivent les valeurs par défaut documentées du contrat.

| Population | Cibles acceptées en 18D | Places de joueurs avec historique | Matchs avec historique pour les dix joueurs |
|---|---:|---:|---:|
| Ensemble audité | 97 894 / 97 916 (99,978 %) | 850 063 / 979 160 (86,82 %) | 44 007 / 97 916 (44,94 %) |
| 30 derniers jours de la source | 24 964 / 24 965 (99,996 %) | 223 336 / 249 650 (89,46 %) | 12 909 / 24 965 (51,71 %) |
| Après l'ancien test réservé | 1 344 / 1 344 (100 %) | 12 480 / 13 440 (92,86 %) | 741 / 1 344 (55,13 %) |

Les places de joueurs sont comptées par match, et non comme des joueurs uniques. La profondeur reste variable : seuls 8 777 matchs sur l'ensemble audité, et 106 parmi les 1 344 récents, ont au moins cinq historiques pour chacun des dix joueurs.

**L'ancien dataset et la base auditée sont distincts.** Les 49 486 identifiants de `runs/aligned_dataset_20261009_reviewed/model.split.json` sont tous absents des 118 310 identifiants de `training_data.db`. Cette absence a été vérifiée directement dans l'index SQLite, indépendamment du filtre des cibles. L'ancien bundle provient de `runs/db_merge_20260924/consolidated.db` ; sa disponibilité 18D reste celle de son rapport initial, soit 0/49 486. Il faut reconstruire et vérifier explicitement les sources et les features d'une expérience, et non remplacer ce chiffre par la couverture de la présente base.

Le chemin actuel de création d'un **nouveau test prospectif** demande 3 000 cibles postérieures au 24 septembre 2026 à 14:06:27.690 UTC. Les 1 344 disponibles sont toutes admissibles en 18D ; il manque donc au moins 1 656 nouvelles cibles pour ce protocole précis. Les deux tentatives de partitionnement de l'audit ont échoué sur cette condition. Ce n'est pas une exigence intrinsèque du 18D. Conserver l'ancien test réservé pour l'expérience initiale reste une possibilité méthodologique, mais exige une reconstruction et une vérification propres à ses sources et à ses memberships ; cet audit ne les certifie pas.

La prochaine étape est de figer une cohorte commune 14D/18D avec les exclusions explicites, de verrouiller son protocole d'évaluation, puis d'entraîner et calibrer les candidats sur train/validation. La comparaison prédictive finale demeure à faire.

Méthode : lecture SQLite en mode lecture seule, capture par groupes de dates complets et vérification exhaustive des identifiants/dates capturés. Le `data_version` de la connexion observatrice est resté à 2. Le replay utilise `FrozenHistoryProvider` et `extract_aligned_inputs` de production pour chaque cible, avec vérification des formes `(1, 5, 18)` et `(1, 15)`. L'historique respecte la fenêtre globale des 10 000 créations strictement antérieures et exige une fin de partie strictement antérieure au début de la cible. Quarante-neuf contrôles supplémentaires ont rapproché les compteurs avec un replay direct. Les hashes des sources de production sont restés inchangés.

La capture a utilisé Python 3.14 et le parseur JSON installé `pydantic_core`, avec 2 200 comparaisons échantillonnées au parseur standard, sans divergence détectée. Les vérifications complémentaires sous l'environnement du projet ont confirmé 30 captures complètes, 1 500 faits de matchs et 2 500 timelines. Cette validation de parité est échantillonnée, pas exhaustive. Le contrôle des features sur les 97 916 cibles est, lui, exhaustif.

Limites : les résultats historiques sont supposés observables à la fin de partie, car les dates d'ingestion ne sont pas disponibles. Les « 30 derniers jours » se terminent à la création du dernier match source, le 11 octobre 2026 à 10:22:30.047 UTC. Une cohorte filtrée change la population évaluée. La couverture des features ne démontre ni la calibration ni l'utilité prédictive du 18D.

Preuves et reproduction : [résultat principal](../../runs/timeline18_audit_20261011/result.json), [synthèse des cohortes](../../runs/timeline18_audit_20261011/cohort_summary.json), [vérification des anciens identifiants](../../runs/timeline18_audit_20261011/old_membership_source_check.json), [script d'audit](../../runs/timeline18_audit_20261011/audit.py), [script de rapprochement](../../runs/timeline18_audit_20261011/summarize.py). Le rapprochement des enregistrements individuels avec les totaux, les histogrammes et les échecs du résultat principal a réussi. Les empreintes des scripts et résultats sont enregistrées dans la synthèse des cohortes.
