# Ordered matchup integration — validation du 27 septembre 2026

Les quatre consommateurs sont effectivement branchés sur le code partagé :

| Consommateur | Représentation | Poids matchup | Vérification réelle |
|---|---|---:|---|
| scripts/train_deepsets_v2.py | tensors, 14 dimensions | 5 | 140 matchs, 1 époque, PCGrad, sauvegarde/rechargement, calibration |
| scripts/train_deepsets_v3.py | v3, 14 + 15 dimensions | 5 | 140 matchs, 1 époque, PCGrad, sauvegarde/rechargement, calibration |
| scripts/train_pre_match_model.py | tensors, 14 dimensions | 15 | 140 matchs, 1 époque, PCGrad, sauvegarde/rechargement, calibration |
| scripts/evaluate_ensemble.py | NN selon contrat + tabulaire 232 | NN sauvegardé / arbres 15 | vrai checkpoint v2, vrais HGB et LightGBM, rapport relu |

Le noyau est core/features/ordered_matchup.py ; la lecture RO est dans core/point_in_time_dataset.py ; les contrats/provenances sont dans core/experiment.py ; la préparation commune des entraînements est core/ordered_training.py.

## Preuves et commandes

- `task1_causal.txt` : commande Task1, 56 passed, exit 0 (avant ajout du test gzip corrompu ; ce test supplémentaire passe dans reader_review_green.txt et la suite complète).
- `contracts_final.txt` : provenance, contrats, experiment et fins, 53 passed, exit 0.
- `architecture.txt` : DeepSets, modèle équivariant, pipeline et manifestes, 28 passed, exit 0.
- `training_validation.txt` : commandes réellement exécutées et codes des trois entraînements durables. Le test des batches réels a passé ; les matrices reçues respectent les poids 5/5/15, les probabilités après rechargement sont finies et les hashes de calibration correspondent.
- `neural_smoke_summary.json` : contrats, tailles 98/21/21, températures et historique d'une époque. Bundles : `smoke_v2_1415/v2`, `smoke_v2_1415/v3`, `smoke_v2_1415/general`; cache et base synthétiques dans `smoke_v2_1415/data`.
- `ensemble_contracts_final.txt` : suite ensemble + contrats + provenance, 48 passed, exit 0.
- `ensemble_v2_1500/ensemble_comparison.json`, `ensemble_v2_1500.log`, `ensemble_evidence.json` : vraie exécution de l'ensemble, code 0, source inchangée ; relance refusée avec code 1 dans `ensemble_v2_1500_rerun_refused.log`.
- `full_suite.txt` : `.venv312/Scripts/python.exe -u -m pytest -q --tb=short`, **851 passed, 1974 warnings, exit 0**, 543.89 s. Cette exécution a commencé avant la dernière correction locale de repli DB dans l'évaluateur.
- `ensemble_post_review.txt` : revalidation de tout l'évaluateur après cette dernière correction, **12 passed, 62 warnings, exit 0**, incluant le vrai quatrième consommateur. Les nouveaux cas DB identique / prior modifié / cache explicite absent sont couverts. Aucune seconde suite globale n'a été prétendue.
- `protected-verification.json` : 22 hashes protégés inchangés, dont anciens caches/checkpoints/rapports et fichiers préexistants hors chantier ; `diff_check.exitcode` vaut 0.
- `independent-review.md` : revue indépendante et relectures ciblées des corrections.

## Benchmark final

Commande réellement exécutée :

```powershell
& '.venv312/Scripts/python.exe' scripts/benchmark_ordered_matchups.py --splits-cache runs/candidate_deepsets_v3/splits_cache_45000_v3.pkl --manifest runs/candidate_selective_prod/model.split.json --database runs/db_merge_20260924/consolidated.db --output runs/ordered_matchup_integration_validation_final
```

Code retour 0. Rapport complet, modèles expérimentaux et prédictions : `../ordered_matchup_integration_validation_final/`. Journal `benchmark_final_45000.txt`, synthèse `benchmark_final_summary.json`. Le premier rejeu `../ordered_matchup_integration_validation/` est préservé ; il a été répété parce que le schéma des diagnostics a été durci après le démarrage de son processus.

31 500 fins explicites valides, aucune exclusion ; 45 000 matchs au total. Versions identiques à A3 : NumPy 2.5.2, scikit-learn 1.9.0. Les 9 empreintes des matrices originales sont identiques à A3 ; les empreintes des swaps sont également enregistrées. Sur les trois variantes et trois splits : aucune différence de classification, différence maximale des probabilités = 0.0.

| Variante | Accuracy test | Log-loss test |
|---|---:|---:|
| Greedy de référence (benchmark uniquement) | 53.05185185 % | 0.7916316057169205 |
| Ordonnée partagée | 57.55555556 % | 0.6795818068597846 |
| Neutre | 57.34814815 % | 0.6794976248053235 |

## Limites

Les entraînements courts sur fixture synthétique sont des validations de fonctionnement sur CPU, pas une mesure de qualité prédictive. Le rejeu de mai est une non-régression sans réglage ; aucun gain du modèle actif n'est établi. La temporalité des priors joueurs attachés reste inchangée et la fin du match reste un proxy optimiste de disponibilité, faute d'heure d'ingestion API.

Une tentative exploratoire du script général avec `use_pcgrad=False` a révélé le défaut préexistant `KeyError: winner_uncertainty`. Ce mode reste non validé ; les trois parcours normaux PCGrad demandés ont été vérifiés. Aucune validation du protocole demandé n'est bloquée. Aucune dépendance ajoutée, migration SQLite, activation dans PredictionService ni remplacement du modèle actif. Aucune recherche bibliographique relancée.
