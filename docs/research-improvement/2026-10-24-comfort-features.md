# Features de confort et historique complet — objectif 60 % d'accuracy (24 octobre 2026)

## Résumé

Objectif demandé : 60 % d'accuracy à couverture 100 %. **Non atteint sur données tenues à l'écart.**

| Modèle (validation officielle, 5 000 matchs, 16–17 oct.) | Accuracy | Wilson 95 % | AUC | Log-loss |
|---|---:|---:|---:|---:|
| Constante (Red) | 51,26 % | — | 0,500 | ≈ 0,693 |
| LightGBM 18D préenregistré (`baseline_lightgbm18.json`) | 55,58 % | [54,20 ; 56,95] | 0,575 | 0,6848 |
| **Mélange 0,4 LR + 0,6 LightGBM (ce travail)** | **58,30 %** | **[56,93 ; 59,66]** | **0,617** | **0,6720** |
| Même modèle, 50 % des matchs les plus confiants (seuil fixé sur le dev) | 62,85 % | [60,96 ; 64,71] | 0,642 | 0,6556 |

- Gain apparié contre le LightGBM 18D : **+2,72 points**. Bootstrap par blocs de 100 matchs consécutifs (2 000 tirages) : [+1,02 ; +4,32]. Discordances : 812 erreurs corrigées, 676 introduites.
- La borne haute de l'intervalle (59,66 %) exclut 60 %.
- Sur le dev (les 12 000 derniers matchs du train), le même modèle obtenait 59,62 %. La baisse en validation tient au moins en partie à l'artefact du crawler décrit plus bas : la validation est une tranche de 21 h issue d'une campagne d'enrichissement.
- Le test scellé du protocole prospectif du 23 octobre n'a été ni lu ni featurisé. L'extraction s'arrête strictement avant sa première création.

## Ce qui a changé par rapport aux tenseurs 14D/18D

1. **Historique complet point-in-time.**
   - Les priors servis ne rejouent que les 10 000 derniers matchs de la base (`history_window_limit=10000`), soit une médiane de 4 à 7 parties connues par joueur.
   - Ici, chaque cible utilise tout match CLASSIC terminé strictement avant sa création. Les remakes sont exclus de l'historique.
2. **Champ select de la cible**, information pré-match :
   - statut de rôle `selectedRolePreferences` (PRIMARY / SECONDARY / AUTOFILL / FILL) ;
   - niveau de compte ;
   - sorts d'invocateur ;
   - keystone et styles de runes.
3. **Confort et forme**, calculés depuis les matchs passés :
   - expérience sur le champion (total, sur 7 et 30 jours, rang dans le pool) ;
   - courbe d'apprentissage globale par champion et par tranche d'expérience ;
   - fréquence récente du rôle ;
   - EWMA des performances (KDA, part d'or et de dégâts, vision, CS) ;
   - Elo global et Elo par rôle ;
   - duos détectés et taux de victoire du duo ;
   - synergie des paires de champions ;
   - taux de victoire « méta » à demi-vie de 8 000 résultats ;
   - vecteur macro clash 15D existant.
4. **Modèles.**
   - Différences Red − Blue par rôle et par équipe.
   - LightGBM avec augmentation par échange des équipes et prédiction symétrisée.
   - Régression logistique sur les seules différences (anti-symétrique par construction, plus un biais de côté).
   - Mélange convexe des deux, poids choisi par log-loss sur le dev.

## Itérations sur le dev (fit sur train[:-12 000], dev = 12 000 derniers matchs du train)

| Étape | Accuracy dev | AUC | Log-loss |
|---|---:|---:|---:|
| Tenseurs 18D existants + LightGBM | 56,85 % | 0,591 | 0,6795 |
| v1 : historique complet + champ select | 58,37 % | 0,609 | 0,6739 |
| v1 + différences + swap | 58,83 % | 0,621 | 0,6695 |
| v2 : forme EWMA, Elo par rôle, synergie, duos | 58,98 % | 0,624 | 0,6682 |
| v3 : expérience champion fine, courbe d'apprentissage | 59,07 % | 0,628 | 0,6665 |
| v4m : méta pondéré dans le temps, macro clash | 59,14 % | 0,629 | 0,6658 |
| v4m, LightGBM réglé (B : 15 feuilles, min_leaf 400, ff 0,5) | 59,14 % | 0,633 | 0,6643 |
| v4m, régression logistique (C = 0,001) | 59,58 % | 0,632 | 0,6652 |
| **v4m, mélange 0,4 LR + 0,6 LGB (retenu)** | **59,62 %** | **0,635** | **0,6636** |

Essais rejetés, sans gain ou avec une dégradation :
- ratings Bradley-Terry en ligne (AUC seul 0,515 à 0,545) ;
- compteurs d'équipe (premières parties sur le champion, hors-rôle, bas niveau, jungler sans Châtiment, séries) ;
- taux joueur-champion hiérarchique ;
- pondération temporelle des exemples (−0,0005 nat, dans le bruit) ;
- v5 : position dans la session, tendance AFK / remake / FF précoce, dernière partie. La log-loss LR passe de 0,6652 à 0,6659.

Environ quinze configurations ont été comparées sur le dev. L'estimation du dev est donc légèrement optimiste. La configuration a été gelée dans `runs/comfort_features_20261024/lock.json` (hachages des features et des sources) avant l'unique lecture de la validation.

## Audits de fuite et de sélection

- **`summonerLevel` est pré-partie.** Sur 741 260 paires de parties consécutives à moins de 2 h d'écart, la montée de niveau entre N et N+1 dépend de la victoire en N (6,92 % contre 6,18 %), pas de celle en N+1 (6,48 % contre 6,63 %).
- **`teamPosition` est égal au rôle attribué en champ select** (premier jeton de `selectedRolePreferences`) dans 99,98 % des 17 000 matchs dev et validation. L'ordre canonique des rôles ne fuit donc pas.
- **Biais de survivant du crawler.** `CollectionService` amorce la file avec les joueurs du haut du ladder (priorité 10, 150 joueurs), puis fait boule de neige sur les participants.
  - Taux de victoire des joueurs, toutes parties classées confondues : seeds du ladder 53,8 % ; joueurs crawlés 50,7 % ; joueurs jamais crawlés 49,0 %.
  - Dans le train, l'équipe qui compte le plus de joueurs crawlés gagne 55,3 %, et l'équipe d'un seed 56,2 %.
  - Les features de profondeur d'historique captent donc en partie un artefact de collecte.
  - Contrôle : sur les matchs sans seed et à nombre de crawlés égal, l'accuracy vaut 59,77 % en dev (9 875 matchs) et 59,42 % en validation (3 024 matchs), contre 59,62 % et 58,30 % sur l'ensemble. Le modèle ne repose donc pas sur l'artefact. En validation, les matchs déséquilibrés (40 % de la tranche) le pénalisent.
  - **À corriger côté collecte**, ou à neutraliser : marquer les seeds et équilibrer, ou pondérer, les matchs selon la couverture.

## Plafond attendu

Synthèse de la littérature : [reports/Signaux prédictifs avant match LoL.md](../../reports/Signaux%20prédictifs%20avant%20match%20LoL.md).

- Le meilleur résultat publié sans fuite en LoL classé est celui de DraftRec : 56,18 % (top 0,1 %, split chronologique).
- Les chiffres de 60 à 75 % reposent sur des splits aléatoires ou des instantanés d'API postérieurs au match.
- Le plafond plausible à couverture complète se situe vers 60–62 % sur une population tous rangs. Au-delà de 63 %, il faut suspecter une fuite.
- 58,3 % sur une tranche de 21 h est donc cohérent avec ce plafond. 60 % reste possible, mais il faudrait un test plus large pour le démontrer : l'IC à 95 % vaut ±1,4 point sur 5 000 matchs et ±1,0 sur 10 000.

## Reproduction

Scripts sous `docs/research-improvement/comfort_features_20261024/`, à lancer avec `.venv312` :

1. `extract_raw.py <out.pkl>` : une passe en lecture seule sur `training_data.db` (environ 7 min sur le HDD).
2. `build_features.py <raw.pkl> runs/aligned_dataset_20261023_prospective/model.split.json <feat.npz>`, puis `join_macro.py <feat.npz> <feat_m.npz>`. `build_features.py` correspond à la v4. Les `patch_v*.py` documentent la construction incrémentale.
3. `evaluate.py <feat_m.npz> --model lr --final --save-pred …`, puis `evaluate.py <feat_m.npz> --swap --threads 10 --max-bin 63 --es 300 --lr 0.02 --leaves 15 --min-leaf 400 --ff 0.5 --final --save-pred …`.
4. `final_report.py <val_lr> <val_lgb> 0.4 <dev_lr> <dev_lgb> <out.json>`.

Artefacts : `runs/comfort_features_20261024/` (verrou, métriques, prédictions dev et validation).

## Limites et suites

- **Rien n'est promu au service.** L'inférence exigerait un fournisseur d'historique sans fenêtre de 10 000 lignes, les champs de champ select en entrée de l'API et une calibration vérifiée, conformément à `docs/model_serving/README.md`.
- **Test scellé.** Une évaluation unique du modèle gelé sur le test scellé (10 000 matchs, 17–18 oct.) donnerait une mesure deux fois plus précise. À faire seulement après la clôture du protocole prospectif, et sans aucun réglage ensuite.
- **Leviers probables, d'après la littérature :**
  - corriger le biais de seeds du crawler ;
  - collecter une population tous rangs sur plusieurs semaines ;
  - modèle séquentiel sur les 20 à 50 dernières parties de chaque joueur (DraftRec : +1,9 point en split chronologique).
