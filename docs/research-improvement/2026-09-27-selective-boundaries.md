# Recherche NNriot — cohérence des frontières d'abstention

## Session et référence

- Invocation du 2026-09-27 : maximum trois cycles, une amélioration par cycle.
- Cycle 1, distance 0, état initial : planifié avant modification du code.
- Objectif observable : mêmes matchs retenus dans évaluation et inférence pour
  les mêmes probabilités et seuils, y compris aux frontières et après swap.
- Racine fixe : classification sélective sur la conviction `abs(P(B) - 0.5)`.
- HEAD initial : `24e7bd9`. Les modifications GPU préexistantes, les artefacts
  sous `runs/` et `docs/pre_match_training.md` sont hors périmètre.
- Référence exécutée : `.venv312/Scripts/python.exe -m pytest tests/test_selective_classification.py tests/test_selective_infer.py tests/test_model_calibration.py -q --tb=short` : **43 passed in 17.23s**.
- Défaut reproduit : `evaluate_selective_coverage([0], [.46], .04)` retient zéro
  match ; le swap `([1], [.54], .04)` en retient un. Le service classe déjà `.46`
  comme ACTIONABLE. Cause : comparaison brute contre arrondi à six décimales
  et tolérance `1e-9` en service.
- Le bac à sable échoue avant démarrage (`setup refresh had errors`) ; commandes
  locales avec élévation approuvée. Firecrawl absent du PATH et du dossier npm
  utilisateur ; recherche via l'outil web disponible.
- Graphe consulté : `graphify query 'calibration selective conviction' --budget 1300`
  renvoie l'ancienne analyse rétrospective, pas les modules sélectifs actuels.
  Dépendances confirmées dans les fichiers réels : `core/training_metrics.py`
  utilise la stratification ; `core/prediction_service.py` utilise l'inférence.

## Historique conservé

Sources d'origine inchangées :

- [Classification sélective, 2026-09-26](../plans/2026-09-26-research-driven-selective-classification-conviction-gate.md).
- [DeepSets et PCGrad, 2026-09-26](../plans/2026-09-26-research-driven-equivariant-deepsets-pcgrad.md).
- [Architecture neuronale, 2026-09-27](../superpowers/plans/2026-09-27-research-driven-neural-architecture.md).

Aucun journal numéroté `docs/research-improvement/` n'existait. Aucune distance
rétrospective inventée pour ces plans. Le plan sélectif confond SelectiveNet (2019)
avec le papier de 2017 et annonce une garantie de 61,5 % non établie par une
accuracy empirique et un Wilson ponctuel. Aucune telle garantie revendiquée ici.

## Recherche et sélection

Requêtes exécutées via web :

1. `Geifman El Yaniv 2017 selective classification deep neural networks selection threshold risk coverage sorted`
2. `Franc Vora Voracek optimal strategies reject option classification selective risk 2023`
3. `"Selective Classification Under Distribution Shifts" paper`
4. `"Learn then Test" Angelopoulos Bates Candès Jordan Lei calibration risk guarantees`

Publications primaires, dédoublonnées :

- **Geifman, El-Yaniv (2017), Selective Classification for Deep Neural Networks**, NeurIPS, arXiv:1705.08500. [Texte](https://papers.nips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf).
  Sections 2–3, équations (1)–(4), algorithme 1 lus (pages 2–4).
  Métriques et prédictions utilisent la même fonction de sélection. Le contrôle
  du risque repose sur une borne binomiale, des comparaisons corrigées et des
  observations iid. NNriot n'implémente pas cet algorithme de certification.
- **Franc, Prusa, Voracek (2023), Optimal Strategies for Reject Option Classifiers**, JMLR 24(11), arXiv:2101.12523. [Texte](https://www.jmlr.org/papers/volume24/21-0048/21-0048.pdf).
  Introduction et sections 2.1–2.2, équations (5)–(8), théorèmes 2–3 lus
  (pages 4–7). Couverture et risque dépendent de la sélection ; égalités incluses
  dans sa définition. L'optimalité suppose distribution/risque connus : elle ne
  prouve pas l'optimalité du score NNriot.
- **Liang, Peng, Sun (2024), Selective Classification Under Distribution Shifts**, TMLR, arXiv:2405.05160v2, OpenReview dmxMGW6J7N. [Texte](https://arxiv.org/html/2405.05160v2).
  Sous-agent : sections 2.1–2.5 lues, aperçu 3.1. La section 2.5 exclut la
  calibration et la sélection du seuil du périmètre. RC/AURC mesurent le potentiel
  des scores, sans certifier un seuil sur un patch futur de NNriot.
- **Angelopoulos, Bates, Candès, Jordan, Lei, Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control**, arXiv:2110.01052v5 (2022), publication finale 2025, DOI 10.1214/24-AOAS1998. [Version lue](https://arxiv.org/pdf/2110.01052v5).
  Sous-agent : sections 1.1, 2.1–2.2, théorème 1, 3.2, annexe C et début D lus.
  P-values valides et contrôle FWER permettent une garantie simultanée, sous iid.
  Optimiser sur le même jeu parmi les seuils simultanément certifiés est permis ;
  un Wilson ponctuel après recherche ne suffit pas à établir cette garantie.

**Transfert proposé (inférence locale, pas résultat des auteurs)** : partager la
règle de sélection déjà servie pour mesurer ce même classifieur. Les six décimales
et `1e-9` sont une convention locale, non une recommandation des publications.
Aucun chiffre expérimental de ces articles n'est transféré à NNriot.

Candidats :

- **Retenu :** cohérence masque évaluation/service ; défaut observé, modification
  localisée et vérifiable sans entraînement.
- **Différé :** recherche exhaustive des seuils. Contre-exemple indépendant :
  `p=.600001+arange(1000)*1e-6`, labels un sauf les deux premiers et les 99 derniers,
  cible `.9007`. Un seuil retient 998 points à `.9008016032` d'accuracy ; la grille
  renvoie `target_met=False`. Aucun appel de production trouvé dans core/scripts.
- **Non retenu :** SGR/contrôle conforme, nouvelle architecture. Nécessite un
  protocole de validation et des hypothèses temporelles au-delà de cet objectif.

## Plan avant édition

1. Tests avec attentes indépendantes : frontières 4 %/8 %, valeurs juste autour,
   seuil personnalisé, swap, couverture/accuracy/Wilson. Observer les échecs.
2. Extraire un prédicat commun dans `core/selective_classification.py`, appelé
   depuis évaluation et `core/selective_infer.py`. Conserver la précision de
   service (round scalaire Python à six décimales, tolérance `1e-9`) ; examiner les
   demi-unités d'arrondi pour ne pas introduire un écart scalaire/vectoriel.
   Garder `compute_conviction` brut et les formats publics.
3. Vérifier les consommateurs stratification et recherche à seuil explicite ;
   tests sélectifs/calibration, suite Python puis revue indépendante.
4. Accepter si contre-exemples résolus et non-régression vérifiée. Pas de nouvelle
   dépendance, feature, modèle, base, checkpoint ni changement de probabilités.
   Retour arrière limité à nos hunks et nouveaux tests, préserver travaux GPU.

## Validation et reprise

À compléter après exécution. Deux places restent après clôture de ce cycle.
Prochaine frontière possible, non explorée : classification sélective → contrôle
statistique des seuils multiples, lien documenté par Geifman/El-Yaniv algorithme 1.
Hypothèse de transfert aux données temporelles encore à vérifier.

### Raffinement avant implémentation : demi-unités

L'audit indépendant a reproduit un défaut supplémentaire de la même règle :
`p=.4600005` est ACTIONABLE, mais `1-p=.5399995` est COINFLIP. Arrondir deux
soustractions différentes ne suffit donc pas. Avant l'arrondi, calculer la
conviction sous la forme équivalente `max(p, 1-p) - .5`, qui canonise le côté
favori. `compute_conviction` garde une valeur brute (non quantifiée), mais peut
changer de quelques ulps. Le service ne change que sur les frontières d'arrondi
auparavant dépendantes du côté ; ces décisions seront explicitement testées.

## Résultats intermédiaires vérifiés

- Tests de régression écrits avant le patch : **9 échecs, 8 réussites**, 41
  désélectionnés (`-k boundar` sur les deux fichiers modifiés). Échecs attendus
  dans la sélection, les métriques et la frontière d'arrondi du service.
- Après patch : `.venv312/Scripts/python.exe -m pytest tests/test_selective_classification.py tests/test_selective_infer.py tests/test_training_metrics_calibration.py tests/test_model_calibration.py -q --tb=short` : **89 passed in 11.65s**.
- [Contrôle numérique enregistré](2026-09-27-boundary-check.json) : 1 100 000
  probabilités (1 000 000 demi-unités six décimales, deux orientations ; 100 000
  aléatoires, seed 20260927). Zéro divergence de conviction canonique et des
  masques aux seuils .025/.04/.08 après swap. Zéro changement de décision au
  seuil .04 sur les 100 000 aléatoires par rapport au service initial.
- Cas `.46` et `.54` : désormais tous deux retenus, conviction `.04`, ACTIONABLE.
  Cas `.4600005` et `.5399995` : tous deux rejetés, conviction `.039999`, COINFLIP.
- L'arrondi Python scalaire est conservé aussi dans le masque vectoriel, via
  `fromiter`, pour éviter les différences de demi-unité de `np.round`.
- Coût observé de l'évaluation 45 000 lignes : 36–86 ms, cinq mesures pendant la
  suite complète. Ce n'est pas un benchmark contrôlé et aucun gain de vitesse
  n'est annoncé. Le changement vise la correction, avec mémoire linéaire.
- Les pourcentages de précision fixes sans preuve ont été retirés de la docstring
  des tiers. Noms des tiers, API, probabilités et formats persistés conservés.
- `git diff --check` limité aux quatre fichiers de code/test : succès. Avertissement
  Git habituel de conversion LF/CRLF, pas d'erreur de whitespace.
- Suite complète en cours, sortie conservée dans
  [2026-09-27-full-pytest.txt](2026-09-27-full-pytest.txt). Acceptation finale en attente.

Revue indépendante terminée : aucune anomalie actionnable introduite. Vérification
sur 110 008 probabilités supplémentaires et dix seuils : conviction exactement
invariante, réarrondi idempotent, concordance service/évaluation aux frontières,
formes et clipping conservés, NaN/infinis rejetés. SciPy est déjà une dépendance
explicite ; le nouveau chemin d'import ne crée aucune dépendance installée.
Limites préexistantes consignées sans extension de périmètre : égalité p=.5 avec
seuil nul, validation des seuils non finis, recherche de seuil approximative.

## Clôture du cycle 1 — accepté

- Validation complète exécutée : `.venv312/Scripts/python.exe -m pytest -q --tb=short`.
  **714 passed, 1664 warnings in 343.14s**, code de sortie **0**. Les avertissements
  concernent notamment dépréciations des dépendances/datetime et rechargement des
  variables d'optimiseur Keras ; aucun test en échec. La sortie complète est jointe.
- Amélioration retenue : règle commune d'abstention avec conviction canonique,
  cohérence des métriques et du service aux frontières. Les neuf échecs nouveaux
  sont corrigés, les 17 nouveaux cas passent et la suite complète passe.
- Invariants cumulés : swap numérique vérifié, probabilités retournées inchangées,
  interfaces et schémas conservés, pas d'édition des features/encodeurs/SQLite,
  aucune nouvelle dépendance. Les travaux GPU préexistants ont été préservés.
- Aucun entraînement de candidat ni déploiement effectué. Les appels de modèles
  dans les tests font partie de la suite de validation isolée. Aucun gain de
  précision, de calibration ou de performance future revendiqué.
- Aucun essai de code rejeté à restaurer : une seule correction implémentée,
  validée puis conservée. Les autres candidats sont différés, pas implémentés.
- Distance atteinte : **0**. Un cycle consommé sur trois, **deux places non utilisées**.
  Compteur de cycles clos consécutifs sans amélioration acceptée : **0**.
- **Raison d'arrêt : objectif observable atteint**, selon la règle du skill.
  Cet arrêt ne signifie pas que la littérature est épuisée.
- **Point de reprise :** relire ce journal, vérifier le nouvel état du dépôt et
  redéfinir le prochain objectif mesurable. Piste locale déjà reproduite : recherche
  exacte des seuils. Prochaine frontière scientifique possible : contrôle
  statistique des seuils multiples, lien établi plus haut ; ne pas présumer que
  les garanties iid s'appliquent aux matchs temporels. Aucun cycle 2 démarré.

Fichiers du correctif : `core/selective_classification.py`,
`core/selective_infer.py`, `tests/test_selective_classification.py`,
`tests/test_training_metrics_calibration.py`. Journal et preuves sous ce dossier.
