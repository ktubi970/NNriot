<!-- PATTERN-DESIGN-START -->
## Conventions d'architecture NNriot

> Conventions validées suite à l'audit architectural de septembre 2026, conformes au skill pattern-design.

### 1. Invariance, Anti-Symétrie et Découplage de Terrain (Domain Invariants)
- **Anti-symétrie de Composition (Draft Combat Engine)** :
  L'interaction pure des compositions (synergies champion FM, confrontations de lane Sinkhorn, clash macro) est strictement anti-symétrique :
  $$s_{\text{draft}}(\text{Team B}, \text{Team A}) = -s_{\text{draft}}(\text{Team A}, \text{Team B})$$
  Pour un affrontement miroir ou équilibré ($T_A = T_B$), $s_{\text{draft}} = 0$ et la tête dédiée `draft_advantage` garantit $P_{\text{draft}} = [0.50, 0.50]$ par construction.
- **Découplage de l'Asymétrie de Terrain (Blue Side Advantage)** :
  L'effet du côté est découplé de l'évaluation pure des compositions via la couche apprenable `DirectionalSideBias`. Son amplitude dépend de la population de matchs et de la période considérées ; elle doit être estimée et documentée :
  $$s_{\text{match}} = s_{\text{draft}} + b_{\text{side}}$$
  La tête `winner` estime la probabilité de victoire en intégrant le biais de côté appris, tandis que `draft_advantage` conserve la pureté analytique de la draft. Une valeur indicative telle que 52,5 % sur draft neutre n'est pas un invariant numérique ni une garantie après entraînement ou calibration.
- **Sémantique du swap d'équipe** :
  Team A correspond au côté Blue et Team B au côté Red. Échanger les compositions en conservant ces emplacements inverse le score de `draft_advantage` et échange ses probabilités, mais conserve le biais de côté de `winner` : la complémentarité de `winner` n'est donc pas exigée pour cette transformation.
  Changer de perspective sur le même match consiste à considérer la victoire de l'autre équipe, sans modifier le match ni l'affectation des côtés ; la probabilité devient alors $1 - P$. Cette opération est distincte d'un échange des compositions entre Blue et Red.
- **Invariance par permutation intra-équipe** :
  L'ordre arbitraire des 5 joueurs au sein d'une même équipe est canonisé par tri de rôle (`TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY`) et par couches DeepSets de pooling symétrique, invariant pour tous profils de tenseurs (14-dim base, 18-dim timeline priors v4, 46-dim spatial).

### 2. Séparation en Couches et Rôles des Modules (Pattern Boundaries)
- **Couche Données (`core/database.py`, `core/point_in_time_dataset.py`)** :
  - Accès SQLite centralisé (WAL, transactions isolées, PRAGMA read-only lors du streaming d'entraînement).
  - Streaming chronologique strict sans fuite d'information future (point-in-time Bayesian priors).
- **Couche Features (`core/features/`) — Pattern Strategy** :
  - `static_meta.py` : Registre immuable des caractéristiques des champions (profils de dégâts AD/AP, indices d'engage et de CC, courbe de scaling).
  - `matchup_matrix.py` : Priors empiriques de winrates de patch et de confrontations de lane directes ($WR_{lane}(A \mid B)$).
  - `draft_composition.py` : Calculs macro sans état de composition (équilibre des dégâts, vulnérabilité full AD, CC total).
  - `pipeline.py` : Façade/Strategy produisant à la fois les tenseurs pour réseaux siamois et les vecteurs tabulaires enrichis pour LightGBM.
- **Couche Modèles (`core/equivariant_model.py`, `core/deepsets_layers.py`)** :
  - Encodeurs DeepSets siamois et mécanismes d'attention croisée anti-symétriques.
  - Découplage strict des têtes principales (`winner`, `team_b_kill_lead`, `kill_handicap`) des têtes auxiliaires bruyantes.
- **Couche Décision & Sélection (`core/selective_classification.py`, `core/selective_infer.py`)** :
  - Règle de Chow pour la classification avec option de rejet.
  - Conviction $C = |P - 0.5|$ invariante sous $P \rightarrow 1 - P$ lors d'un changement de perspective sur le même match. Cette invariance n'est pas exigée pour `winner` lors d'un échange des compositions à côtés fixes.
  - Intervalles de confiance de Wilson pour encadrer statistiquement les taux d'erreur.
- **Couche Service & Inférence (`core/prediction_service.py`, `app/routes/api.py`) — Pattern Facade** :
  - Aucun endpoint HTTP Flask ne manipule directement de modèles bruts ou de tenseurs NumPy sans passer par `PredictionService`.
  - La réponse d'inférence expose les probabilités brutes, les probabilités calibrées lorsqu'un artefact de calibration vérifié est disponible (sinon `null`), et le statut de calibration. Elle inclut le seuil de conviction appliqué, le tier d'actionnabilité et le statut d'abstention.
  - Si la calibration n'est pas vérifiée ou si le modèle servi ne dispose pas d'une validation indépendante, l'abstention est obligatoire. Ne jamais présenter des probabilités brutes comme calibrées. Le contrat détaillé est documenté dans [docs/model_serving/README.md](docs/model_serving/README.md).

### 3. Règles d'Évolution et Tests de Non-Régression
- Tout nouveau format de feature ou changement de dimension doit être testé sous permutation intra-équipe et échange des compositions à côtés fixes (`tests/test_deepsets_layers.py`, `tests/test_equivariant_model.py`). Les tests doivent préciser la transformation et la tête concernées : permutation intra-équipe pour les deux têtes ; anti-symétrie et complémentarité pour `draft_advantage` ; maintien du biais de côté pour `winner`, sans lui imposer la complémentarité lorsque ce biais est activé. L'invariance de la conviction sous $P \rightarrow 1 - P$ se vérifie séparément comme changement de perspective sur le même match.
- Les modifications de schéma SQLite restent soumises au protocole de migration documenté dans `docs/database_merge.md`.
<!-- PATTERN-DESIGN-END -->
