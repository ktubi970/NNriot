<!-- PATTERN-DESIGN-START -->
## Conventions d'architecture NNriot

> Conventions validées suite à l'audit architectural de septembre 2026, conformes au skill pattern-design.

### 1. Invariance et Anti-Symétrie Mathématique (Domain Invariant)
- **Anti-symétrie sous permutation d'équipe** : Tout modèle de prédiction du vainqueur, de kill lead ou de handicap de kills DOIT respecter mathématiquement l'anti-symétrie :
  $$P(\text{winner}=1 \mid \text{Team B}, \text{Team A}) = 1 - P(\text{winner}=1 \mid \text{Team A}, \text{Team B})$$
  Pour les modèles neuronaux, cela est garanti par construction via `SiameseTeamEncoder` et `AntiSymmetricSoftmax`.
  Pour les modèles tabulaires d'arbres (LightGBM, GBDT), cela est imposé par symétrisation explicite à l'inférence :
  $$\hat{P}_{\text{sym}} = \frac{1}{2} \left( P(B \mid A, B) + 1 - P(B \mid B, A) \right)$$
- **Invariance par permutation intra-équipe** : L'ordre arbitraire des 5 joueurs au sein d'une même équipe est canonisé par tri de rôle (`TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY`) et par couches DeepSets de pooling symétrique.

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
  - Conviction $C = |P - 0.5|$ invariante par inversion d'équipe.
  - Intervalles de confiance de Wilson pour encadrer statistiquement les taux d'erreur.
- **Couche Service & Inférence (`core/prediction_service.py`, `app/routes/api.py`) — Pattern Facade** :
  - Aucun endpoint HTTP Flask ne manipule directement de modèles bruts ou de tenseurs NumPy sans passer par `PredictionService`.
  - La réponse d'inférence inclut toujours les probabilités calibrées, le seuil de conviction, le tier d'actionnabilité et le statut d'abstention.

### 3. Règles d'Évolution et Tests de Non-Régression
- Tout nouveau format de feature ou changement de dimension doit être testé sous permutation de joueurs et swap d'équipe (`tests/test_deepsets_layers.py`, `tests/test_equivariant_model.py`).
- Les modifications de schéma SQLite restent soumises au protocole de migration documenté dans `docs/database_merge.md`.
<!-- PATTERN-DESIGN-END -->
