### Task 6: Dataset réel, audit de disponibilité et documentation de réception

**Files:** Modify `docs/pre_match_training.md`, `docs/model_serving/README.md`; Regenerate `docs/model_serving/feature_contracts.v1.json`, `docs/model_serving/reference.manifest.json`, `docs/model_serving/manifests/` via le script existant ; Create les artefacts sous `runs/aligned_dataset_20261009/` et `docs/model_serving/aligned_dataset_verification.md`.

**Interfaces:**
- Consumes les trois sous-commandes `inventory`, `freeze`, `verify` et le nouveau paramètre `--dataset-dir` des trainers.
- Produces un snapshot réel 14D avec parité vérifiée, un rapport de disponibilité 18D, les commandes exactes et un bilan des limites.

- [ ] **Step 1: Fixer les assertions de réception avant préparation réelle.** Ajouter ce test de CLI sur une fixture publiée, sans dépendre des bases de plusieurs Go :

```python
def test_verify_cli_has_no_prediction_metrics(tmp_path, capsys):
    from scripts.prepare_aligned_dataset import main
    directory, _ = tiny_bundle(tmp_path, timeline=True)
    assert main(['verify', '--dataset-dir', str(directory)]) == 0
    text = capsys.readouterr().out
    assert 'verified' in text
    assert 'test_metrics_computed' in text
    assert 'accuracy' not in text and 'log_loss' not in text
```

- [ ] **Step 2: Exécuter la réception avant le run réel**, `.\.venv312\Scripts\python.exe -m pytest tests/test_aligned_artifacts.py tests/test_aligned_parity.py tests/test_aligned_training.py -q`. Un échec interdit de publier le dataset comme vérifié. Cette tâche ne demande pas de fabriquer un test rouge sur une documentation ou une commande déjà implémentée.

- [ ] **Step 3: Préparer puis vérifier les artefacts réels.** Vérifier que `runs/aligned_dataset_20261009/` n'existe pas ; s'il existe, utiliser un suffixe neuf et reporter les chemins réellement utilisés dans la documentation. La base consolidée reste candidate jusqu'à la lecture du rapport de sélection.

```powershell
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset inventory --db training_data.db --db runs/training_data_23h_20260922_191434.db --db runs/db_merge_20260924/consolidated.db
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset freeze --db runs/db_merge_20260924/consolidated.db --after-cohort runs/served_model_comparison_20261009_complete/cohort.jsonl --output-dir runs/aligned_dataset_20261009 --source-limit 61000
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset verify --dataset-dir runs/aligned_dataset_20261009
```

Si la purge ou la sélection ne laisse pas les minima définis, écrire le diagnostic et corriger la source/les données avant tout changement de protocole. Ne pas réduire discrètement les minima. Une source sans timelines doit publier `ready` pour 14D et `blocked` pour 18D, avec couverture 0 et les causes effectives.

Choisir un panel de 20 IDs de train/validation, déterministe et fixé avant comparaison : premières/dernières lignes et IDs espacés dans l'ordre temporel. Rejouer chacun au cutoff stocké avec la base source en lecture seule et `DatabaseHistoryProvider`, puis enregistrer hashes et comparaison exacte des priors/tenseurs. Si la base a évolué depuis le snapshot, déclarer cette différence et comparer sur une copie/source stable ; ne pas réécrire les tensors figés pour faire passer l'audit.

Ce chantier ne lance pas d'entraînement long. Les tests de tâche 4 établissent le branchement au premier batch et la clôture sans test. Documenter les commandes de futurs pilots, mais ne pas les exécuter sur le dataset réel avant le protocole d'expérience :

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_deepsets_v3 --dataset-dir runs/aligned_dataset_20261009 --output-dir runs/aligned_v3_pilot --epochs 1 --batch-size 64 --seed 42
.\.venv312\Scripts\python.exe -m scripts.train_deepsets_v4_timeline --dataset-dir runs/aligned_dataset_20261009 --output-dir runs/aligned_v4_pilot --epochs 1 --batch-size 64 --seed 42
```

La deuxième commande doit refuser un dataset `blocked_18d` avant création du modèle ou apprentissage. Indiquer cette condition à côté de la commande.

- [ ] **Step 4: Remettre les manifestes de service en cohérence et vérifier les régressions.**

```powershell
.\.venv312\Scripts\python.exe -m scripts.formalize_served_model
.\.venv312\Scripts\python.exe -m pytest tests/test_frozen_history.py tests/test_aligned_partitions.py tests/test_aligned_artifacts.py tests/test_aligned_training.py tests/test_aligned_parity.py tests/test_serving_contract.py tests/test_pre_match_history_serving.py tests/test_serving_http.py tests/test_predict_api.py tests/test_config_cli.py tests/test_ordered_training_integration.py tests/test_train_deepsets_v4_timeline.py tests/test_equivariant_model.py tests/test_deepsets_layers.py -q
```

Les checkpoints et calibrations existants ne changent pas. Le script de formalisation doit conserver `unverified` et l'abstention existante. Les anciens protocoles de comparaison gardent leurs hashes de runtime d'origine ; un replay avec le nouveau code doit refuser la divergence, pas réécrire leur historique.

Dans les deux documents existants, ajouter les commandes exactes, les schémas d'artefacts, la politique de purge, la logique de réservation test et la limitation 18D. Dans le rapport de réception, inclure les nombres réellement sélectionnés/exclus/purgés, dates UTC, régions/queues/patches, hashes, couverture par contrat, comparaison SQLite, commande pytest et résultat observé. Aucun chiffre de performance nouvelle n'est attendu.

- [ ] **Step 5: Revoir les artefacts et isoler le commit** `docs: record aligned dataset provenance and availability`. Inclure uniquement les docs/manifestes concernés ; ne pas ajouter la base SQLite ni des centaines de Mo de snapshot à Git. Conserver les gros artefacts sous `runs/` comme les runs existants et référencer leurs hashes dans la documentation.

## Autorelecture du plan

- Couverture : replay partagé = tâche 1 ; population/coupures = tâche 2 ; hashes et disponibilité = tâche 3 ; trainers/calibration/réservation = tâche 4 ; parité et invariants = tâche 5 ; données réelles et documentation = tâche 6.
- Les cinq classes de risque de Review Focus disposent de tests propriétaires indiqués ; les nouveaux tests de SQLite réutilisent un schéma Riot de timeline valide.
- Les signatures et noms de contrats sont communs à toutes les tâches ; les tableaux ont un ordre fixé par les records et aucun trainer ne filtre sa propre cohorte.
- Les blocs de code décrivent les nouvelles interfaces et les modifications minimales ; l'architecture réseau et les chemins legacy restent hors de la certification nouvelle.
- Point de décision matériel : les données actuelles permettent de préparer 14D ; 18D récent nécessite des timelines supplémentaires. Ce plan livre son diagnostic et le refus explicite, puis la collecte a son propre périmètre.

Autorelecture effectuée le 9 octobre 2026 : les blocs Python du plan ont été analysés avec `ast.parse`, sans erreur de syntaxe ; les contraintes globales ont été comparées à la spec et sont identiques. La fixture de labels et l'exemple de purges aux frontières ont été exécutés en mémoire et correspondent aux assertions proposées. Le chemin du cohort JSONL a été vérifié sur disque. Ces contrôles portent sur le plan ; les nouvelles suites d'implémentation restent à créer et exécuter.

## Handoff

Recommandation : **Native**, avec exécution des six tâches dans l'ordre puis une revue indépendante finale, car les interfaces et le contrat d'historique se transmettent d'une tâche à la suivante. La correction d'une fuite temporelle ou d'une mauvaise réservation test exige une revue attentive ; les tests des frontières doivent précéder les gros runs.

Faire relire le plan et choisir la méthode d'exécution avant l'implémentation, conformément au skill `superpowers:writing-plans`. Une fois la méthode choisie, utiliser `superpowers:executing-plans` pour Native ou `superpowers:subagent-driven-development` pour l'exécution avec sous-agents.
