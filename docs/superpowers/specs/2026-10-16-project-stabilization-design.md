# Stabilisation complète de NNriot — périmètre et décisions

Date : 16 octobre 2026. Demande : « /plan fix all », après le bilan du projet.
Ce document décrit les travaux proposés ; il ne constate pas leur exécution.

## Résultat recherché

Obtenir une version reproductible du logiciel dont la suite complète passe, dont
les migrations et fusions préservent les données, et dont l'interface respecte
les mêmes contrats que le service réel. Fournir ensuite une évaluation ML
indépendante qui permet une décision explicite, y compris le maintien de
l'abstention si les résultats restent insuffisants.

Le périmètre retenu est la correction et la validation des capacités existantes.
La création de Live/OCR et l'activation du modèle spatial 46D constituent des
fonctionnalités supplémentaires : conserver leur indisponibilité explicite.
Ce choix de périmètre est une interprétation de la demande, pas une préférence
supplémentaire exprimée par l'utilisateur.

## Éléments établis

- Dernière suite Python globale conservée : 1 009 réussites, 36 échecs le 11 octobre.
  Référence : `docs/research-improvement/2026-10-11-threshold-evidence/full-after.txt`.
- Contrôles du 16 octobre : typecheck frontend réussi, 13 tests frontend et 46
  tests de sélection réussis. Pas de nouveau passage global ni E2E réel.
- 7 attentes analytics et 1 test de migration sont figés sur v12 ; les fixtures
  créent v14. 24 échecs passent par le merger réellement limité à v12 ; une
  fixture de queue fournit cinq valeurs à neuf colonnes.
- Deux tests de service injectent encore `global_trainer` alors que les routes
  utilisent `PredictionService`, avec des compositions invalides.
- Le rapport de `train_pre_match_model.py` mesure la complémentarité de `winner`
  et la nomme anti-symétrie, malgré l'ajout du biais de côté.
- Le service calibré en mémoire sans manifeste peut franchir la sélection :
  son garde actuel ne vérifie le statut indépendant que si le manifeste existe.
  Le chargement HTTP normal impose déjà un manifeste ; cette observation ne
  prouve pas un contournement de ce chargement.
- Le rapport de validation de service est lié au checkpoint et au contrat,
  mais pas à la calibration ni à la politique de décision exacte.
- Le pilote du 11 octobre a évalué 1 344 matchs. Ils sont consommés, même si les
  manifests immuables conservent leur ancien champ `reserved`.
- Nombreux changements préexistants et une campagne de collecte du 16 octobre
  préparée dans le même dépôt : les traiter comme du travail à préserver.

## Contraintes

1. Aucune suppression, publication ou migration de base réelle pour réparer les
   tests. Sources/cibles de fusion en lecture seule ; sortie intermédiaire séparée.
2. Préserver les modifications, expériences et fichiers de collecte existants.
   Pas de reset, clean, stash global ni staging global du répertoire.
3. Schéma courant v14, version issue de `schema_version`, pas `PRAGMA user_version`.
4. Anti-symétrie et miroir neutre pour `draft_advantage` ; biais directionnel
   permis pour `winner` ; permutation intra-équipe pour les deux.
5. Toutes les routes de prédiction passent par `PredictionService`.
6. Absence de calibration ou de validation indépendante : abstention obligatoire.
   Checkpoint/manifeste corrompu : erreur de contrat, sans modèle de secours.
7. Pas de suppression de tests, xfail ou assouplissement d'invariants pour rendre
   artificiellement la suite verte. Adapter les attentes obsolètes au contrat
   documenté en conservant une assertion indépendante de comportement.
8. Aucun réglage, choix de seed, calibration ou seuil à partir d'un test évalué.
9. Aucun gain prédictif ni passage en production garanti par ce plan.

## Décisions de conception proposées

### Données

Supporter les couples source/cible v12/v12, v12/v13, v12/v14, v13/v13,
v13/v14 et v14/v14. La sortie garde le schéma de la cible. Rejeter une version
future, une structure incompatible ou une source plus récente que la cible.
L'outil ne migre aucune entrée.

En v14, préserver les identifiants de queue de la cible ; allouer de nouveaux
identifiants aux lignes source. Conflits par PUUID, sinon identité non ambiguë
nom/tag/région ; cible prioritaire. Pour une queue legacy sans région, accepter
`--legacy-queue-region` explicite ; sinon archiver la ligne avec sa raison.
Ne pas déduire la région du tag Riot.

Conserver exactement les embeddings spatiaux de la cible, BLOBs compris.
Archiver les embeddings de la source sans provenance d'encodeur vérifiable ;
ne pas les activer implicitement. Comptabiliser chaque ligne et conserver un
archivage réversible. Cette politique ne répare pas la provenance du 46D.

### Service et mesures

Un service sans manifeste validé doit s'abstenir, même avec des probabilités
calibrées. Une validation utilisable doit identifier checkpoint, calibration,
contrat/features, splits, cohorte et politique de décision. Toute modification
de l'un de ces éléments invalide son usage. Une demande de seuil différente
ne doit pas hériter automatiquement de la validation du seuil précédent.

Mesurer les invariants sur la partition de validation, avec une tolérance
`1e-5`, et nommer explicitement la tête évaluée. Un ancien checkpoint dépourvu
de tête draft doit indiquer ce diagnostic comme indisponible.

### Validation produit et ML

Conserver les tests UI rapides simulés et ajouter un parcours de la build React
servie par Flask, avec SQLite et petit checkpoint temporaires réels. Couvrir
succès calibré, abstention et artefact invalide sans appeler Riot.

Protéger cumulativement les identifiants des anciens tests, dont les 1 344 du
pilote. L'évaluateur réserve atomiquement l'usage de la nouvelle cohorte dans
ce registre avant sa première inférence test ; interruption ou échec ne la
rendent pas de nouveau indépendante. Préparer une nouvelle cohorte avec au moins 3 000 cibles de test jamais
évaluées et 3 000 de validation, en respectant les purges temporelles. Le nombre
3 000 conserve le protocole courant ; il ne constitue pas une garantie de puissance.
Si les données manquent, produire le déficit exact, sans réutiliser le pilote.

Protocole proposé pour confirmer 14D/18D : seeds 42 à 46, cinq époques chacune,
mêmes partitions et budget, choix/calibration sur validation seulement. Choisir
le candidat avant ouverture du test ; présenter aussi la variation entre seeds.
L'acceptation prédictive et l'acceptation d'une politique de décision sont
distinctes. Les objectifs de risque/couverture, la méthode et les critères
d'acceptation sont verrouillés avant la première évaluation test. S'ils sont
absents à cette date, le résultat reste descriptif et non actionnable ; fixer
ces objectifs après lecture exige une autre cohorte pour valider le service.

## Critères de clôture

La stabilisation logicielle est terminée lorsque les tests Python complets,
types/build frontend, tests unitaires, E2E simulés et E2E réel passent sur la
même révision ; les preuves sont conservées et les changements organisés.

Le volet ML est terminé lorsqu'un protocole verrouillé a produit un résultat
reproductible et une décision argumentée. Un résultat négatif ou inconclusif
est un résultat valide ; il n'autorise pas à déclarer un modèle prêt pour le service.

## Références

- `AGENTS.md`
- `docs/database_merge.md`
- `docs/model_serving/README.md`
- `docs/model_serving/aligned_14d_18d_pilot_20261011.md`
- `docs/research-improvement/2026-10-11-exact-selective-threshold.md`
- `docs/superpowers/plans/2026-10-16-project-stabilization.md`
