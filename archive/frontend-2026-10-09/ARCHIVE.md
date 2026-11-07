# Frontend React/Vite archivé — 9 octobre 2026

## Reprise le 9 octobre 2026

Les 44 fichiers du manifeste ont été recopiés à leurs chemins actifs après
vérification des SHA-256 et de l'absence de conflits aux destinations. Le cache
`node_modules/` a été déplacé à la racine du projet ; les sources, le manifeste,
les snapshots et les anciens rapports restent conservés ici. Les commandes
`build`, `dev` et `test` du lanceur ont été réactivées. Les sections suivantes
décrivent l'état historique lors de l'archivage.

La copie initiale conservait les défauts connus du frontend. La refonte validée
ensuite a corrigé les erreurs TypeScript, supprimé les résultats et statistiques
de démonstration, et intégré le contrat de calibration/abstention. Le cockpit
actif est désormais responsive, avec une draft vide au départ, des dialogues
accessibles au clavier et la prise en compte de la réduction des animations.
Les imports live/OCR sont signalés comme indisponibles. Les sources archivées
restent intactes pour documenter l'état antérieur.

Vérifications de la copie initiale, avant refonte :

- `npm.cmd run build` : réussi (Vite, 1 610 modules). Le premier essai a été
  bloqué par `EPERM` dans le bac à sable Windows ; le build local autorisé a réussi.
- `pytest tests/test_config_cli.py tests/test_predictor_ui.py tests/test_final_web_app.py -q` :
  58 tests passés, avec les répertoires temporaires isolés dans `scratch/`.
- Client de test Flask sur une base temporaire, sans chargement de trainer :
  `/` sert le cockpit React, les deux assets compilés répondent 200 et
  `/dashboard` sert toujours la page Jinja.
- `playwright test --list --project=chromium` : 21 tests découverts ; la suite
  navigateur n'a pas été exécutée. L'ouverture de la prévisualisation isolée
  n'a pas abouti dans le bac à sable Windows.
- `tsc --noEmit -p frontend/tsconfig.json` : échec sur les erreurs archivées
  (imports de types manquants, props d'icônes et éléments inutilisés).

Vérifications après refonte :

- Build Vite réussi (1 611 modules), contrôles TypeScript du frontend et des
  configurations navigateur réussis.
- `npm run test:unit` : 13 tests du contrat API passés.
- `npm run test:e2e` : 24 tests passés sur desktop, tablette et mobile avec
  réduction des animations. Cette suite utilise Vite et des API simulées ;
  elle ne démarre ni Flask, ni collecte, ni entraînement.
- Les 58 tests Python de configuration et d'interface Flask passent.
- Captures de l'interface dans `output/playwright/` et rapport navigateur dans
  `playwright-report/` à la racine active.

La commande `npm run test:e2e:legacy` conserve les anciens tests en accès
explicite. Ils incluent une action d'entraînement manuel et un script de
captures avec un chemin externe ; ils ne font plus partie de la suite par défaut.

Le frontend React/Vite est mis de côté temporairement pour concentrer le travail
sur la collecte, les données, les modèles et le backend Flask. Cette archive
conserve les chemins relatifs d'origine pour permettre sa restauration.

## Contenu conservé

- `frontend/` : sources React, TypeScript, composants, styles et configuration TypeScript.
- `package.json`, `package-lock.json` : dépendances npm et versions verrouillées.
- `vite.config.ts`, `postcss.config.cjs`, `tailwind.config.cjs` : configuration du build.
- `e2e/`, `playwright.config.ts`, `tsconfig.json` : tests navigateur et configuration.
- `.github/workflows/playwright.yml` : ancien workflow, hors du dossier actif de GitHub Actions.
- `static/dist/` : build présent au moment de l'archivage, conservé localement.
- `node_modules/`, `playwright-report/`, `test-results/` : caches et résultats locaux, ignorés par Git.
- `snapshots/` : copies du README, du résumé du projet et du lanceur avant archivage.
- `archive-manifest.json` : HEAD de référence, liste des déplacements et empreintes SHA-256.

Les 44 fichiers de source, configuration, tests et build recensés dans le
manifeste ont été vérifiés après déplacement. Les caches npm et les rapports
générés ne sont pas inclus dans ces empreintes.

## État au moment de l'archivage

Le build Vite avait réussi lors de l'état des lieux. Le contrôle TypeScript
échouait sur des imports de types manquants, des props incompatibles et des
éléments inutilisés. Ces erreurs sont conservées pour la reprise du frontend.

Le backend Flask, ses API, les pages Jinja, les données et les modèles restent
dans leurs emplacements actifs. Le retrait de `static/dist/` fait revenir la
route `/` aux pages Jinja existantes. Le lanceur actif propose les commandes
Python et signale l'archivage pour les anciennes commandes `build`, `dev` et `test`.

Après archivage, les 58 tests de `tests/test_final_web_app.py`,
`tests/test_predictor_ui.py` et `tests/test_config_cli.py` passent. Ils vérifient
les pages Flask/Jinja, les API et la configuration CLI conservées.

## Restauration

1. Vérifier les changements effectués depuis l'archivage et l'absence de conflits
   aux emplacements de destination.
2. Remettre les entrées de `Entries` dans `archive-manifest.json` à la racine du
   projet, en conservant leurs chemins relatifs. Par exemple, `frontend/` revient
   à `frontend/` et `.github/workflows/playwright.yml` retrouve son emplacement
   dans le dossier actif de GitHub Actions. Les caches et rapports sont facultatifs.
3. Réintégrer les commandes frontend du lanceur à partir de `snapshots/start.bat`,
   en préservant les évolutions du lanceur réalisées entre-temps. Actualiser la
   documentation active à partir du contexte conservé dans `snapshots/`.
4. Depuis la racine du projet, exécuter `npm ci`, puis corriger les erreurs
   TypeScript et vérifier `npm run build` avant de remettre l'interface en service.
5. Reprendre les tests Playwright avec `npm run test:e2e` lorsque le backend et le
   frontend restaurés sont prêts.

Ne pas écraser des fichiers créés depuis l'archivage lors de la restauration.
