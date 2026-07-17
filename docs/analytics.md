# Analytics explicables et chatbot français

Ce module décrit des associations observées dans l’état final des matchs. Il ne
démontre pas de causalité et ne prédit pas, avant le match, la manière de gagner.
Chaque nombre publié par le chatbot est relié à un `evidence_id`, à une version
de données, à une cohorte et à une méthode statistique reproductible.

## Démarrage

Les commandes suivantes utilisent `training_data.db` comme source opérationnelle
et le mart séparé `analytics.db` :

```powershell
python -m analytics --source training_data.db --analytics analytics.db build
python -m analytics --source training_data.db --analytics analytics.db update
python -m analytics --source training_data.db --analytics analytics.db health
python -m analytics --analytics analytics.db train-model --artifacts analytics_artifacts
python -m analytics --source training_data.db --analytics analytics.db benchmark --iterations 30
python run.py
```

L’interface est alors disponible sur `http://127.0.0.1:5000/analytics`. Les
variables correspondantes sont `NNRIOT_DB_PATH`, `NNRIOT_ANALYTICS_DB_PATH` et
`NNRIOT_ANALYTICS_SEED`.

`build` reconstruit le mart dans un fichier temporaire puis le publie de manière
atomique. `update` ajoute les nouveaux matchs dans une transaction et invalide
le cache si la version des données change. Une source qui évolue pendant une
construction fait échouer l’opération. La connexion source est toujours ouverte
en mode SQLite `mode=ro` et `query_only=ON`; aucune commande Analytics n’écrit
dans `training_data.db`.

## Données et traçabilité

Le schéma version 1 contient :

- `match_facts`, une ligne par match classique accepté ;
- `team_facts`, exactement deux lignes par match ;
- `player_facts`, exactement dix lignes par match ;
- `extraction_issues`, les exclusions explicables sans ligne partielle ;
- `evidence_cache`, les paquets de preuve complets et versionnés ;
- `analytics_meta`, la version, les compteurs, le rapport qualité et les versions
  d’extracteur et de méthode.

La `dataset_version` est un SHA-256 déterministe de la version d’extracteur et
des identifiants/horodatages des matchs CLASSIC éligibles. Le jeton de fraîcheur
à faible coût compare la version de schéma source, le nombre de matchs et
l’horodatage maximal. Une requête est refusée avec `ANALYTICS_DB_STALE` si ce
jeton ne correspond plus au mart.

Les métriques autorisées sont : éliminations, morts, assistances, or, dégâts aux
champions, vision, sbires de voie, monstres de jungle, barons, dragons, tours,
inhibiteurs, hérauts, Atakhan, premier sang, premier baron, premier dragon,
première tour et premier inhibiteur. Les filtres sont paramétrés ; aucune surface
SQL arbitraire n’est exposée.

Un paquet de preuve contient la question, la requête structurée, la cohorte, les
effectifs, l’estimation, l’intervalle à 95 %, les valeurs p et q, la taille
d’effet, la méthode, la graine, les replis temporels, les limites, la version du
dataset et le hash des identifiants de matchs.

## Questions françaises prises en charge

Le planificateur déterministe comprend :

- le classement des conditions de victoire ;
- l’explication d’une métrique ;
- la comparaison d’une métrique entre deux régions ;
- la comparaison d’une métrique entre deux patchs ;
- les relances demandant les preuves ou les limites de l’analyse précédente.

Exemples :

```text
Comment gagne-t-on le plus souvent ?
La vision est-elle associée à la victoire en EUW ?
Existe-t-il une différence robuste de vision entre EUW et KR ?
La vision diffère-t-elle entre les patchs 16.13 et 16.14 ?
Quelles sont les preuves ?
Quelles sont les limites ?
```

Une comparaison inter-région doit nommer une métrique ; sinon la réponse est une
abstention `UNSUPPORTED_FILTER` avec `required: metric`. Le contexte de relance
accepté a cette forme et aucun autre champ n’est autorisé :

```json
{
  "intent": "explain_metric",
  "metric": "vision_score",
  "filters": {
    "regions": ["EUW"],
    "patches": [],
    "queue_ids": [420],
    "date_start_ms": null,
    "date_end_ms": null,
    "champion": null,
    "role": null
  }
}
```

API :

- `POST /api/analytics/chat` avec `message`, `context` et `filters` ;
- `GET /api/analytics/health` ;
- `GET /api/analytics/evidence/<evidence_id>`.

## Seuils de publication

La cohorte commence au patch le plus récent et peut s’élargir à trois patchs au
maximum. Une analyse exige au moins 500 matchs et 100 joueurs distincts. Une
comparaison binaire exige au moins 100 événements dans chaque groupe. Les
calculs utilisent une graine fixe (`20260716` par défaut), 2 000 rééchantillonnages
et la correction de Benjamini-Hochberg lorsque plusieurs métriques sont testées.

Une association « modérée » n’est autorisée que si toutes les conditions sont
réunies :

- intervalle de confiance à 95 % excluant zéro et q ≤ 0,05 ;
- au moins 500 matchs, 100 joueurs et 100 observations par groupe ;
- effet pratique ≥ 0,20 écart-type, ou différence de taux de victoire ≥ 3 points ;
- direction cohérente dans au moins deux des trois replis temporels.

La qualification « forte » exige en plus au moins 2 000 matchs, q ≤ 0,01, un
effet ≥ 0,50 écart-type ou 5 points de victoire, et la même direction dans les
trois replis. Sinon les nombres peuvent rester descriptifs, mais aucune
recommandation de gameplay n’est publiée.

Une comparaison inter-région est standardisée par patch. Elle exige au moins
500 matchs dans chaque région, une couverture commune des patchs, les mêmes
portes statistique et pratique, et une direction récente identique à la
direction globale. Pour une métrique binaire, chaque région doit aussi contenir
au moins 100 expositions/non-expositions et 100 victoires/défaites.

## Modèle interprétable et réseau de neurones

Le modèle logistique est rétrospectif : il apprend sur les différences finales
bleu moins rouge, avec contrôles région/patch et trois interactions fixées. La
séparation principale est chronologique 70/15/15 ; `C` est choisi uniquement sur
la validation, puis le test final produit log-loss, score de Brier, erreur de
calibration, ROC-AUC, intervalles bootstrap, coefficients nommés et odds ratios.
La sensibilité « joueurs disjoints » ne possède aucun repli permissif : si elle
est sous-alimentée, elle retourne `INSUFFICIENT_MATCHES`.

Le réseau de neurones existant sert uniquement à proposer des candidats de
découverte. Une proposition NN ne peut être formulée dans le chatbot qu’après
validation par la chaîne statistique et attribution d’un `evidence_id`. Le NN
ne constitue donc jamais, seul, une preuve ni une explication causale.

## Refus et limites connues

- Pas de timelines : les questions à 5, 10 ou 15 minutes répondent
  `TIMELINE_DATA_UNAVAILABLE`.
- Pas d’identité stable d’équipes esport : les profils T1, G2, Fnatic, etc.
  répondent `ESPORT_TEAM_IDENTITY_UNAVAILABLE`.
- Pas d’affirmation causale, de garantie, de pari ou de requête SQL arbitraire.
- Les données décrivent la sélection de matchs collectés, pas l’ensemble des
  joueurs de League of Legends. Le biais d’échantillonnage doit être conservé
  dans l’interprétation.
- Le rang n’est pas disponible dans le corpus analytique actuel ; aucun profil
  par niveau ne doit être inféré.
- Les variables sont des états finaux et peuvent contenir la conséquence de la
  victoire. Elles ne doivent jamais être présentées comme des caractéristiques
  pré-match ni comme une stratégie garantie.

## Mesures du 17 juillet 2026

Le corpus local disponible lors de la vérification contenait une ligne CLASSIC,
mais son `raw_json` était nul. Résultat réel du build : 1 match éligible, 0
accepté, `INVALID_RAW_JSON: 1`, 0 échec inexpliqué, porte qualité validée,
dataset `825cb1403b6f962a7a5c08aca050c71f9a7ad76efe7b02d960495dbe6fd09668`.
Le SHA-256 de la source avant/après est resté
`0F2CC7C0F5959BE8D1FF87219B1A363A0343CF21D3C08C33EFF9F283D3F3B0E0`.

Le modèle réel s’est abstenu : train/validation/test = 0/0/0,
`final_test=INSUFFICIENT_MATCHES`, contrôle joueurs disjoints indisponible et
aucun artefact publié. Le benchmark réel s’est également abstenu avec
`INSUFFICIENT_MATCHES`.

Un benchmark d’intégration de 30 itérations a donc été mesuré sur 600 matchs
synthétiques passés par le même décodeur, extracteur, mart, moteur et service :

| Mesure | p95 mesuré | Seuil |
| --- | ---: | ---: |
| Preuve en cache | 28,82 ms | < 100 ms |
| Analyse d’une métrique sans lecture cache | 138,75 ms | < 2 000 ms |
| Chat déterministe sans LLM externe | 92,46 ms | < 3 000 ms |

Contexte : Windows 11 `10.0.26200`, Python 3.13.9. Les trois seuils sont
respectés ; ces latences synthétiques ne remplacent pas une mesure future sur un
corpus réel suffisamment alimenté.
