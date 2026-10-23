# Revue indépendante finale — gpt-6-astra, contexte neuf

Verdict : **With fixes**. Aucun problème Critical, un Important, aucun Minor étayé.

Points solides : séparation sélection/replay/matérialisation/chargement ; résultats strictement disponibles avant cutoff, créations ex aequo groupées et purges ; optimisations SQLite conservant la fenêtre globale dans une transaction ; histoire indépendante des labels ML ; trainers limités à train/validation ; hashes, chemins, formes, dtype et valeurs finies contrôlés. Preuves consultées : 144 tests, 148 458 comparaisons exactes et 20/20 audits SQLite. Pas de réexécution massive par le reviewer.

## Important confirmé : disponibilité 18D artificielle sans observations

Références : core/aligned_artifacts.py:60/71 ; dépendance core/features/timeline_metrics.py:102. Le writer déclare ready dès que l'extraction réussit. L'extracteur permissif transforme les participantFrames/mesures absentes en zéros ; le provider compte la timeline comme disponible.

Reproducer en mémoire : source_row('NA1_old', 10, 20), timeline_json égal à {"info":{"frames":[{}]}} ; enrichir target(source_row('NA1_current', 30, 40)) avec FrozenHistoryProvider, include_timeline=True, puis extract_aligned_inputs(..., 'nnriot.players18.v1'). Observation : timeline_source_available=True, timeline_matches=1, team_a.shape=(1,5,18), priors timeline=[0,0,0,0.083333...]. Une collecte partielle peut donc autoriser v4 sans mesures. Le dataset réel sans timelines reste correctement blocked_18d.

Recommandation : valider les observations requises avant comptage, garder la cause d'indisponibilité explicite et préserver les résultats Riot pour 14D. Tester frames:[{}], participant absent, mesure absente, refus 18D et disponibilité writer. Nouveau binding explicite si les sources hashées changent. L'hypothèse initiale d'une exception timeline supprimant 14D n'a pas été confirmée et n'est pas un second finding.

## Declined to judge — liste exhaustive du reviewer

- Performance prédictive, choix des hyperparamètres et qualité finale : aucun pilot réel ni protocole comparatif dans cette livraison.
- Acquisition/fusion de timelines récentes : chantier séparé, blocage explicite livré.
- Test jamais consulté dans toutes les recherches antérieures : impossible à déduire ; réservation limitée à la future expérience dans la documentation.
- Promotion/calibration du modèle servi : pas de promotion autorisée ; statuts unverified.
- Authenticité face au remplacement coordonné des fichiers et hashes du manifeste : SHA = intégrité liée au manifeste, pas signature de confiance ; système non présenté comme barrière de sécurité.

La réalisation suit les six tâches ; les adaptations documentées sont cohérentes. Corriger la disponibilité 18D avant d'accepter des timelines partielles. Revue unique, aucune seconde revue prévue.
