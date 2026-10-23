# Output semantics

Only the heads actually present in `manifest.output_heads` are exposed. `heads_raw` contains their unrounded raw checkpoint outputs (one batch row); calibration applies exclusively to `winner`. Auxiliary head quality and legacy checkpoint target-normalization provenance are unvalidated. A current source definition is not proof of the target transform used to train an undocumented historical checkpoint.

| Head | Current source shape per match | Meaning / current source units |
|---|---|---|
| `winner` | 2 | Index 0 Blue / team A; index 1 Red / team B. Winner label 0/1 verified from Riot team IDs and outcomes. |
| `draft_advantage` | 2 | Pure composition favorability, A/B order; raw, anti-symmetric, exactly 50/50 for mirror features. Absent/null for reference v3. |
| `winner_uncertainty` | 1 | Raw Dirichlet epistemic uncertainty; uncalibrated, absent/null on legacy checkpoints. |
| `team_b_kill_lead` | 2 | Current source index 1 means team B finishes with more kills; index 0 means A has at least as many kills, including ties. Uncalibrated. |
| `first_blood`, `first_baron`, `first_inhibitor`, `first_tower` | 3 | A first, B first, neither, in that order. |
| `first_to_5_kills`, `first_to_10_kills`, `first_to_15_kills`, `first_to_20_kills` | 3 | A first to threshold, B first, neither. Legacy target preparation defaults missing timeline labels to neither; this is a known limitation. |
| `kills_odd`, `both_baron`, `both_inhibitor`, `both_dragon`, `elder_dragon` | 1 | Sigmoid probability of the named event. Uncalibrated. |
| `kill_handicap` | 1 | Current source `(team_a_kills - team_b_kills) / 12`, mean 0 / standard deviation 12; positive favors A. |
| `total_kills` | 1 | Current target z-score with mean 30, standard deviation 12. |
| `team_a_kills`, `team_b_kills` | 1 each | Current target z-score with mean 15, standard deviation 7. |
| `total_barons` | 1 | Current target z-score with mean 1, standard deviation 1. |
| `total_dragons` | 1 | Current target z-score with mean 2.5, standard deviation 1.5. |
| `total_towers` | 1 | Current target z-score with mean 10, standard deviation 5. |

The service does not silently apply these current regression transforms to an unknown historical training run. The winner comparison uses the independently verified team perspective only. Counterfactual credits remain raw leave-one-player-out effects, not additive Shapley values or causal effects; their legacy `draft_delta` key is a raw winner margin including side bias and is not the separate `draft_advantage` head.
