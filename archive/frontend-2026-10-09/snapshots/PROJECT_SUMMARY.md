# NNriot project summary

NNriot is a League of Legends data and prediction application. Riot match collection writes to a configurable SQLite source. The Flask app serves player history, prediction routes, and retrospective analytics, while a React and TypeScript frontend is built by the root Vite configuration into `static/dist`. Server-rendered Jinja pages remain in `templates/`.

The operational schema is version 12. `matches` retains compressed Riot payloads, `match_participants` supports indexed player history, and `training_dataset` stores features and labels. Player names and aliases live in separate tables. `analytics/` reads the operational source and builds a separate, rebuildable SQLite evidence mart. Analytics reports post-match associations with traceable evidence; it does not establish causality or pre-match model quality.

`python cli.py train` runs candidate pre-match training from the configured source. The candidate uses draft and game-context inputs, chronological train/validation/test partitions, and a saved split manifest and metrics report. It does not automatically replace the active application checkpoint. The older continuous loop remains available as `python cli.py train-continuous`. See [docs/pre_match_training.md](docs/pre_match_training.md) for the protocol and its limits.

The database merge tool uses explicit source, target, staged output, and report paths. It preserves the newer target's conflicting records and archives excluded or conflicting old data before old databases are removed. The [merge guide](docs/database_merge.md) covers validation and publication. Runtime paths come from `NNRIOT_DB_PATH` and `NNRIOT_ANALYTICS_DB_PATH`; the example file under `runs/` is a local deployment choice.

Install Python dependencies with `pip install -r requirements.txt`, install the frontend from the root with `npm ci`, then run `npm run build`. Start Flask with `python cli.py web`. Tests live in `tests/` and `e2e/`.
