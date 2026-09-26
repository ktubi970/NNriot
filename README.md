# NNriot

NNriot collects League of Legends matches into SQLite, serves a Flask API and React interface, trains candidate pre-match models, and builds a separate retrospective analytics database. The source database is configured with `NNRIOT_DB_PATH`; the analytics database uses `NNRIOT_ANALYTICS_DB_PATH` (or `ANALYTICS_DB_PATH`). Both paths can point anywhere. The local collection database under `runs/` is an example, not a required filename.

## Setup

Use Python 3.10 or newer and Node.js with npm. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
npm run build
```

Copy `.env.example` to `.env`, set `RIOT_API_KEY` when using Riot collection, and set the database paths for your deployment. Keep database files and credentials out of version control. The root `package.json` is the only frontend package manifest; the build writes to `static/dist`.

Database selection is explicit command argument, process environment, root `.env`, then the default. Relative database paths are resolved from the project root. A configured source that is missing produces an error. For a new installation only, create an empty source intentionally with `python -c "from core import config, database; database.init_db(config.get_db_path())"`; existing installations should select their collected database instead.

## Run

```powershell
python cli.py web
python cli.py train --samples 5000 --epochs 5 --batch-size 64
python cli.py train-continuous
python cli.py backfill
```

`train` reads the configured source database unless `--db` is supplied. It produces a separate candidate checkpoint, split manifest, and metrics report. It does not replace the model currently loaded by the web app. `train-continuous` is the historical continuous training loop. `backfill` repairs missing training labels; pass `--db` to target another database. On Windows, `start.bat` forwards its arguments to the CLI.

For an explicit source and output, use `python cli.py train --db path/to/source.db --output runs/pilot/model.keras`. The dedicated `scripts/train_pre_match_model.py` also exists and requires `--db`; its [training guide](docs/pre_match_training.md) explains eligibility, chronological splits, leakage controls, and how to read the report. Candidate accuracy must be evaluated against the reported baseline; the project does not claim a fixed pre-match accuracy.

## Data and analytics

The operational SQLite schema is version 12. Player history reads indexed `match_participants`; historical raw matches can be repaired with `scripts/backfill_participants.py`. The analytics mart is derived from the configured source and can be rebuilt:

```powershell
python -m analytics build
python -m analytics health
```

The [analytics guide](docs/analytics.md) describes supported retrospective questions and limitations. Analytics describes observed post-match associations; it is not the input to a pre-match prediction. To consolidate an older source into a newer collection, follow the [database merge guide](docs/database_merge.md). The merge stages a separate output and report; publishing that output and removing obsolete databases are explicit later steps.

## Project layout

- `core/`: SQLite schema, repositories, feature extraction, model construction, and training helpers.
- `scripts/`: collection, candidate training, continuous training, backfills, and database merge tooling.
- `app/`: Flask routes and services; `templates/` retains server-rendered pages.
- `frontend/`: React and TypeScript UI built with root Vite, PostCSS, and Tailwind configs.
- `analytics/`: rebuildable evidence mart and French analytics interface.
- `tests/` and `e2e/`: Python and browser checks.

Run `python -m pytest` for Python tests; the test bootstrap selects temporary databases and a temporary checkpoint before importing the application. `NNRIOT_CHECKPOINT_PATH` can explicitly select the continuous trainer checkpoint; candidates keep their separate `--output` path. Run `npm run test:e2e` for browser tests. The frontend development server is `npm run dev` and proxies API requests to Flask on port 5000.

## Contributing

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) and [.github/SECURITY.md](.github/SECURITY.md). This project is licensed under [MIT](LICENSE).
