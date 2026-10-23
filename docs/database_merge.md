# Consolidating SQLite databases

The merge command combines a historical source with a newer operational database into a **separate staged output**. It never treats a merge report as permission to replace the configured database or delete an original file. Stop collectors, trainers, and the web app before publication. Use explicit paths so the result does not depend on a process's environment or import order.

```powershell
python -m scripts.merge_databases `
  --source path/to/older.db `
  --target path/to/current.db `
  --output path/to/staged.db `
  --report path/to/merge-report.json `
  --dry-run
```

Review the dry-run JSON printed to the terminal, then run the same command without `--dry-run` to create the staged database and report. Put the output on the same volume as the target when you intend to publish it. A local example of a target is `runs/training_data_23h_20260922_191434.db`; choose your own actual path. Use fresh output and report names for a rerun.

The target wins where keys conflict. Only validated `CLASSIC` matches from the old source enter the staged operational tables. Incoming participants, training features, and labels are reconstructed or validated; the old `is_trained` flag is not transferred. Player identities and aliases are retained, and derived player statistics are recalculated. Excluded, corrupt, and conflicting old rows and their related data are retained in a local gzip JSONL archive recorded by the report. The archive is part of the retention set when removing the old SQLite file. No candidate model is promoted by the database merge.

Before publishing, check the report's accepted and archived totals, SQLite integrity and foreign-key results, representative player history, and the canonical match and training rows that were supposed to win conflicts. Confirm the staged database and archive are readable and the merge report names the intended source and target. Rebuild and verify the analytics mart against the staged source; it is derived data, separate from the operational database.

Publication is an explicit maintenance step: close every SQLite connection, checkpoint WAL content, retain a rollback copy of the current operational database, then switch `NNRIOT_DB_PATH` to the validated staged result or replace the configured file. Keep the SQLite main file and its `-wal` and `-shm` sidecars together until connections are closed and checkpointed. Point `NNRIOT_ANALYTICS_DB_PATH` at the rebuilt analytics mart. Smoke test the web app, history, training command, and analytics health with the configured paths.

Delete only the specifically obsolete source and derived analytics files after those checks pass and the gzip JSONL archive and report are retained. Include their matching SQLite sidecars once safely checkpointed. Keep unrelated databases in worktrees, scratch directories, and other runs out of the cleanup. The merge tool itself does not publish or delete any database.

## Current schema and version verification

The application schema is v14. Read `SELECT MAX(version) FROM schema_version`; `PRAGMA user_version` is not the application migration ledger and can remain 0. v12 introduced indexed match participants, v13 added `player_spatial_embeddings`, and v14 migrated the crawl queue to PUUID/region records. `core/database.py` applies ordered migrations. The 9 October serving/evaluation work opened the source read-only and made no schema changes. Keep the staged merge, backup, validation and publication protocol above for future schema/data changes.

## Schema v14 merge contract

`scripts/merge_databases.py` accepts only schema v14 inputs on both sides and refuses anything else (`only schema version 14 is supported`). To merge an older database, migrate a **copy** first, for example `python -c "from core.database import init_db; init_db('path/to/copy.db')"`, and never migrate the original. A test pins the tool's `SCHEMA_VERSION` to `core.database.SCHEMA_VERSION`, so a future migration fails the suite until the merger gets an explicit policy for it.

- **Crawl queue:** target rows and their ids are kept unchanged. A source entry matches on PUUID, or on Riot ID plus region for legacy rows without a PUUID. Conflicting entries (different priority, processed state or region) keep the target's version and archive the source row as `canonical_queue_precedence`. New valid entries are inserted with fresh AUTOINCREMENT ids, so they come after target entries of equal priority, and the original row is archived as `regenerated_queue_id` when its id changes. Entries with a synthetic PUUID, an unknown region, a non-integer priority, a processed flag outside 0/1, or no PUUID and no Riot ID are archived as `invalid_queue_entry`.
- **Spatial embeddings:** source rows are never imported. `scripts/compute_spatial_embeddings.py` builds the trajectory encoder from random weights on every run, so embeddings from two databases do not share a space. Target embeddings are preserved, identical source rows are skipped, other source rows are archived as `derived_embedding_not_merged`, and orphans as `orphan_relationship`. Matches imported from the source therefore have no embeddings in the staged output. Recomputing them with a new encoder run would mix spaces again until the encoder weights are fixed and recorded.
- **Preservation check:** the report compares the canonical queue by `id` and embeddings by `(match_id, puuid)`, in addition to the existing tables.
