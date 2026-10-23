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

**Known limitation:** `scripts/merge_databases.py` still implements the schema-v12 contract only. It refuses v13/v14 inputs (`only schema version 12 is supported`), so current operational databases cannot be merged until the tool handles `player_spatial_embeddings` and the PUUID/region crawl queue. Its tests build v12 fixtures explicitly and assert that the current schema is refused.
