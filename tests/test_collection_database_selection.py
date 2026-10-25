"""Collection writes and queued web jobs must use their selected source database."""
import sqlite3

import pytest


@pytest.mark.parametrize("collector", ["seed", "top", "puuid", "batch"])
@pytest.mark.parametrize("selection", ["explicit", "environment"])
def test_collectors_write_only_selected_database(tmp_path, monkeypatch, collector, selection):
    from core import database
    from scripts import data_collector as module

    default = tmp_path / "default.db"
    selected = tmp_path / "selected.db"
    database.init_db(str(default))
    database.init_db(str(selected))
    monkeypatch.setattr(database, "DB_PATH", str(default))
    monkeypatch.setenv("NNRIOT_DB_PATH", str(selected if selection == "environment" else default))
    kwargs = {"db_path": str(selected)} if selection == "explicit" else {}
    payload = {"info": {"gameMode": "CLASSIC", "gameCreation": 1000,
        "gameDuration": 1200, "gameVersion": "16.1.1",
        "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
        "participants": [{"puuid": f"p{i}", "teamId": 100 if i < 5 else 200,
            "championName": "Ahri", "teamPosition": "TOP", "win": i < 5,
            "kills": 1, "deaths": 1, "assists": 1, "goldEarned": 1000} for i in range(10)]}}

    class API:
        def __init__(self, region): pass
        def get_puuid(self, *args): return "p0"
        def get_account_by_puuid(self, *args): return {"game_name": "Player", "tag_line": "EUW"}
        def get_top_players(self, **kwargs): return [{"puuid": "p0"}]
        def get_match_ids(self, *args, **kwargs): return [f"EUW1_{i}" for i in range(6)]
        def get_match_details_batch(self, ids, **kwargs): return {mid: payload for mid in ids}
        def get_match_timelines_batch(self, ids, **kwargs): return {mid: {"info": {}} for mid in ids}

    monkeypatch.setattr(module.riot_api, "RiotAPI", API)
    monkeypatch.setattr(module, "FETCH_TIMELINES", True)
    class CheckedLock:
        """Fail immediately instead of hanging if batch logging recursively locks."""
        held = False
        def __enter__(self):
            assert not self.held, "batch lock acquired recursively"
            self.held = True
        def __exit__(self, *args): self.held = False
    monkeypatch.setattr(module, "batch_lock", CheckedLock())
    if collector == "seed": module.collect_training_data([("Player", "EUW")], **kwargs)
    elif collector == "top": module.collect_top_leagues_data(regions=["EUW"], **kwargs)
    elif collector == "puuid": module.collect_by_puuid([{"puuid": "p0", "server": "EUW"}], **kwargs)
    else: module.collect_batch_with_smurfs([("Player", "EUW")], [], **kwargs)
    stored = ("matches", "training_dataset", "match_timelines")
    # Top-league anchors only discover the frontier: their matches predate their
    # selection on rank, so nothing is stored, but the anchor and its frontier are.
    written = ("players", "crawl_queue") if collector == "top" else ("players", *stored)
    for table in ("players", "crawl_queue", *stored):
        with sqlite3.connect(default) as conn:
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
        with sqlite3.connect(selected) as conn:
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        if table in written:
            assert count > 0
        elif collector == "top":
            assert count == 0


@pytest.mark.parametrize("route", ["seed", "players", "batch", "lolpros", "warmup", "spectator"])
def test_web_collection_jobs_capture_selected_database(tmp_path, monkeypatch, route):
    from app import create_app, core
    from app.routes import api, history, lolpros
    from core import database
    from core.task_runner import global_task_runner
    from scripts import data_collector

    selected = str(tmp_path / "selected.db")
    other = str(tmp_path / "other.db")
    database.init_db(other)
    monkeypatch.setenv("NNRIOT_DB_PATH", other)
    application = create_app({"TESTING": True, "NNRIOT_DB_PATH": selected})
    monkeypatch.setitem(core.collection_status, "status", "idle")
    monkeypatch.setitem(data_collector.batch_status, "status", "idle")
    monkeypatch.setattr(core, "tf_available", False)
    jobs, calls = [], []
    def enqueue(kind, fn, *args, **kwargs):
        jobs.append((fn, args, kwargs))
        return "job-test"
    def collect(*args, **kwargs):
        calls.append(kwargs)
        return {}
    monkeypatch.setattr(global_task_runner, "submit_task", enqueue)
    for name in ("collect_training_data", "collect_by_puuid", "collect_batch_with_smurfs"):
        monkeypatch.setattr(data_collector, name, collect)
    from core.collection import CollectionService
    def service_collect(self, *args, **kwargs):
        calls.append({"db_path": self.db_path, **kwargs})
        return {}
    for name in ("collect_seed_players", "collect_by_puuid_accounts", "collect_batch_with_smurfs"):
        monkeypatch.setattr(CollectionService, name, service_collect)
    client = application.test_client()
    if route == "seed":
        response = client.post("/api/history/collect", json={"game_name": "Player", "tag_line": "EUW"})
    elif route == "players":
        response = client.post("/api/history/collect_batch", json={"players": [{"game_name": "Player", "tag_line": "EUW"}]})
    elif route == "batch":
        response = client.post("/api/batch/collect", json={"players": "Player#EUW"})
    elif route == "lolpros":
        response = client.post("/api/lolpros/import", json={"player": {"name": "Pro", "league_player": {"accounts": [
            {"encrypted_puuid": "main", "gamename": "Main", "tagline": "EUW", "server": "EUW", "rank": {"score": 2}},
            {"encrypted_puuid": "alias", "gamename": "Alias", "tagline": "EUW", "server": "EUW", "rank": {"score": 1}},
        ]}}})
        with sqlite3.connect(selected) as conn:
            assert conn.execute("SELECT count(*) FROM players").fetchone()[0] == 2
            assert conn.execute("SELECT * FROM player_aliases").fetchall() == [("alias", "main")]
    elif route == "warmup":
        database.save_player("stale", "Player", "EUW", db_path=selected)
        monkeypatch.setattr(database, "get_stale_puuids", lambda *args, **kwargs: ["stale"])
        response = client.post("/api/predict/custom", json={"blue_team": [{"puuid": "stale"}]*5, "red_team": [{"puuid": "stale"}]*5})
    else:
        monkeypatch.setattr(api.live_game, "fetch_active_game_by_summoner", lambda *args: {"participants": [{"puuid": "new", "summonerName": "Player"}]})
        monkeypatch.setattr(api.live_game, "resolve_puuid_for_participants", lambda region, players: players)
        response = client.get("/api/spectator/active?name=Player&tag=EUW")
    assert response.status_code == (503 if route == "warmup" else 200), response.get_json()
    assert len(jobs) == 1
    # Execute after request teardown and a subsequent app reconfiguration: job ownership is fixed.
    application.config["NNRIOT_DB_PATH"] = other
    fn, args, kwargs = jobs[0]
    fn(*args, **kwargs)
    assert calls and calls[0]["db_path"] == selected
    with sqlite3.connect(other) as conn:
        assert conn.execute("SELECT count(*) FROM players").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM player_aliases").fetchone()[0] == 0


def test_repository_defaults_are_resolved_at_construction(tmp_path, monkeypatch):
    from core import config, database
    from core.repository import MatchRepository
    root = tmp_path / "project"
    root.mkdir()
    selected = root / "selected.db"
    database.init_db(str(selected))
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setenv("NNRIOT_DB_PATH", "selected.db")
    monkeypatch.chdir(tmp_path)
    assert MatchRepository().db_path == str(selected)
    assert MatchRepository("selected.db").db_path == str(selected)
    with pytest.raises(FileNotFoundError):
        MatchRepository("missing.db")
    assert not (root / "missing.db").exists()


@pytest.mark.parametrize("collector", ["collect_training_data", "collect_top_leagues_data", "collect_by_puuid", "collect_batch_with_smurfs"])
def test_collector_refuses_missing_database(tmp_path, collector):
    from scripts import data_collector
    missing = tmp_path / "missing.db"
    kwargs = {"db_path": str(missing)}
    if collector == "collect_top_leagues_data": kwargs["regions"] = []
    elif collector == "collect_batch_with_smurfs": kwargs.update(player_list=[], sources=[])
    elif collector == "collect_training_data": kwargs["seed_players"] = []
    else: kwargs["accounts"] = []
    with pytest.raises(FileNotFoundError):
        getattr(data_collector, collector)(**kwargs)
    assert not missing.exists()
