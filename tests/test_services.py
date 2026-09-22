import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch

from app.services.player_service import PlayerService
from app.services.system_service import SystemService


class TestPlayerService:
    @pytest.fixture
    def mock_repo(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_repo):
        return PlayerService(repo=mock_repo)

    def test_get_all_aliases_returns_data_directly(self, service, mock_repo):
        mock_repo.get_all_aliases.return_value = [{"canonical_puuid": "c1", "alias_puuid": "a1"}]
        result = service.get_all_aliases()
        # Must return list directly, NOT a (result, 200) tuple
        assert not isinstance(result, tuple)
        assert result == [{"canonical_puuid": "c1", "alias_puuid": "a1"}]
        mock_repo.get_all_aliases.assert_called_once()

    def test_unlink_accounts_success(self, service, mock_repo):
        result = service.unlink_accounts("alias-123")
        assert not isinstance(result, tuple)
        assert result == {"success": True}
        mock_repo.unlink_accounts.assert_called_once_with("alias-123")

    def test_unlink_accounts_missing_puuid_raises_value_error(self, service, mock_repo):
        with pytest.raises(ValueError, match="alias_puuid is required"):
            service.unlink_accounts("")

        with pytest.raises(ValueError, match="alias_puuid is required"):
            service.unlink_accounts(None)

        mock_repo.unlink_accounts.assert_not_called()

    def test_search_players_short_or_empty_query(self, service, mock_repo):
        assert service.search_players("") == []
        assert service.search_players("a") == []
        assert service.search_players(None) == []
        mock_repo.search_players.assert_not_called()

    def test_search_players_valid_query(self, service, mock_repo):
        mock_repo.search_players.return_value = [{"game_name": "Faker", "tag_line": "KR1"}]
        result = service.search_players("Faker", limit=5)
        assert not isinstance(result, tuple)
        assert result == [{"game_name": "Faker", "tag_line": "KR1"}]
        mock_repo.search_players.assert_called_once_with("Faker", limit=5)

    def test_get_random_players(self, service, mock_repo):
        players = [{"game_name": f"P{i}", "tag_line": "EUW"} for i in range(25)]
        mock_repo.get_top_players.return_value = players
        result = service.get_random_players(limit=5)
        assert not isinstance(result, tuple)
        assert len(result) == 5
        mock_repo.get_top_players.assert_called_once()

    def test_get_top_players(self, service, mock_repo):
        mock_repo.get_top_players.return_value = [{"game_name": "Chovy", "tag_line": "KR1"}]
        result = service.get_top_players(limit=10)
        assert not isinstance(result, tuple)
        assert result == [{"game_name": "Chovy", "tag_line": "KR1"}]
        mock_repo.get_top_players.assert_called_once_with(limit=10)

    def test_link_accounts_success(self, service, mock_repo):
        result = service.link_accounts("canon-1", "alias-1")
        assert not isinstance(result, tuple)
        assert result == {"success": True, "message": "Accounts completely linked."}
        mock_repo.link_accounts.assert_called_once_with("canon-1", "alias-1")

    def test_link_accounts_missing_arguments_raises_value_error(self, service, mock_repo):
        with pytest.raises(ValueError, match="canonical_puuid and alias_puuid are required"):
            service.link_accounts("", "alias-1")

        with pytest.raises(ValueError, match="canonical_puuid and alias_puuid are required"):
            service.link_accounts("canon-1", "")

        with pytest.raises(ValueError, match="canonical_puuid and alias_puuid are required"):
            service.link_accounts(None, None)

        mock_repo.link_accounts.assert_not_called()


class TestSystemService:
    @pytest.fixture
    def mock_repo(self, tmp_path):
        repo = MagicMock()
        repo.db_path = str(tmp_path / "test.db")
        return repo

    @pytest.fixture
    def service(self, mock_repo):
        return SystemService(repo=mock_repo)

    def test_get_db_health_returns_data_directly(self, service, mock_repo):
        mock_repo.get_db_stats.return_value = {
            "players": 100,
            "matches": 500,
            "training_records": 1000,
            "trained_records": 800,
            "untrained_records": 200,
            "trained_ratio": 0.8,
        }
        result = service.get_db_health()
        assert not isinstance(result, tuple)
        assert result["players"] == 100
        assert result["matches"] == 500
        assert result["trained_ratio"] == 0.8
        assert result["success"] is True
        assert result["ok"] is True

    def test_get_explorer_stats_returns_data_directly(self, service, mock_repo):
        mock_repo.get_db_stats.return_value = {"total_matches": 50}
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=2 * 1024 * 1024):
            result = service.get_explorer_stats()
            assert not isinstance(result, tuple)
            assert result["total_matches"] == 50
            assert result["db_size_mb"] == 2.0

    def test_get_explorer_stats_fallback_import(self):
        """Verify that when repo has no db_path, fallback import resolves to core.database without error."""
        repo = MagicMock(spec=[])  # No db_path attribute
        repo.get_db_stats = MagicMock(return_value={"total_matches": 10})
        service = SystemService(repo=repo)
        
        with patch("os.path.exists", return_value=False):
            result = service.get_explorer_stats()
            assert not isinstance(result, tuple)
            assert result["total_matches"] == 10
            assert result["db_size_mb"] == 0.0
