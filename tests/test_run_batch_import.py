import pytest
from unittest.mock import patch, mock_open, call
import run_batch_import

@patch("run_batch_import.data_collector.collect_batch_with_smurfs")
@patch("run_batch_import.time.sleep")
@patch("import_liquipedia.resolve_pro_name")
@patch("run_batch_import.os.path.exists")
def test_main_resolves_pro_names(mock_exists, mock_resolve, mock_sleep, mock_collect, monkeypatch):
    mock_exists.return_value = True
    
    # Mock file content
    file_content = "Faker#KR1\nRazork\nUnknownPro"
    mocked_open = mock_open(read_data=file_content)
    
    # Mock resolve_pro_name behavior
    def side_effect(name):
        if name == "Razork":
            return ("Razork", "EUW")
        return None
    mock_resolve.side_effect = side_effect
    
    with patch("run_batch_import.open", mocked_open):
        run_batch_import.main()
        
    # Check the players passed to collect_batch_with_smurfs
    expected_players = [
        ("Faker", "KR1"),
        ("Razork", "EUW")
    ]
    mock_collect.assert_called_once()
    args, kwargs = mock_collect.call_args
    assert args[0] == expected_players
    
    # Check sleep was called for the two names without '#'
    assert mock_sleep.call_count == 2
