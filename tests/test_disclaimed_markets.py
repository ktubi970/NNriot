import numpy as np
import pytest
from app.core import _format_multi_output_response


def test_format_multi_output_disclaims_kills_odd_and_timeline():
    dummy_preds = {
        "winner": np.array([[0.52, 0.48]]),
        "team_b_kill_lead": np.array([[0.51, 0.49]]),
        "first_blood": np.array([[0.48, 0.48, 0.04]]),
        "first_baron": np.array([[0.40, 0.40, 0.20]]),
        "first_inhibitor": np.array([[0.40, 0.40, 0.20]]),
        "first_tower": np.array([[0.49, 0.49, 0.02]]),
        "kills_odd": np.array([[0.73]]),  # Even if raw prediction outputs 0.73
        "both_baron": np.array([[0.15]]),
        "both_inhibitor": np.array([[0.30]]),
        "both_dragon": np.array([[0.85]]),
        "elder_dragon": np.array([[0.10]]),
        "kill_handicap": np.array([[0.0]]),
        "total_kills": np.array([[0.0]]),
        "team_a_kills": np.array([[0.0]]),
        "team_b_kills": np.array([[0.0]]),
        "total_barons": np.array([[0.0]]),
        "total_dragons": np.array([[0.0]]),
        "total_towers": np.array([[0.0]]),
        "first_to_5_kills": np.array([[0.0, 0.0, 1.0]]),
        "first_to_10_kills": np.array([[0.0, 0.0, 1.0]]),
        "first_to_15_kills": np.array([[0.0, 0.0, 1.0]]),
        "first_to_20_kills": np.array([[0.0, 0.0, 1.0]]),
    }
    formatted = _format_multi_output_response(dummy_preds)
    
    assert formatted["is_pre_match"] is True
    # kills_odd must include honesty disclaimer and fixed 0.5 prior
    assert formatted["kills"]["odd_probability"] == 0.5
    assert formatted["kills"]["odd_is_coinflip"] is True
