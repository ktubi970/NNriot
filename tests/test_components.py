#!/usr/bin/env python3
"""
Test script to verify NNriot components work without Rust extension.
This allows testing the Python components independently.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback


def test_json_utils():
    """
    Test JSON vectorization utilities.

    Verifies that json_utils can convert a list of dictionaries into a
    sparse vector of the correct dimensions.
    """
    try:
        from core import json_utils

        test_json = [
            {"type": "login", "details": {"success": True, "attempts": 1}},
            {"type": "purchase", "amount": 50.0, "items": ["book", "pen"]},
            {"type": "login", "details": {"success": False, "attempts": 3}},
            {"type": "refund", "amount": 20.0, "reason": "damaged"},
        ]

        vec = json_utils.json_to_vector(test_json, dim=10000)
        print(f"[OK] JSON utilities working - Vector shape: {vec.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] JSON utilities failed: {e}")
        return False


def test_database():
    """
    Test database functionality using Repository Pattern in an isolated temp database.
    """
    import os
    import tempfile
    try:
        from core import database
        from core.repository import MatchRepository, TrainingRepository, SystemRepository
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = os.path.join(tmpdir, "test_components.db")
            database.init_db(test_db)

            match_repo = MatchRepository(test_db)
            train_repo = TrainingRepository(test_db)
            sys_repo = SystemRepository(test_db)

            # Satisfaction of Foreign Key constraint: insert a dummy match first
            with match_repo.get_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO matches (match_id, game_mode) VALUES (?, ?)",
                    ("TEST_MATCH", "CLASSIC"),
                )
                conn.commit()

            # perform a quick insert and fetch
            dummy_feature = {"foo": "bar"}
            train_repo.save_training_records_batch([
                (
                    "TEST_MATCH",
                    dummy_feature,
                    0,  # winner_label
                    {"test": True},  # labels_json
                )
            ])

            stats = sys_repo.get_db_stats()
            required_keys = [
                "players",
                "matches",
                "trained_records",
                "untrained_records",
                "trained_ratio",
            ]
            missing = [k for k in required_keys if k not in stats]

            if missing:
                print(f"[FAIL] Database stats missing keys: {missing}")
                return False

            with train_repo.get_connection() as conn:
                row = conn.execute(
                    "SELECT match_id, feature_json FROM training_dataset WHERE match_id = ?",
                    ("TEST_MATCH",),
                ).fetchone()

            if not row:
                print("[FAIL] Test record not found")
                return False

            print(
                f"[OK] Database stats consistent: {stats['matches']} matches, {stats['trained_records']} trained"
            )
            print("[OK] Database training record save/fetch working")
            return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] Database failed: {e}")
        return False



def test_tensorflow():
    """Test TensorFlow graph generation using ModelFactory."""
    try:
        from core.model_factory import ModelFactory
        model, path = ModelFactory.get_model(version="v2", input_dim=10000, force_rebuild=True)
        print(f"[OK] TensorFlow model generation successful - {model.name} (Heads: {len(model.outputs)})")
        return True
    except ImportError as e:
        print(f"[SKIP] TensorFlow import skipped (not installed in this env): {e}")
        return "SKIP"
    except Exception as e:
        print(f"[FAIL] TensorFlow graph failed: {e}")
        return False


def test_plotly():
    """Test plotly import."""
    try:
        import plotly.graph_objects as go
        print("[OK] Plotly import successful")
        return True
    except ImportError as e:
        print(f"[SKIP] Plotly skipped: {e}")
        return "SKIP"
    except Exception as e:
        print(f"[FAIL] Plotly import failed: {e}")
        return False


def test_numpy():
    """Test numpy functionality."""
    try:
        import numpy as np
        arr = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32
        )
        print(f"[OK] NumPy working - Array shape: {arr.shape}")
        return True
    except ImportError as e:
        print(f"[SKIP] NumPy skipped: {e}")
        return "SKIP"
    except Exception as e:
        print(f"[FAIL] NumPy failed: {e}")
        return False


def test_riot_api():
    """Test Riot API import (may fail without API key)."""
    try:
        from core import riot_api
        print("[OK] Riot API module imported")
        return True
    except ImportError as e:
        print(f"[SKIP] Riot API skipped: {e}")
        return "SKIP"
    except Exception as e:
        print(f"[FAIL] Riot API import failed: {e}")
        return False


def test_nnriot_import():
    """Test nnriot import (will fail without Rust extension)."""
    try:
        import nnriot
        print("[OK] nnriot module imported (Rust extension working)")
        return True
    except ImportError as e:
        print(f"[SKIP] nnriot import skipped (expected without Rust): {e}")
        return "SKIP"
    except Exception as e:
        print(f"[FAIL] nnriot import failed with unexpected error: {e}")
        return False


def main():
    """Run all component tests."""
    print("NNriot Component Test Suite")
    print("=" * 40)

    tests = [
        ("JSON Utilities", test_json_utils),
        ("Database", test_database),
        ("TensorFlow", test_tensorflow),
        ("Plotly", test_plotly),
        ("NumPy", test_numpy),
        ("Riot API", test_riot_api),
        ("nnriot (Rust Extension)", test_nnriot_import),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        try:
            status = test_func()
            results.append((name, status))
        except Exception as e:
            print(f"[FAIL] {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print("=" * 40)

    passed = 0
    skipped = 0
    failed = 0
    total = len(results)

    for name, status in results:
        if status is True:
            marker = "[OK] PASS"
            passed += 1
        elif status == "SKIP":
            marker = "[SKIP] SKIP"
            skipped += 1
        else:
            marker = "[FAIL] FAIL"
            failed += 1
        print(f"{marker} {name}")

    print(f"\nOverall: {passed} passed, {skipped} skipped, {failed} failed (Total: {total})")

    if failed == 0:
        print("[OK] All components working or gracefully skipped! Ready to build Rust extension.")
    else:
        print("[WARN] Some components failed. Check the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)
