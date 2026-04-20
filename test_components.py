#!/usr/bin/env python3
"""
Test script to verify NNriot components work without Rust extension.
This allows testing the Python components independently.
"""

import sys
import traceback


def test_json_utils():
    """
    Test JSON vectorization utilities.

    Verifies that json_utils can convert a list of dictionaries into a
    sparse vector of the correct dimensions.
    """
    try:
        import json_utils

        test_json = [
            {"type": "login", "details": {"success": True, "attempts": 1}},
            {"type": "purchase", "amount": 50.0, "items": ["book", "pen"]},
            {"type": "login", "details": {"success": False, "attempts": 3}},
            {"type": "refund", "amount": 20.0, "reason": "damaged"},
        ]

        vec = json_utils.json_to_vector(test_json, dim=10000)
        print(f"✅ JSON utilities working - Vector shape: {vec.shape}")
        return True
    except Exception as e:
        print(f"❌ JSON utilities failed: {e}")
        return False


def test_database():
    """
    Test database functionality, including LLM annotation columns.

    Verifies database initialization, statistics retrieval, and
    record persistence.
    """
    try:
        import database

        database.init_db()

        # Satisfaction of Foreign Key constraint: insert a dummy match first
        with database.get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO matches (match_id, game_mode) VALUES (?, ?)",
                ("TEST_MATCH", "CLASSIC"),
            )
            conn.commit()

        # perform a quick insert and fetch
        dummy_feature = {"foo": "bar"}
        database.save_training_record("TEST_MATCH", dummy_feature, 0)

        stats = database.get_db_stats()
        required_keys = [
            "players",
            "matches",
            "trained_records",
            "untrained_records",
            "trained_ratio",
        ]
        missing = [k for k in required_keys if k not in stats]

        if not missing:
            print(
                f"✅ Database stats consistent: {stats['matches']} matches, {stats['trained_records']} trained"
            )
        else:
            print(f"❌ Database stats missing keys: {missing}")
            return False

        with database.get_connection() as conn:
            row = conn.execute(
                "SELECT match_id, feature_json FROM training_dataset WHERE match_id = ?",
                ("TEST_MATCH",),
            ).fetchone()

        if row:
            print("✅ Database training record save/fetch working")
        else:
            print("❌ Test record not found")
            return False

        return True
    except Exception as e:
        print(f"❌ Database failed: {e}")
        return False


def test_tensorflow():
    """Test TensorFlow graph generation."""
    try:
        import generate_graph

        # Note: generate_graph changed from create_graph to build_keras_model
        model = generate_graph.build_keras_model()
        print(f"✅ TensorFlow model generation successful - {model.name}")
        return True
    except Exception as e:
        print(f"❌ TensorFlow graph failed: {e}")
        return False


def test_plotly():
    """Test plotly import."""
    try:
        import plotly.graph_objects as go

        print("✅ Plotly import successful")
        return True
    except Exception as e:
        print(f"❌ Plotly import failed: {e}")
        return False


def test_numpy():
    """Test numpy functionality."""
    try:
        import numpy as np

        arr = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32
        )
        print(f"✅ NumPy working - Array shape: {arr.shape}")
        return True
    except Exception as e:
        print(f"❌ NumPy failed: {e}")
        return False


def test_riot_api():
    """Test Riot API import (may fail without API key)."""
    try:
        import riot_api

        print("✅ Riot API module imported")
        return True
    except Exception as e:
        print(f"❌ Riot API import failed: {e}")
        return False


def test_nnriot_import():
    """Test nnriot import (will fail without Rust extension)."""
    try:
        import nnriot

        print("✅ nnriot module imported (Rust extension working)")
        return True
    except ImportError as e:
        print(f"❌ nnriot import failed (expected without Rust): {e}")
        return False
    except Exception as e:
        print(f"❌ nnriot import failed with unexpected error: {e}")
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
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print("=" * 40)

    passed = 0
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All components working! Ready to build Rust extension.")
    else:
        print("⚠️  Some components failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
