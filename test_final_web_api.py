"""
Test script for the Final NNriot Web API.
Verifies the integration between the Flask web server and the backend
prediction/training components.
"""

import requests
import json
import time


def test_final_web_api():
    """Test the final web API endpoints."""
    base_url = "http://localhost:5000"

    print("🧪 Testing Final NNriot Web API")
    print("=" * 50)

    # Test 1: Check system status
    print("\n1️⃣ Testing system status...")
    try:
        response = requests.get(f"{base_url}/api/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"   ✅ Status: Model loaded: {status.get('model_loaded', False)}")
            print(
                f"   🚀 TensorFlow version: {status.get('tensorflow_version', 'Unknown')}"
            )
            print(f"   📊 Input dimension: {status.get('input_dimension', 'Unknown')}")
        else:
            print(f"   ❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Status check error: {e}")
        return False

    # Test 2: Get sample match data
    print("\n2️⃣ Testing sample match endpoint...")
    try:
        response = requests.get(f"{base_url}/api/sample-match", timeout=10)
        if response.status_code == 200:
            sample = response.json()
            print(
                f"   ✅ Sample match retrieved: {len(sample.get('participants', []))} participants"
            )
            print(f"   🎮 Game mode: {sample.get('game_mode', 'Unknown')}")
            print(f"   ⏱️ Duration: {sample.get('game_duration', 0)} seconds")
        else:
            print(f"   ❌ Sample match failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Sample match error: {e}")

    # Test 3: Test prediction with sample data
    print("\n3️⃣ Testing prediction endpoint...")
    sample_match = {
        "match_id": "TEST_123456",
        "game_mode": "CLASSIC",
        "game_duration": 1800,
        "participants": [
            {
                "summoner_name": "TestPlayer1",
                "champion_name": "Aatrox",
                "team_id": 100,
                "win": True,
                "kills": 10,
                "deaths": 2,
                "assists": 8,
                "gold_earned": 15000,
                "cs": 180,
                "role": "TOP",
                "items": [3082, 3071, 3075, 3006, 3053, 1036],
            },
            {
                "summoner_name": "TestPlayer2",
                "champion_name": "Kaisa",
                "team_id": 200,
                "win": False,
                "kills": 8,
                "deaths": 4,
                "assists": 6,
                "gold_earned": 14000,
                "cs": 160,
                "role": "CARRY",
                "items": [3085, 3005, 3035, 3086, 3006, 3153],
            },
        ],
        "teams": [
            {"team_id": 100, "win": True, "dragon_kills": 2, "baron_kills": 1},
            {"team_id": 200, "win": False, "dragon_kills": 1, "baron_kills": 0},
        ],
    }

    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json=sample_match,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"   ✅ Prediction successful!")
                print(f"   🎯 Outcome: {result['predicted_outcome']}")
                print(f"   📊 Win Probability: {result['win_probability']:.3f}")
                print(f"   📊 Lose Probability: {result['lose_probability']:.3f}")
                print(f"   🔒 Confidence: {result['confidence']:.3f}")
            else:
                print(
                    f"   ❌ Prediction failed: {result.get('error', 'Unknown error')}"
                )
        else:
            print(f"   ❌ Prediction request failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")

    # Test 4: Test training endpoint
    print("\n4️⃣ Testing training endpoint...")
    training_data = {
        "matches": [
            {
                "match_id": "TRAIN_001",
                "game_mode": "CLASSIC",
                "game_duration": 1800,
                "participants": [
                    {
                        "summoner_name": "TrainPlayer1",
                        "champion_name": "Aatrox",
                        "team_id": 100,
                        "kills": 12,
                        "deaths": 3,
                        "assists": 8,
                        "gold_earned": 15200,
                        "cs": 180,
                    }
                ],
                "teams": [
                    {"team_id": 100, "win": True, "dragon_kills": 3, "baron_kills": 1}
                ],
            }
        ],
        "labels": [[1.0, 0.0]],
    }

    try:
        response = requests.post(
            f"{base_url}/api/train",
            json=training_data,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"   ✅ Training successful!")
                print(f"   📈 Epochs: {result['epochs']}")
                print(f"   📉 Final Loss: {result['final_loss']:.4f}")
                print(f"   📊 Accuracy: {result['accuracy']:.4f}")
                print(f"   📊 Training Samples: {result['training_samples']}")
                print(f"   ⏱️ Training Time: {result['training_time']}s")
            else:
                print(f"   ❌ Training failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Training request failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Training error: {e}")

    print("\n🎉 Final Web API testing completed!")
    print("\n💡 To access the web interface:")
    print("   🌐 Open your browser and navigate to: http://localhost:5000")
    print("   📱 The interface is responsive and works on mobile devices")


if __name__ == "__main__":
    print("🚀 Starting Final NNriot Web API Test")
    print("Make sure the web server is running: python final_web_app.py")
    print("Waiting 5 seconds for server to start...")

    time.sleep(5)
    test_final_web_api()
