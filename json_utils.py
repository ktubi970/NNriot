import numpy as np
import hashlib
import json
from typing import Union, List, Dict, Any, Optional
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize


def extract_match_features(
    match_details: Dict[str, Any], region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Centralized logic to convert raw Riot API match details into the feature dict
    used for vectorization and training.

    Args:
        match_details: Raw match JSON from Riot API
        region: Optional region tag

    Returns:
        Feature dictionary containing team_a, team_b, and metadata
    """
    try:
        # Handle Match-V5 structure
        info = match_details.get(
            "info", match_details
        )  # Support both full and 'info' subcomponent
        participants = info.get("participants", [])

        team_a, team_b = [], []

        for p in participants:
            # Extract standard metrics
            p_data = {
                "champion": p.get("championName") or p.get("champion_name", "UNKNOWN"),
                "kda": (p.get("kills", 0) + p.get("assists", 0))
                / max(1, p.get("deaths", 1)),
                "gold": p.get("goldEarned") or p.get("gold_earned", 0),
                "role": p.get("teamPosition") or p.get("role", "UNKNOWN"),
            }

            # Determine team (100 is Blue/A, 200 is Red/B)
            team_id = p.get("teamId") or p.get("team_id")
            if team_id == 100:
                team_a.append(p_data)
            else:
                team_b.append(p_data)

        game_version = info.get("gameVersion") or info.get("game_version", "UNKNOWN")
        season_id = None
        if game_version and game_version != "UNKNOWN":
            try:
                # Version format is "14.1.1...", major part is season
                season_id = int(game_version.split(".")[0])
            except (ValueError, IndexError):
                pass

        feature = {
            "team_a": team_a,
            "team_b": team_b,
            "game_mode": info.get("gameMode") or info.get("game_mode", "CLASSIC"),
            "game_version": game_version,
            "season_id": season_id,
        }

        if region:
            feature["region"] = region

        return feature
    except Exception as e:
        # Re-raise with context
        raise ValueError(f"Failed to extract features from match data: {e}")


def flatten_json(
    y: Union[Dict, List], separator: str = ".", max_depth: int = 10
) -> Dict[str, Any]:
    """
    Flatten a nested JSON object into a single-level dictionary with improved handling.

    Args:
        y: JSON object to flatten (dict or list)
        separator: Separator for nested keys (default: '.')
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Flattened dictionary with dot-separated keys
    """
    out = {}

    def flatten_recursive(x: Any, name: str = "", depth: int = 0):
        if depth > max_depth:
            return

        if isinstance(x, dict):
            for key in x:
                new_name = f"{name}{separator}{key}" if name else key
                flatten_recursive(x[key], new_name, depth + 1)
        elif isinstance(x, list):
            for i, item in enumerate(x):
                new_name = f"{name}{separator}{i}" if name else str(i)
                flatten_recursive(item, new_name, depth + 1)
        else:
            out[name] = x

    flatten_recursive(y)
    return out


def normalize_value(
    value: Any, normalize_numeric: bool = True, max_string_length: int = 100
) -> float:
    """
    Normalize different types of values to numerical features.

    Args:
        value: The value to normalize
        normalize_numeric: Whether to normalize numeric values
        max_string_length: Maximum length for string truncation

    Returns:
        Normalized float value
    """
    # NOTE: bool must be checked before int/float because bool is a subclass of int
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        if normalize_numeric:
            # Log transformation for large numbers to reduce skew
            if abs(value) > 1:
                return np.sign(value) * np.log1p(abs(value))
            return float(value)
        return float(value)
    elif isinstance(value, str):
        # String length as feature
        return min(len(value), max_string_length) / max_string_length
    elif value is None:
        return 0.0
    else:
        # For other types, use deterministic MD5 hash as feature
        hash_val = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
        return float(hash_val % 1000) / 1000.0


def create_feature_string(
    key: str,
    value: Any,
    include_path: bool = True,
    include_type: bool = True,
    include_value_hash: bool = False,
) -> str:
    """
    Create a feature string with multiple encoding strategies.

    Args:
        key: The flattened key path
        value: The value at this key
        include_path: Include the full path in feature
        include_type: Include value type in feature
        include_value_hash: Include value hash for categorical values

    Returns:
        Feature string for hashing
    """
    parts = []

    if include_path and key:
        parts.append(f"path:{key}")

    if include_type:
        parts.append(f"type:{type(value).__name__}")

    if include_value_hash and isinstance(value, (str, bool)):
        # Include hash for categorical/string values
        value_hash = hashlib.md5(str(value).encode()).hexdigest()[:8]
        parts.append(f"hash:{value_hash}")

    return "|".join(parts)


def enhanced_json_to_vector(
    json_data: Union[Dict, List[Dict]],
    dim: int = 100000,
    normalize_numeric: bool = True,
    feature_combinations: bool = True,
    n_grams: int = 2,
    sparsity_threshold: float = 0.01,
) -> np.ndarray:
    """
    Convert JSON to vector with enhanced feature engineering using FeatureHasher.

    Args:
        json_data: JSON object or list of objects
        dim: Output vector dimension
        normalize_numeric: Normalize numeric values
        feature_combinations: Create interaction features
        n_grams: Create n-gram features from keys
        sparsity_threshold: Minimum value to include in sparse representation

    Returns:
        Enhanced feature vector as a scipy sparse matrix
    """
    if isinstance(json_data, dict):
        json_data = [json_data]

    features_list = []

    for item in json_data:
        flattened = flatten_json(item)
        base_features = {}

        for key, value in flattened.items():
            feature_strs = [
                create_feature_string(key, value, include_path=True, include_type=True),
                create_feature_string(
                    key, value, include_path=True, include_type=False
                ),
                create_feature_string(
                    key,
                    value,
                    include_path=False,
                    include_type=True,
                    include_value_hash=True,
                ),
            ]

            for feature_str in feature_strs:
                normalized_value = normalize_value(value, normalize_numeric)
                base_features[feature_str] = (
                    base_features.get(feature_str, 0.0) + normalized_value
                )

        # Create feature combinations (interaction features)
        if feature_combinations and len(base_features) > 1:
            feature_names = list(base_features.keys())
            for i in range(len(feature_names)):
                for j in range(
                    i + 1, min(i + 4, len(feature_names))
                ):  # Limit combinations
                    name1, name2 = feature_names[i], feature_names[j]
                    combo_name = f"combo:{name1}*{name2}"
                    combo_value = base_features[name1] * base_features[name2]
                    base_features[combo_name] = (
                        base_features.get(combo_name, 0.0) + combo_value
                    )

        # Create n-gram features from key paths
        if n_grams > 1:
            for key in flattened.keys():
                path_parts = key.split(".")
                for n in range(2, min(n_grams + 1, len(path_parts) + 1)):
                    for i in range(len(path_parts) - n + 1):
                        ngram = ".".join(path_parts[i : i + n])
                        ngram_feature = f"ngram:{ngram}"
                        base_features[ngram_feature] = (
                            base_features.get(ngram_feature, 0.0) + 1.0
                        )

        # Sparsity filter
        filtered_features = {
            k: v for k, v in base_features.items() if abs(v) > sparsity_threshold
        }
        features_list.append(filtered_features)

    if not features_list:
        return csr_matrix((0, dim), dtype=np.float32)

    hasher = FeatureHasher(
        n_features=dim, input_type="dict", alternate_sign=True, dtype=np.float32
    )
    vectorized = hasher.transform(features_list)

    # L2 normalization
    normalized_vectorized = normalize(vectorized, norm="l2", axis=1)
    return normalized_vectorized


def create_sparse_vector(
    json_data: Union[Dict, List[Dict]], dim: int = 100000, sparsity: float = 0.1
) -> np.ndarray:
    """
    Create a sparse vector representation optimized for high-dimensional data.

    Args:
        json_data: JSON object or list
        dim: Vector dimension
        sparsity: Target sparsity level (0.0 to 1.0)

    Returns:
        Sparse vector representation
    """
    vectors = enhanced_json_to_vector(json_data, dim, sparsity_threshold=sparsity)

    # Additional sparsification on sparse array data
    if vectors.nnz > 0:
        threshold = np.percentile(np.abs(vectors.data), 100 * (1 - sparsity))
        mask = np.abs(vectors.data) >= threshold
        vectors.data = vectors.data * mask
        vectors.eliminate_zeros()

    return vectors


def get_feature_importance(
    json_data: Union[Dict, List[Dict]], dim: int = 100000
) -> Dict[str, float]:
    """
    Analyze feature importance in the vectorized representation.

    Args:
        json_data: JSON data to analyze
        dim: Vector dimension

    Returns:
        Dictionary of feature importance scores
    """
    vectors = enhanced_json_to_vector(json_data, dim)

    # Calculate feature importance based on magnitude
    feature_importance = {}
    coo = vectors.tocoo()
    for col, value in zip(coo.col, coo.data):
        if abs(value) > 0.001:  # Ignore very small values
            feature_importance[f"feature_{col}"] = feature_importance.get(
                f"feature_{col}", 0
            ) + abs(value)

    # Sort by importance
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )
    return dict(sorted_features)


# Backward compatibility - keep original function name
def json_to_vector(json_data, dim=100000):
    """
    Enhanced version of the original json_to_vector function.
    Uses improved feature engineering while maintaining the same interface.
    """
    return enhanced_json_to_vector(json_data, dim)


if __name__ == "__main__":
    import test_data

    # Quick test with enhanced features - League of Legends match data
    test_json = test_data.get_extended_test_data()

    print("=== Enhanced JSON Vectorization Test - League of Legends Matches ===")

    # Test enhanced vectorization
    vec = enhanced_json_to_vector(test_json, dim=1000)
    print(f"Enhanced vectorized JSON shape: {vec.shape}")
    total_size = vec.shape[0] * vec.shape[1]
    print(f"Non-zero features: {vec.nnz} / {total_size}")
    print(f"Sparsity: {1 - vec.nnz / total_size:.3f}")

    # Test sparse vectorization
    sparse_vec = create_sparse_vector(test_json, dim=1000, sparsity=0.05)
    print(f"Sparse vector non-zero features: {sparse_vec.nnz} / {total_size}")

    # Test feature importance
    importance = get_feature_importance(test_json, dim=1000)
    print(f"Top 5 most important features: {list(importance.items())[:5]}")

    # Compare with original method
    print("\n=== Comparison with Original Method ===")
    original_vec = json_to_vector(test_json, dim=1000)
    enhanced_vec = enhanced_json_to_vector(test_json, dim=1000)

    print(f"Original method non-zero: {original_vec.nnz}")
    print(f"Enhanced method non-zero: {enhanced_vec.nnz}")
    print(
        f"Correlation between methods: {np.corrcoef(original_vec.toarray().flatten(), enhanced_vec.toarray().flatten())[0,1]:.3f}"
    )
