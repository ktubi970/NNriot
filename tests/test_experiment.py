from copy import deepcopy
import pytest
from core.pre_match_dataset import chronological_split, to_pre_match_features
from core.experiment import build_manifest, restore_splits


def records(n=40):
    from core.feature_labels import LABEL_KEYS
    rows = []
    for i in range(n):
        labels = dict.fromkeys(LABEL_KEYS, 0)
        labels['winner'] = i % 2
        player = {'championName': 'Ahri', 'teamPosition': 'MIDDLE',
                  'gold': 10000, 'career_win_rate': 0.99}
        rows.append({'match_id': f'M_{i:04d}', 'game_creation_time': 1000 + i,
                     'winner_label': i % 2, 'labels_json': labels,
                     'feature_json': {'game_version': '16.18.1',
                                      'team_a': [dict(player) for _ in range(5)],
                                      'team_b': [dict(player) for _ in range(5)]}})
    return rows


def manifest_for(rows):
    return build_manifest(chronological_split(rows), source_db=None,
        split_metadata={'requested_strategy': 'chronological',
                        'effective_strategy': 'chronological', 'fallback_reason': None})


def test_normalization_preserves_aliases_and_excludes_undated_stats():
    clean = to_pre_match_features(records()[0]['feature_json'])
    assert clean['team_a'][0] == {'champion': 'Ahri', 'role': 'MIDDLE'}


def test_replay_uses_manifest_membership_even_with_newer_rows():
    rows = records()
    manifest = manifest_for(rows)
    restored = restore_splits(list(reversed(records(45))), manifest)
    assert [r['match_id'] for r in restored['test']] == [
        r['match_id'] for r in chronological_split(rows)['test']]


@pytest.mark.parametrize('mutation', ['missing', 'timestamp', 'label', 'feature', 'duplicate'])
def test_replay_rejects_changed_source(mutation):
    rows = records()
    manifest = manifest_for(rows)
    changed = deepcopy(rows)
    if mutation == 'missing':
        changed.pop()
    elif mutation == 'timestamp':
        changed[-1]['game_creation_time'] += 100
    elif mutation == 'label':
        changed[-1]['labels_json']['winner'] ^= 1
        changed[-1]['winner_label'] ^= 1
    elif mutation == 'feature':
        changed[-1]['feature_json']['team_a'][0]['championName'] = 'Zed'
    else:
        changed.append(deepcopy(changed[-1]))
    with pytest.raises(ValueError):
        restore_splits(changed, manifest)
