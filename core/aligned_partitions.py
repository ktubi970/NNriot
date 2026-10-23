# core/aligned_partitions.py
from core.pre_match_dataset import _decode, _valid_labels
from core.pre_match_history import decode_history_row
from core.serving_contract import ContractError, FeatureContract

def select_targets(rows, *, excluded_ids, warmup_rows=10000):
    if type(warmup_rows) is not int or warmup_rows < 0:
        raise ValueError('warmup_rows must be nonnegative')
    selected, rejected, seen = [], [], set()
    ordered = sorted(rows, key=lambda r: (r['game_creation_time'], r['match_id']))
    for index, row in enumerate(ordered):
        mid = row['match_id']
        if mid in seen:
            raise ValueError(f'duplicate match_id: {mid}')
        seen.add(mid)
        if index < warmup_rows:
            continue
        try:
            if warmup_rows and row['game_creation_time'] <= ordered[warmup_rows - 1]['game_creation_time']:
                raise ValueError('history_context_incomplete')
            if mid in excluded_ids:
                raise ValueError('previous_test')
            rec = decode_history_row(row)
            info = rec['raw_info']
            if info.get('queueId') not in (420, 440):
                raise ValueError('unsupported_queue')
            if rec['result_end_ms'] is None:
                raise ValueError('invalid_result_end')
            FeatureContract('nnriot.players14.v1').validate_match(rec['feature_json'])
            parts = rec['participants']
            blue = [p for p in parts if p.get('teamId') == 100]
            red = [p for p in parts if p.get('teamId') == 200]
            if len(blue) != 5 or len(red) != 5 or any(
                    type(p.get('win')) is not bool for p in parts):
                raise ValueError('invalid_riot_winner')
            if len({p['win'] for p in blue}) != 1 or len({p['win'] for p in red}) != 1 or blue[0]['win'] == red[0]['win']:
                raise ValueError('invalid_riot_winner')
            winner = row['winner_label']
            if type(winner) is not int or winner != int(red[0]['win']):
                raise ValueError('winner_disagrees_with_riot')
            labels = _decode(row.get('labels_json'))
            if not _valid_labels(labels, winner):
                raise ValueError('invalid_multitask_labels')
            selected.append(dict(match_id=mid, game_creation_time=rec['game_creation_time'],
                result_end_ms=rec['result_end_ms'], feature_json=rec['feature_json'],
                labels_json=labels, queue_id=info['queueId'], patch=rec['patch'],
                region=rec['feature_json'].get('region')))
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append(dict(match_id=mid, reason=str(exc)))
    return selected, rejected

def partition_targets(targets, *, after_ms, validation_count=3000, test_count=3000):
    if any(type(n) is not int or n < 1 for n in (validation_count, test_count)):
        raise ValueError('partition counts must be positive integers')
    if type(after_ms) is not int or after_ms < 0:
        raise ValueError('after_ms must be nonnegative UTC milliseconds')
    ordered = sorted(targets, key=lambda r: (r['game_creation_time'], r['match_id']))
    new = [r for r in ordered if r['game_creation_time'] > after_ms]
    if len(new) < test_count:
        raise ValueError('Insufficient new test targets')
    test_start = new[-test_count]['game_creation_time']
    before_test = [r for r in ordered if r['game_creation_time'] < test_start]
    if len(before_test) <= validation_count:
        raise ValueError('Insufficient train/validation targets')
    available_validation = [r for r in before_test if r['result_end_ms'] < test_start]
    if len(available_validation) <= validation_count:
        raise ValueError('Insufficient completed validation targets')
    validation_start = available_validation[-validation_count]['game_creation_time']
    splits = {'train': [], 'validation': [], 'test': []}
    purged = []
    for row in ordered:
        start = row['game_creation_time']
        name = 'test' if start >= test_start else 'validation' if start >= validation_start else 'train'
        cutoff = validation_start if name == 'train' else test_start if name == 'validation' else None
        if cutoff is not None and row['result_end_ms'] >= cutoff:
            purged.append(dict(match_id=row['match_id'], game_creation_time=start,
                               reason='result_not_available_at_partition_start'))
        else:
            splits[name].append(row)
    if not splits['train'] or len(splits['validation']) < validation_count or len(splits['test']) < test_count:
        raise ValueError('Insufficient targets after temporal purge')
    return splits, purged
