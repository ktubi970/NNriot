"""Summarize completed target audit without reading or changing the source DB."""
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    result = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'completed'
    previous = json.loads((ROOT / 'runs/aligned_dataset_20261009_reviewed/model.split.json').read_text(encoding='utf-8'))
    previous_ids = {r['match_id'] for group in previous['splits'].values() for r in group}
    previous_by_id = {r['match_id']: name for name, group in previous['splits'].items() for r in group}
    previous_seen = set()
    cutoff = int(datetime.fromisoformat(result['future_test_after_utc']).timestamp() * 1000)
    last_ms = int(datetime.fromisoformat(result['source_last_utc']).timestamp() * 1000)
    cohorts = {}
    failures = []
    for name in ('all', 'after_previous_reserved_test', 'previous_frozen_membership', 'recent_30d',
                 'previous_train', 'previous_validation', 'previous_test'):
        cohorts[name] = {'counts': Counter(), 'players_with_history': Counter(), 'contract_failures': Counter()}
    with gzip.open(OUT / 'targets.jsonl.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            groups = ['all']
            if row['game_creation_time'] > cutoff:
                groups.append('after_previous_reserved_test')
            if row['game_creation_time'] >= last_ms - 30 * 86400000:
                groups.append('recent_30d')
            if row['match_id'] in previous_ids:
                groups.append('previous_frozen_membership')
                groups.append('previous_' + previous_by_id[row['match_id']])
                previous_seen.add(row['match_id'])
            for name in groups:
                cohort = cohorts[name]
                counts = cohort['counts']
                counts['targets'] += 1
                counts['eligible18'] += row['status'] == 'eligible'
                counts['protected_prior_test'] += row['protected_prior_test']
                counts['own_timeline_usable'] += row['own_timeline_usable']
                counts['players_with_history'] += row['player_timeline_count']
                counts['player_slots'] += 10
                counts['all_ten_players_with_history'] += row['player_timeline_count'] == 10
                counts['at_least_five_players_with_history'] += row['player_timeline_count'] >= 5
                for threshold in (3, 5, 10):
                    counts[f'all_ten_players_with_at_least_{threshold}_histories'] += row['minimum_player_history'] >= threshold
                cohort['players_with_history'][row['player_timeline_count']] += 1
                if row['status'] != 'eligible':
                    cohort['contract_failures'][row['status']] += 1
            if row['status'] != 'eligible':
                failures.append(row)
    assert cohorts['all']['counts']['targets'] == result['target_count']
    assert cohorts['all']['counts']['eligible18'] == result['eligible18']
    assert dict(cohorts['all']['contract_failures']) == result['contract_failures']
    assert {str(k): v for k, v in cohorts['all']['players_with_history'].items()} == result['players_with_history_per_target']
    for cohort in cohorts.values():
        counts = cohort['counts']
        cohort['coverage18'] = counts['eligible18'] / counts['targets'] if counts['targets'] else None
        cohort['player_history_coverage'] = counts['players_with_history'] / counts['player_slots'] if counts['player_slots'] else None
    summary = {
        'status': 'verified',
        'target_records_reconciled_with_result': True,
        'previous_frozen_membership_count': len(previous_ids),
        'previous_membership_not_in_selected_targets': len(previous_ids - previous_seen),
        'previous_membership_not_selected_ids': sorted(previous_ids - previous_seen),
        'new_test_after_utc': result['future_test_after_utc'],
        'cohorts': cohorts,
        'all_contract_failures': failures,
        'artifact_sha256': {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                            for name in ('audit.py', 'result.json', 'targets.jsonl.gz', 'summarize.py')},
    }
    (OUT / 'cohort_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('previous_membership_not_selected_ids', 'all_contract_failures')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
