"""One read-only pass over training_data.db: compact per-participant records for CLASSIC matches.

Output: pickle with a list of match dicts (metadata + 10 participant tuples).
Pre-match fields (champ select): champion, position, role preference status, level, spells, runes.
Post-match fields are kept ONLY to build point-in-time history for later matches.
"""
import pickle
import sqlite3
import sys
import time
sys.path.insert(0, r'D:\projet\NNriot')
from core.pre_match_dataset import _decode

DB = r'D:\projet\NNriot\training_data.db'
OUT = sys.argv[1]

PFIELDS = ('puuid', 'teamId', 'championName', 'teamPosition', 'individualPosition',
           'selectedRolePreferences', 'summonerLevel', 'summoner1Id', 'summoner2Id', 'profileIcon',
           'win', 'kills', 'deaths', 'assists', 'totalMinionsKilled', 'neutralMinionsKilled', 'goldEarned',
           'visionScore', 'totalDamageDealtToChampions', 'damageSelfMitigated', 'champLevel',
           'gameEndedInEarlySurrender', 'timePlayed', 'wasAfk', 'teamEarlySurrendered', 'eligibleForProgression')


def perks_of(p):
    perks = p.get('perks') or {}
    styles = perks.get('styles') or []
    prim = styles[0] if styles else {}
    sub = styles[1] if len(styles) > 1 else {}
    sel = prim.get('selections') or [{}]
    return (sel[0].get('perk'), prim.get('style'), sub.get('style'))


def main():
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    conn.execute('PRAGMA query_only=ON')
    conn.execute('PRAGMA cache_size=-262144')
    t0 = time.time()
    out, bad = [], 0
    cur = conn.execute("SELECT match_id, game_creation_time, raw_json FROM matches WHERE game_mode='CLASSIC'")
    for i, (mid, creation, raw) in enumerate(cur):
        d = _decode(raw)
        info = d.get('info') if isinstance(d, dict) else None
        if not isinstance(info, dict) or not isinstance(info.get('participants'), list):
            bad += 1
            continue
        parts = []
        for p in info['participants']:
            parts.append(tuple(p.get(k) for k in PFIELDS) + perks_of(p))
        out.append({'match_id': mid, 'creation': creation, 'start': info.get('gameStartTimestamp'),
                    'end': info.get('gameEndTimestamp'), 'duration': info.get('gameDuration'),
                    'version': info.get('gameVersion'), 'queue': info.get('queueId'),
                    'platform': info.get('platformId'), 'parts': parts})
        if i % 10000 == 0:
            print(i, round(time.time() - t0, 1), flush=True)
    with open(OUT, 'wb') as fh:
        pickle.dump({'fields': PFIELDS + ('keystone', 'primary_style', 'sub_style'), 'matches': out}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print('done', len(out), 'bad', bad, round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
