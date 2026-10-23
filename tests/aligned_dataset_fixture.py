# tests/aligned_dataset_fixture.py
import json
from core.feature_labels import LABEL_KEYS
from core.pre_match_history import decode_history_row

ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

def source_row(mid, creation, end, winner=0, *, timeline=False):
    parts = [dict(puuid=f'p{tid}_{slot}', participantId=i + 1,
                  teamId=tid, championName='Ahri' if tid == 100 else 'Lux',
                  teamPosition=ROLES[slot], win=(winner == (tid == 200)),
                  kills=2, deaths=1, assists=3, totalMinionsKilled=100,
                  neutralMinionsKilled=0, goldEarned=5000)
             for i, (tid, slot) in enumerate(
                 (tid, slot) for tid in (100, 200) for slot in range(5))]
    labels = dict.fromkeys(LABEL_KEYS, 0)
    labels.update(winner=winner, team_b_kill_lead=winner,
                  total_kills=20, team_a_kills=10, team_b_kills=10,
                  kill_handicap=0)
    metrics = {p['puuid']: dict(gd15=1., csd15=2., xpd15=3., fbi10=0.)
               for p in parts} if timeline else None
    return dict(match_id=mid, game_creation_time=creation,
                winner_label=winner, labels_json=json.dumps(labels),
                timeline_json=None, timeline_metrics=metrics,
                raw_json=json.dumps({'info': dict(
                    gameCreation=creation, gameStartTimestamp=creation + 1,
                    gameEndTimestamp=end, gameDuration=100,
                    gameVersion='16.18.1', gameMode='CLASSIC', queueId=420,
                    participants=parts)}))

def target(row):
    return decode_history_row(row)['feature_json']
