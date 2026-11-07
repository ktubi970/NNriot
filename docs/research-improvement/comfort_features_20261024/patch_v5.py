"""v5 = v4 + session position, quit/AFK/remake/early-FF tendency (incl. remakes), last-game form."""
import os
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))
shutil.copy('build_features_v4.py', 'build_features_v5.py')
p = 'build_features_v5.py'
s = open(p).read()
rep = [
("""            'ctimes': defaultdict(list), 'rec_roles': deque(maxlen=20), 'rec_champs': deque(maxlen=20)}""",
 """            'ctimes': defaultdict(list), 'rec_roles': deque(maxlen=20), 'rec_champs': deque(maxlen=20),
            'all_n': 0, 'afk': 0, 'remade': 0, 'early_ff': 0, 'sess_start': None, 'sess_games': 0,
            'last_vec': None}"""),
# record quit tendency for every finished game, remakes included, before the remake early-return
("""    remake = any(p[F['gameEndedInEarlySurrender']] for p in m['parts'])
    if remake:
        return
    clock[0] += 1""",
 """    remake = any(p[F['gameEndedInEarlySurrender']] for p in m['parts'])
    for side in t.values():
        for p in side:
            s = players[p[F['puuid']]]
            s['all_n'] += 1
            s['afk'] += bool(p[F['wasAfk']])
            s['remade'] += bool(remake and p[F['teamEarlySurrendered']])
            s['early_ff'] += bool((not remake) and p[F['teamEarlySurrendered']])
            gap_ok = s['last_end'] is not None and m['start'] is not None and 0 <= m['start'] - s['last_end'] < 30 * 6e4
            if gap_ok:
                s['sess_games'] += 1
            else:
                s['sess_games'] = 1; s['sess_start'] = m['start']
            if remake:
                s['last_end'] = m['end']
    if remake:
        return
    clock[0] += 1"""),
("""            s['ew'] = vec if s['ew'] is None else 0.8 * s['ew'] + 0.2 * vec""",
 """            s['ew'] = vec if s['ew'] is None else 0.8 * s['ew'] + 0.2 * vec
            s['last_vec'] = vec"""),
("""            [0, 0, 0, np.nan, np.nan, curve_wr(champ, 0) - g_wr]""",
 """            [0, 0, 0, np.nan, np.nan, curve_wr(champ, 0) - g_wr] + quit_feats(s, now) + [np.nan] * 4"""),
("""                   len(set(s['rec_champs'])), curve_wr(champ, c[0]) - g_wr]""",
 """                   len(set(s['rec_champs'])), curve_wr(champ, c[0]) - g_wr] + quit_feats(s, now) + \\
                  [float(s['last_vec'][0]), float(s['last_vec'][1]), float(s['last_vec'][2]),
                   float(s['last_vec'][1] - s['ew'][1])]"""),
("""def curve_wr(champ, n_on_champ):""",
 """def quit_feats(s, now):
    if s is None or s['all_n'] == 0:
        return [0.0, 0.02, 0.02, 0.05, 0, 0.0]
    k = s['all_n']
    in_session = s['last_end'] is not None and 0 <= now - s['last_end'] < 30 * 6e4
    return [math.log1p(k), (s['afk'] + 0.2) / (k + 10), (s['remade'] + 0.2) / (k + 10), (s['early_ff'] + 0.5) / (k + 10),
            s['sess_games'] if in_session else 0,
            (now - s['sess_start']) / 6e4 if in_session and s['sess_start'] else 0.0]


def curve_wr(champ, n_on_champ):"""),
("""           'pool_size20', 'champ_curve_delta'])""",
 """           'pool_size20', 'champ_curve_delta', 'log_all_n', 'afk_rate', 'remade_rate', 'early_ff_rate',
           'session_games', 'session_minutes', 'last_win', 'last_kda', 'last_gshare', 'last_kda_vs_ewm'])"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('ok')
