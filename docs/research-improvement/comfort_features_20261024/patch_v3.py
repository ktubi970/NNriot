import os
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))
shutil.copy('build_features_v2.py', 'build_features_v3.py')
p = 'build_features_v3.py'
s = open(p).read()
rep = [
("""            'cperf': defaultdict(lambda: [0, 0.0, 0.0]), 'clast': {}, 'opp': 0.0, 'elo2': 0.0}""",
 """            'cperf': defaultdict(lambda: [0, 0.0, 0.0]), 'clast': {}, 'opp': 0.0, 'elo2': 0.0,
            'ctimes': defaultdict(list), 'rec_roles': deque(maxlen=20), 'rec_champs': deque(maxlen=20)}"""),
("""pair = defaultdict(lambda: [0, 0])                 # sorted (champ, champ) same team -> [n, w]""",
 """pair = defaultdict(lambda: [0, 0])                 # sorted (champ, champ) same team -> [n, w]
curve = defaultdict(lambda: [0, 0])                # (champ, experience bucket) -> [n, w]


def bucket(n_on_champ):
    return min(int(math.log2(1 + n_on_champ)), 6)"""),
# record experience curve BEFORE incrementing the player's champion count
("""            c = s['champ'][p[F['championName']]]; c[0] += 1; c[1] += win""",
 """            c = s['champ'][p[F['championName']]]
            cv = curve[(p[F['championName']], bucket(c[0]))]; cv[0] += 1; cv[1] += win
            c[0] += 1; c[1] += win
            s['ctimes'][p[F['championName']]].append(m['end'])
            s['rec_roles'].append(ROLES[r]); s['rec_champs'].append(p[F['championName']])"""),
("""            [np.nan] * 7 + [0.0, 0.0, np.nan, np.nan, np.nan, np.nan]""",
 """            [np.nan] * 7 + [0.0, 0.0, np.nan, np.nan, np.nan, np.nan] + \\
            [0, 0, 0, np.nan, np.nan, curve_wr(champ, 0) - g_wr]"""),
("""                   math.log1p((now - s['clast'][champ]) / 3.6e6) if champ in s['clast'] else np.nan]""",
 """                   math.log1p((now - s['clast'][champ]) / 3.6e6) if champ in s['clast'] else np.nan] + \\
                  [sum(1 for x in s['ctimes'].get(champ, ()) if now - x < 7 * 864e5),
                   sum(1 for x in s['ctimes'].get(champ, ()) if now - x < 30 * 864e5),
                   pool_rank(s, champ), sum(1 for x in s['rec_roles'] if x == role) / max(1, len(s['rec_roles'])),
                   len(set(s['rec_champs'])), curve_wr(champ, c[0]) - g_wr]"""),
("""    g = champ_glob.get((champ, role), [0, 0])""",
 """    g = champ_glob.get((champ, role), [0, 0])
    g_all = [0, 0]
    for rr_ in ROLES:
        x = champ_glob.get((champ, rr_))
        if x:
            g_all[0] += x[0]; g_all[1] += x[1]
    g_wr = sm(g_all[1], g_all[0], k=50)"""),
("""def player_feats(p, r, now, mates_on_team):""",
 """def curve_wr(champ, n_on_champ):
    cv = curve.get((champ, bucket(n_on_champ)), [0, 0])
    return sm(cv[1], cv[0], k=50)


def pool_rank(s, champ):
    c = s['champ'].get(champ)
    if not c or c[0] == 0:
        return 0
    return 1 + sum(1 for v in s['champ'].values() if v[0] > c[0])


def player_feats(p, r, now, mates_on_team):"""),
("""           'log_hours_since_champ'])""",
 """           'log_hours_since_champ', 'champ_games_7d', 'champ_games_30d', 'pool_rank', 'role_recent_frac',
           'pool_size20', 'champ_curve_delta'])"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('ok')
