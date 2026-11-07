"""Point-in-time feature table for the prospective train/validation cohort (sealed test never featurized).

History = every CLASSIC match in the DB whose result ended strictly before the target's creation
(no 10 000-row window). Champ-select fields of the target itself (role preference status, spells,
keystone, account level) are used as pre-match information.
"""
import heapq
import json
import math
import pickle
import sys
from collections import Counter, defaultdict, deque

import numpy as np

RAW, SPLIT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ROLES = ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
RIDX = {r: i for i, r in enumerate(ROLES)}
STATUS = {'PRIMARY': 0, 'SECONDARY': 1, 'AUTOFILL': 2, 'FILL': 3}
SPELLS = {4: 'flash', 11: 'smite', 12: 'tp', 14: 'ignite', 7: 'heal', 3: 'exhaust', 21: 'barrier', 6: 'ghost', 1: 'cleanse'}

raw = pickle.load(open(RAW, 'rb'))
F = {k: i for i, k in enumerate(raw['fields'])}
split = json.load(open(SPLIT))['splits']
target_split = {e['match_id']: name for name in ('train', 'validation') for e in split[name]}
matches = sorted(raw['matches'], key=lambda m: (m['creation'], m['match_id']))
# Hard stop: nothing at or after the first sealed-test creation is featurized or replayed as a target.
test_start = min(e['game_creation_time'] for e in split['test'])
print('test_start', test_start, flush=True)


def new_player():
    return {'n': 0, 'w': 0, 'champ': defaultdict(lambda: [0, 0]), 'role': defaultdict(lambda: [0, 0]),
            'recent': deque(maxlen=20), 'streak': 0, 'last_end': None, 'times': deque(maxlen=50),
            'k': 0.0, 'd': 0.0, 'a': 0.0, 'cs': 0.0, 'gold': 0.0, 'vis': 0.0, 'dmg': 0.0, 'mins': 0.0,
            'gshare': 0.0, 'dshare': 0.0, 'elo': 0.0, 'auto_n': 0, 'auto_w': 0, 'level': None,
            'mates': Counter(), 'ewm': 0.5, 'mates_w': Counter(), 'ew': None, 'relo': defaultdict(float),
            'cperf': defaultdict(lambda: [0, 0.0, 0.0]), 'clast': {}, 'opp': 0.0, 'elo2': 0.0,
            'ctimes': defaultdict(list), 'rec_roles': deque(maxlen=20), 'rec_champs': deque(maxlen=20)}


players = defaultdict(new_player)
champ_glob = defaultdict(lambda: [0, 0])          # (champ, role) -> [n, w]
lane = defaultdict(lambda: [0, 0])                 # (role, champ, opp) -> [n, w for champ]
pair = defaultdict(lambda: [0, 0])                 # sorted (champ, champ) same team -> [n, w]
curve = defaultdict(lambda: [0, 0])                # (champ, experience bucket) -> [n, w]
HALF = 8000.0                                      # meta half-life, in applied results
clock = [0]
meta = {}                                          # (champ, role) -> [n, w, t]
meta_lane = {}                                     # (role, champ, opp) -> [n, w, t]


def dec_add(store, key, win):
    n, w, t0 = store.get(key, (0.0, 0.0, clock[0]))
    f = 0.5 ** ((clock[0] - t0) / HALF)
    store[key] = (n * f + 1, w * f + win, clock[0])


def dec_get(store, key):
    n, w, t0 = store.get(key, (0.0, 0.0, clock[0]))
    f = 0.5 ** ((clock[0] - t0) / HALF)
    return n * f, w * f


def bucket(n_on_champ):
    return min(int(math.log2(1 + n_on_champ)), 6)
pending = []
serial = 0


def parse_pref(s):
    if not isinstance(s, str) or '.' not in s:
        return -1, 0
    parts = s.split('.')
    return STATUS.get(parts[1], 4), len(parts) - 2


def team_of(m):
    t = {100: [None] * 5, 200: [None] * 5}
    for p in m['parts']:
        pos = p[F['teamPosition']]
        if pos in RIDX and p[F['teamId']] in t:
            t[p[F['teamId']]][RIDX[pos]] = p
    return t


def apply_result(m):
    t = team_of(m)
    if any(x is None for side in t.values() for x in side):
        return
    red_win = bool(t[200][0][F['win']])
    mins = max(5.0, min(60.0, (m['duration'] or 1500) / 60.0))
    remake = any(p[F['gameEndedInEarlySurrender']] for p in m['parts'])
    if remake:
        return
    clock[0] += 1
    # Elo-style team rating update (logistic, scale 400 -> natural units 1.0)
    rb = sum(players[p[F['puuid']]]['elo'] for p in t[100]) / 5
    rr = sum(players[p[F['puuid']]]['elo'] for p in t[200]) / 5
    exp_red = 1 / (1 + math.exp(-(rr - rb)))
    rb_r = sum(players[p[F['puuid']]]['relo'][ROLES[r]] for r, p in enumerate(t[100])) / 5
    rr_r = sum(players[p[F['puuid']]]['relo'][ROLES[r]] for r, p in enumerate(t[200])) / 5
    exp_red_r = 1 / (1 + math.exp(-(rr_r - rb_r)))
    rb2 = sum(players[p[F['puuid']]]['elo2'] for p in t[100]) / 5
    rr2 = sum(players[p[F['puuid']]]['elo2'] for p in t[200]) / 5
    exp_red2 = 1 / (1 + math.exp(-(rr2 - rb2)))
    for tid, side in t.items():
        gold_tot = sum(p[F['goldEarned']] or 0 for p in side) or 1
        dmg_tot = sum(p[F['totalDamageDealtToChampions']] or 0 for p in side) or 1
        win = red_win if tid == 200 else not red_win
        for r, p in enumerate(side):
            s = players[p[F['puuid']]]
            s['n'] += 1; s['w'] += win
            c = s['champ'][p[F['championName']]]
            cv = curve[(p[F['championName']], bucket(c[0]))]; cv[0] += 1; cv[1] += win
            c[0] += 1; c[1] += win
            s['ctimes'][p[F['championName']]].append(m['end'])
            s['rec_roles'].append(ROLES[r]); s['rec_champs'].append(p[F['championName']])
            ro = s['role'][ROLES[r]]; ro[0] += 1; ro[1] += win
            s['recent'].append(win)
            s['streak'] = (s['streak'] + 1 if s['streak'] >= 0 else 1) if win else (s['streak'] - 1 if s['streak'] <= 0 else -1)
            s['last_end'] = m['end']; s['times'].append(m['end'])
            s['k'] += p[F['kills']] or 0; s['d'] += p[F['deaths']] or 0; s['a'] += p[F['assists']] or 0
            s['cs'] += (p[F['totalMinionsKilled']] or 0) + (p[F['neutralMinionsKilled']] or 0)
            s['gold'] += p[F['goldEarned']] or 0; s['vis'] += p[F['visionScore']] or 0
            s['dmg'] += p[F['totalDamageDealtToChampions']] or 0; s['mins'] += mins
            s['gshare'] += (p[F['goldEarned']] or 0) / gold_tot; s['dshare'] += (p[F['totalDamageDealtToChampions']] or 0) / dmg_tot
            st, _ = parse_pref(p[F['selectedRolePreferences']])
            if st == 2:
                s['auto_n'] += 1; s['auto_w'] += win
            if p[F['summonerLevel']]:
                s['level'] = p[F['summonerLevel']]
            s['ewm'] = 0.9 * s['ewm'] + 0.1 * win
            kf = 0.6 / math.sqrt(1 + s['n'] / 10)
            s['elo'] += kf * ((1 if win else 0) - (exp_red if tid == 200 else 1 - exp_red))
            s['elo2'] += 0.25 * ((1 if win else 0) - (exp_red2 if tid == 200 else 1 - exp_red2))
            s['relo'][ROLES[r]] += 0.8 / math.sqrt(1 + s['role'][ROLES[r]][0] / 10) * ((1 if win else 0) - (exp_red_r if tid == 200 else 1 - exp_red_r))
            s['opp'] += (rb if tid == 200 else rr) - (rr if tid == 200 else rb)
            kda = ((p[F['kills']] or 0) + (p[F['assists']] or 0)) / ((p[F['deaths']] or 0) + 1)
            gsh = (p[F['goldEarned']] or 0) / gold_tot; dsh = (p[F['totalDamageDealtToChampions']] or 0) / dmg_tot
            vec = np.array([win, min(kda, 15.0), gsh, dsh, (p[F['visionScore']] or 0) / mins,
                            ((p[F['totalMinionsKilled']] or 0) + (p[F['neutralMinionsKilled']] or 0)) / mins, (p[F['deaths']] or 0)])
            s['ew'] = vec if s['ew'] is None else 0.8 * s['ew'] + 0.2 * vec
            cp = s['cperf'][p[F['championName']]]; cp[0] += 1; cp[1] += gsh; cp[2] += min(kda, 15.0)
            s['clast'][p[F['championName']]] = m['end']
            for q in side:
                if q is not p:
                    s['mates'][q[F['puuid']]] += 1
                    s['mates_w'][q[F['puuid']]] += win
            g = champ_glob[(p[F['championName']], ROLES[r])]; g[0] += 1; g[1] += win
            dec_add(meta, (p[F['championName']], ROLES[r]), win)
    for tid, side in t.items():
        win = red_win if tid == 200 else not red_win
        cs_ = sorted(p[F['championName']] for p in side)
        for i in range(5):
            for j in range(i + 1, 5):
                pr = pair[(cs_[i], cs_[j])]; pr[0] += 1; pr[1] += win
    for r in range(5):
        a, b = t[100][r][F['championName']], t[200][r][F['championName']]
        l1 = lane[(r, a, b)]; l1[0] += 1; l1[1] += not red_win
        l2 = lane[(r, b, a)]; l2[0] += 1; l2[1] += red_win
        dec_add(meta_lane, (r, b, a), red_win)


def sm(w, n, prior=0.5, k=10.0):
    return (w + k * prior) / (n + k)


def curve_wr(champ, n_on_champ):
    cv = curve.get((champ, bucket(n_on_champ)), [0, 0])
    return sm(cv[1], cv[0], k=50)


def pool_rank(s, champ):
    c = s['champ'].get(champ)
    if not c or c[0] == 0:
        return 0
    return 1 + sum(1 for v in s['champ'].values() if v[0] > c[0])


def player_feats(p, r, now, mates_on_team):
    s = players.get(p[F['puuid']])
    champ, role = p[F['championName']], ROLES[r]
    st, npref = parse_pref(p[F['selectedRolePreferences']])
    spells = {p[F['summoner1Id']], p[F['summoner2Id']]}
    level = p[F['summonerLevel']] or 0
    g = champ_glob.get((champ, role), [0, 0])
    g_all = [0, 0]
    for rr_ in ROLES:
        x = champ_glob.get((champ, rr_))
        if x:
            g_all[0] += x[0]; g_all[1] += x[1]
    g_wr = sm(g_all[1], g_all[0], k=50)
    mn, mw = dec_get(meta, (champ, role))
    base = [st, npref, math.log1p(level)] + [float(sid in spells) for sid in SPELLS] + \
           [p[F['keystone']] or -1, p[F['primary_style']] or -1, p[F['sub_style']] or -1,
            sm(g[1], g[0], k=50), math.log1p(g[0]), sm(mw, mn, k=30), math.log1p(mn)]
    if s is None or s['n'] == 0:
        return base + [0, 0.5, 0.5, 0.5, 0, 0.5, 0, 0.5, 0.5, 0, -1, 0, 0, np.nan, np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan, 0.0, 0, 0.5, np.nan, mates_on_team, 0.5, 0] + \
            [np.nan] * 7 + [0.0, 0.0, np.nan, np.nan, np.nan, np.nan] + \
            [0, 0, 0, np.nan, np.nan, curve_wr(champ, 0) - g_wr]
    n = s['n']
    c = s['champ'].get(champ, [0, 0]); ro = s['role'].get(role, [0, 0])
    rec = list(s['recent'])
    rec5 = rec[-5:]
    hours = (now - s['last_end']) / 3.6e6 if s['last_end'] else -1
    day = sum(1 for x in s['times'] if now - x < 864e5)
    week = sum(1 for x in s['times'] if now - x < 7 * 864e5)
    mins = max(s['mins'], 1.0)
    lvl_delta = (level - s['level']) if (s['level'] and level) else np.nan
    return base + [math.log1p(n), sm(s['w'], n), sm(c[1], c[0]), sm(c[1], c[0], k=3), math.log1p(c[0]),
                   sm(ro[1], ro[0]), ro[0] / n, sum(rec) / len(rec), (sum(rec5) + 1) / (len(rec5) + 2),
                   s['streak'], math.log1p(hours) if hours >= 0 else -1, day, week,
                   (s['k'] + s['a'] + 5) / (s['d'] + 2), s['cs'] / mins, s['gold'] / mins, s['vis'] / mins,
                   s['dmg'] / mins, s['gshare'] / n, s['dshare'] / n, s['d'] / n,
                   s['elo'], s['auto_n'] / n, sm(s['auto_w'], s['auto_n']), lvl_delta, mates_on_team,
                   s['ewm'], c[0] / n] + list(s['ew']) + \
                  [s['relo'].get(role, 0.0), s['elo2'], s['opp'] / n,
                   (s['cperf'][champ][1] / s['cperf'][champ][0]) if champ in s['cperf'] and s['cperf'][champ][0] else np.nan,
                   (s['cperf'][champ][2] / s['cperf'][champ][0]) if champ in s['cperf'] and s['cperf'][champ][0] else np.nan,
                   math.log1p((now - s['clast'][champ]) / 3.6e6) if champ in s['clast'] else np.nan] + \
                  [sum(1 for x in s['ctimes'].get(champ, ()) if now - x < 7 * 864e5),
                   sum(1 for x in s['ctimes'].get(champ, ()) if now - x < 30 * 864e5),
                   pool_rank(s, champ), sum(1 for x in s['rec_roles'] if x == role) / max(1, len(s['rec_roles'])),
                   len(set(s['rec_champs'])), curve_wr(champ, c[0]) - g_wr]


PNAMES = (['pref_status', 'n_prefs', 'log_level'] + [f'spell_{v}' for v in SPELLS.values()] +
          ['keystone', 'primary_style', 'sub_style', 'champ_role_wr', 'champ_role_log_n', 'meta_wr', 'meta_log_n',
           'log_n', 'wr', 'champ_wr', 'champ_wr_k3', 'champ_log_n', 'role_wr', 'role_frac', 'recent20_wr',
           'recent5_wr', 'streak', 'log_hours_since', 'games_24h', 'games_7d', 'kda', 'cspm', 'gpm', 'vpm',
           'dpm', 'gold_share', 'dmg_share', 'deaths_pg', 'elo', 'autofill_rate', 'autofill_wr',
           'level_delta', 'premade_mates', 'ewm_wr', 'champ_frac', 'ew_win', 'ew_kda', 'ew_gshare', 'ew_dshare',
           'ew_vpm', 'ew_cspm', 'ew_deaths', 'role_elo', 'elo_slow', 'opp_strength', 'champ_gshare', 'champ_kda',
           'log_hours_since_champ', 'champ_games_7d', 'champ_games_30d', 'pool_rank', 'role_recent_frac',
           'pool_size20', 'champ_curve_delta'])

rows, ys, ids, splits, times = [], [], [], [], []
for m in matches:
    if m['creation'] >= test_start:
        break
    now = m['creation']
    while pending and pending[0][0] < now:
        _, _, done = heapq.heappop(pending)
        apply_result(done)
    if m['match_id'] in target_split:
        t = team_of(m)
        if all(x is not None for side in t.values() for x in side):
            feats = {}
            for tid, side in t.items():
                ids_team = [p[F['puuid']] for p in side]
                out = []
                for r, p in enumerate(side):
                    s = players.get(p[F['puuid']])
                    mates = sum(1 for q in ids_team if q != p[F['puuid']] and s is not None and s['mates'].get(q, 0) >= 2)
                    out.append(player_feats(p, r, now, mates))
                feats[tid] = np.array(out, dtype=np.float64)
            lanes = []
            for r in range(5):
                a, b = t[100][r][F['championName']], t[200][r][F['championName']]
                l = lane.get((r, b, a), [0, 0])
                ln_, lw_ = dec_get(meta_lane, (r, b, a))
                lanes += [sm(l[1], l[0], k=20), math.log1p(l[0]), sm(lw_, ln_, k=20) - 0.5]
            rb = feats[100][:, PNAMES.index('elo')].sum(); rr = feats[200][:, PNAMES.index('elo')].sum()
            team_x = {}
            for tid, side in t.items():
                cs_ = sorted(p[F['championName']] for p in side)
                syn = [sm(pair.get((cs_[i], cs_[j]), [0, 0])[1], pair.get((cs_[i], cs_[j]), [0, 0])[0], k=30)
                       for i in range(5) for j in range(i + 1, 5)]
                ids_team = [p[F['puuid']] for p in side]
                duo_n = duo_w = duo_pairs = 0
                for i in range(5):
                    si = players.get(ids_team[i])
                    for j in range(i + 1, 5):
                        if si is not None and si['mates'].get(ids_team[j], 0) >= 2:
                            duo_pairs += 1; duo_n += si['mates'][ids_team[j]]; duo_w += si['mates_w'][ids_team[j]]
                cold = sum(1 for q in ids_team if q not in players or players[q]['n'] == 0)
                team_x[tid] = [float(np.mean(syn)), float(np.min(syn)), duo_pairs, sm(duo_w, duo_n), cold]
            extra = [r_ - b_ for r_, b_ in zip(team_x[200], team_x[100])]
            rows.append(np.concatenate([feats[100].ravel(), feats[200].ravel(), lanes,
                                        [m['queue'] == 440, rr - rb], extra]))
            ys.append(int(bool(t[200][0][F['win']])))
            ids.append(m['match_id']); splits.append(target_split[m['match_id']]); times.append(now)
    if m['end'] and m['start'] and m['end'] > m['creation']:
        serial += 1
        heapq.heappush(pending, (m['end'], serial, m))

names = [f'blue_{ROLES[r]}_{n}' for r in range(5) for n in PNAMES] + \
        [f'red_{ROLES[r]}_{n}' for r in range(5) for n in PNAMES] + \
        [f'lane_{ROLES[r]}_{k}' for r in range(5) for k in ('red_wr', 'log_n', 'meta_diffrb')] + ['flex_queue', 'elo_diffrb'] + \
        [f'{k}_diffrb' for k in ('syn_mean', 'syn_min', 'duo_pairs', 'duo_wr', 'cold_players')]
X = np.vstack(rows).astype(np.float32)
np.savez_compressed(OUT, X=X, y=np.array(ys), ids=np.array(ids), split=np.array(splits), t=np.array(times),
                    names=np.array(names), pnames=np.array(PNAMES))
print('rows', X.shape, 'train', splits.count('train'), 'val', splits.count('validation'), flush=True)
