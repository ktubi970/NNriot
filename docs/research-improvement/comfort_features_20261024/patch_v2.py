import shutil
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
shutil.copy('build_features.py', 'build_features_v2.py')
p = 'build_features_v2.py'
s = open(p).read()
rep = [
("""            'mates': Counter(), 'ewm': 0.5}""",
 """            'mates': Counter(), 'ewm': 0.5, 'mates_w': Counter(), 'ew': None, 'relo': defaultdict(float),
            'cperf': defaultdict(lambda: [0, 0.0, 0.0]), 'clast': {}, 'opp': 0.0, 'elo2': 0.0}"""),
("""lane = defaultdict(lambda: [0, 0])                 # (role, champ, opp) -> [n, w for champ]""",
 """lane = defaultdict(lambda: [0, 0])                 # (role, champ, opp) -> [n, w for champ]
pair = defaultdict(lambda: [0, 0])                 # sorted (champ, champ) same team -> [n, w]"""),
("""    exp_red = 1 / (1 + math.exp(-(rr - rb)))""",
 """    exp_red = 1 / (1 + math.exp(-(rr - rb)))
    rb_r = sum(players[p[F['puuid']]]['relo'][ROLES[r]] for r, p in enumerate(t[100])) / 5
    rr_r = sum(players[p[F['puuid']]]['relo'][ROLES[r]] for r, p in enumerate(t[200])) / 5
    exp_red_r = 1 / (1 + math.exp(-(rr_r - rb_r)))
    rb2 = sum(players[p[F['puuid']]]['elo2'] for p in t[100]) / 5
    rr2 = sum(players[p[F['puuid']]]['elo2'] for p in t[200]) / 5
    exp_red2 = 1 / (1 + math.exp(-(rr2 - rb2)))"""),
("""            kf = 0.6 / math.sqrt(1 + s['n'] / 10)
            s['elo'] += kf * ((1 if win else 0) - (exp_red if tid == 200 else 1 - exp_red))
            for q in side:
                if q is not p:
                    s['mates'][q[F['puuid']]] += 1""",
 """            kf = 0.6 / math.sqrt(1 + s['n'] / 10)
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
                    s['mates_w'][q[F['puuid']]] += win"""),
("""    for r in range(5):
        a, b = t[100][r][F['championName']], t[200][r][F['championName']]
        l1 = lane[(r, a, b)]""",
 """    for tid, side in t.items():
        win = red_win if tid == 200 else not red_win
        cs_ = sorted(p[F['championName']] for p in side)
        for i in range(5):
            for j in range(i + 1, 5):
                pr = pair[(cs_[i], cs_[j])]; pr[0] += 1; pr[1] += win
    for r in range(5):
        a, b = t[100][r][F['championName']], t[200][r][F['championName']]
        l1 = lane[(r, a, b)]"""),
("""    if s is None or s['n'] == 0:
        return base + [0, 0.5, 0.5, 0.5, 0, 0.5, 0, 0.5, 0.5, 0, -1, 0, 0, np.nan, np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan, 0.0, 0, 0.5, np.nan, mates_on_team, 0.5, 0]""",
 """    if s is None or s['n'] == 0:
        return base + [0, 0.5, 0.5, 0.5, 0, 0.5, 0, 0.5, 0.5, 0, -1, 0, 0, np.nan, np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan, 0.0, 0, 0.5, np.nan, mates_on_team, 0.5, 0] + \\
            [np.nan] * 7 + [0.0, 0.0, np.nan, np.nan, np.nan, np.nan]"""),
("""                   s['elo'], s['auto_n'] / n, sm(s['auto_w'], s['auto_n']), lvl_delta, mates_on_team,
                   s['ewm'], c[0] / n]""",
 """                   s['elo'], s['auto_n'] / n, sm(s['auto_w'], s['auto_n']), lvl_delta, mates_on_team,
                   s['ewm'], c[0] / n] + list(s['ew']) + \\
                  [s['relo'].get(role, 0.0), s['elo2'], s['opp'] / n,
                   (s['cperf'][champ][1] / s['cperf'][champ][0]) if champ in s['cperf'] and s['cperf'][champ][0] else np.nan,
                   (s['cperf'][champ][2] / s['cperf'][champ][0]) if champ in s['cperf'] and s['cperf'][champ][0] else np.nan,
                   math.log1p((now - s['clast'][champ]) / 3.6e6) if champ in s['clast'] else np.nan]"""),
("""           'level_delta', 'premade_mates', 'ewm_wr', 'champ_frac'])""",
 """           'level_delta', 'premade_mates', 'ewm_wr', 'champ_frac', 'ew_win', 'ew_kda', 'ew_gshare', 'ew_dshare',
           'ew_vpm', 'ew_cspm', 'ew_deaths', 'role_elo', 'elo_slow', 'opp_strength', 'champ_gshare', 'champ_kda',
           'log_hours_since_champ'])"""),
("""            rb = feats[100][:, PNAMES.index('elo')].sum(); rr = feats[200][:, PNAMES.index('elo')].sum()
            rows.append(np.concatenate([feats[100].ravel(), feats[200].ravel(), lanes,
                                        [m['queue'] == 440, rr - rb]]))""",
 """            rb = feats[100][:, PNAMES.index('elo')].sum(); rr = feats[200][:, PNAMES.index('elo')].sum()
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
                                        [m['queue'] == 440, rr - rb], extra]))"""),
("""[f'lane_{ROLES[r]}_{k}' for r in range(5) for k in ('red_wr', 'log_n')] + ['flex_queue', 'elo_diff']""",
 """[f'lane_{ROLES[r]}_{k}' for r in range(5) for k in ('red_wr', 'log_n')] + ['flex_queue', 'elo_diffrb'] + \\
        [f'{k}_diffrb' for k in ('syn_mean', 'syn_min', 'duo_pairs', 'duo_wr', 'cold_players')]"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, 'w').write(s)
e = open('evaluate2.py').read()
a = """            elif n == 'elo_diff':
                tail[:, i] = -tail[:, i]"""
assert e.count(a) == 1
e = e.replace(a, """            elif n.endswith('_diffrb') or n == 'elo_diff':
                tail[:, i] = -tail[:, i]""")
open('evaluate2.py', 'w').write(e)
print('ok')
