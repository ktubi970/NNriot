import os
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))
shutil.copy('build_features_v3.py', 'build_features_v4.py')
p = 'build_features_v4.py'
s = open(p).read()
rep = [
("""curve = defaultdict(lambda: [0, 0])                # (champ, experience bucket) -> [n, w]""",
 """curve = defaultdict(lambda: [0, 0])                # (champ, experience bucket) -> [n, w]
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
    return n * f, w * f"""),
("""    remake = any(p[F['gameEndedInEarlySurrender']] for p in m['parts'])
    if remake:
        return""",
 """    remake = any(p[F['gameEndedInEarlySurrender']] for p in m['parts'])
    if remake:
        return
    clock[0] += 1"""),
("""            g = champ_glob[(p[F['championName']], ROLES[r])]; g[0] += 1; g[1] += win""",
 """            g = champ_glob[(p[F['championName']], ROLES[r])]; g[0] += 1; g[1] += win
            dec_add(meta, (p[F['championName']], ROLES[r]), win)"""),
("""        l2 = lane[(r, b, a)]; l2[0] += 1; l2[1] += red_win""",
 """        l2 = lane[(r, b, a)]; l2[0] += 1; l2[1] += red_win
        dec_add(meta_lane, (r, b, a), red_win)"""),
("""    base = [st, npref, math.log1p(level)] + [float(sid in spells) for sid in SPELLS] + \\
           [p[F['keystone']] or -1, p[F['primary_style']] or -1, p[F['sub_style']] or -1,
            sm(g[1], g[0], k=50), math.log1p(g[0])]""",
 """    mn, mw = dec_get(meta, (champ, role))
    base = [st, npref, math.log1p(level)] + [float(sid in spells) for sid in SPELLS] + \\
           [p[F['keystone']] or -1, p[F['primary_style']] or -1, p[F['sub_style']] or -1,
            sm(g[1], g[0], k=50), math.log1p(g[0]), sm(mw, mn, k=30), math.log1p(mn)]"""),
("""PNAMES = (['pref_status', 'n_prefs', 'log_level'] + [f'spell_{v}' for v in SPELLS.values()] +
          ['keystone', 'primary_style', 'sub_style', 'champ_role_wr', 'champ_role_log_n',""",
 """PNAMES = (['pref_status', 'n_prefs', 'log_level'] + [f'spell_{v}' for v in SPELLS.values()] +
          ['keystone', 'primary_style', 'sub_style', 'champ_role_wr', 'champ_role_log_n', 'meta_wr', 'meta_log_n',"""),
("""                l = lane.get((r, b, a), [0, 0])
                lanes += [sm(l[1], l[0], k=20), math.log1p(l[0])]""",
 """                l = lane.get((r, b, a), [0, 0])
                ln_, lw_ = dec_get(meta_lane, (r, b, a))
                lanes += [sm(l[1], l[0], k=20), math.log1p(l[0]), sm(lw_, ln_, k=20) - 0.5]"""),
("""[f'lane_{ROLES[r]}_{k}' for r in range(5) for k in ('red_wr', 'log_n')]""",
 """[f'lane_{ROLES[r]}_{k}' for r in range(5) for k in ('red_wr', 'log_n', 'meta_diffrb')]"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('ok')
