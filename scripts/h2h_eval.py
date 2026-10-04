"""対戦比較（keiba.h2h）がどれだけ当たるか。

    python scripts/h2h_eval.py [レース数=600] [開始日=2025-01-01]

開始日以降のレースから無作為に選び、各レースの出走馬の全組み合わせについて
「そのレースより前のデータだけ」で先着を予想し、実際の着順と比べる。

出力
- 段階 × 推定着差 ごとの的中率（＝推定着差が示す側が実際に先着した割合）
- 同じ組み合わせを人気で予想した場合の的中率（比較の物差し）
- レース単位: 序列1位の馬の勝率・3着内率と、1番人気のそれ
- 段階ごとのロジスティックの傾き（h2h.LEVEL_SLOPE に入れる値）
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from keiba import h2h, store

N_RACES = int(sys.argv[1]) if len(sys.argv) > 1 else 600
START = sys.argv[2] if len(sys.argv) > 2 else '2025-01-01'

t0 = time.time()
races = store.read_table('races')
results = store.read_table('results')
hist = h2h.History(races, results)
print(f'索引 {len(hist.horse_races):,} 頭  {time.time() - t0:.0f}秒', flush=True)

races['date'] = pd.to_datetime(races['date'])
cand = races[(races['date'] >= START) & races['race_id'].astype(str).isin(hist.race_rows)]
sample = cand.sample(min(N_RACES, len(cand)), random_state=0).sort_values('date')
res = results.assign(race_id=results['race_id'].astype(str), horse_id=results['horse_id'].astype(str))
res = res[res['race_id'].isin(set(sample['race_id'].astype(str)))]

pairs, tops = [], []
for i, (rid, d) in enumerate(zip(sample['race_id'].astype(str), sample['date'])):
    rr = res[(res['race_id'] == rid) & res['finish_position'].notna()]
    if len(rr) < 2:
        continue
    table, pw = h2h.rank(hist, rr['horse_id'], d)
    pos = dict(zip(rr['horse_id'], rr['finish_position']))
    pop = dict(zip(rr['horse_id'], rr['popularity']))
    pw['a_won'] = [pos[a] < pos[b] for a, b in zip(pw['a'], pw['b'])]
    pw['a_pop_better'] = [pop[a] < pop[b] for a, b in zip(pw['a'], pw['b'])]
    pw['race_id'] = rid
    pairs.append(pw)
    top = table.iloc[0]['horse_id']
    fav = min(pop, key=pop.get)
    tops.append({'race_id': rid, 'top_pos': pos[top], 'fav_pos': pos[fav],
                 'top_is_fav': top == fav, 'top_known': table.iloc[0]['known'],
                 'field': len(rr)})
    if (i + 1) % 100 == 0:
        print(f'  {i + 1}/{len(sample)}  {time.time() - t0:.0f}秒', flush=True)
        hist.clear_cache()

pw = pd.concat(pairs, ignore_index=True)
pw = pw[pw['a_won'] != (pw['a'] == pw['b'])]   # 念のため
known = pw[pw['level'] > 0].copy()
# 推定が示す側が先着したか（着差0は除く）
known = known[known['margin'] != 0]
known['hit'] = (known['margin'] > 0) == known['a_won']
known['pop_hit'] = known['a_pop_better'] == known['a_won']
known['abs_m'] = known['margin'].abs()
known['bucket'] = pd.cut(known['abs_m'], [0, .2, .5, 1.0, 2.0, 99],
                         labels=['〜0.2秒', '0.2〜0.5', '0.5〜1.0', '1.0〜2.0', '2.0秒〜'])

level_name = {1: '1 直接対決', 2: '2 共通の相手', 3: '3 2頭はさむ'}
print(f'\n組み合わせ {len(pw):,}  レース {len(tops)}  {time.time() - t0:.0f}秒')
cov = pw['level'].value_counts(normalize=True).sort_index()
print('\n■ どの段階で比較できたか（全組み合わせに対する割合）')
for lv, r in cov.items():
    print(f'  {level_name.get(lv, "0 比較できず")}: {r:.1%}')

print('\n■ 段階ごとの的中率（推定で上とした馬が実際に先着した割合）')
g = known.groupby('level').agg(n=('hit', 'size'), hit=('hit', 'mean'), pop_hit=('pop_hit', 'mean'))
g.index = g.index.map(level_name)
print(g.to_string(float_format=lambda x: f'{x:.3f}'))

print('\n■ 段階 × 推定着差の大きさ')
g = known.groupby(['level', 'bucket']).agg(n=('hit', 'size'), hit=('hit', 'mean'),
                                           pop_hit=('pop_hit', 'mean'))
g = g[g['n'] > 0]
g.index = g.index.set_levels(g.index.levels[0].map(level_name), level=0)
print(g.to_string(float_format=lambda x: f'{x:.3f}'))

print('\n■ 段階1: 対戦回数と、前回までの先着の一貫性')
d1 = known[known['level'] == 1].copy()
d1['meet'] = pd.cut(d1['n'], [0, 1, 2, 99], labels=['1回', '2回', '3回以上'])
print(d1.groupby('meet').agg(n=('hit', 'size'), hit=('hit', 'mean'),
                             pop_hit=('pop_hit', 'mean')).to_string(float_format=lambda x: f'{x:.3f}'))

print('\n■ 段階ごとのロジスティック傾き k（P = 1/(1+exp(-k·着差))）')
for lv in (1, 2, 3):
    x = known.loc[known['level'] == lv, 'margin'].to_numpy()
    y = known.loc[known['level'] == lv, 'a_won'].to_numpy().astype(float)
    if len(x) < 50:
        continue
    k = 1.0
    for _ in range(50):   # ニュートン法（切片なし1変数）
        p = 1 / (1 + np.exp(-k * x))
        k += np.sum((y - p) * x) / max(np.sum(p * (1 - p) * x * x), 1e-9)
    print(f'  {level_name[lv]}: k = {k:.2f}   （1秒差で {1 / (1 + np.exp(-k)):.1%}）')

t = pd.DataFrame(tops)
print('\n■ レース単位: 序列1位の馬 vs 1番人気')
print(f'  序列1位   勝率 {(t["top_pos"] == 1).mean():.1%}  3着内 {(t["top_pos"] <= 3).mean():.1%}')
print(f'  1番人気   勝率 {(t["fav_pos"] == 1).mean():.1%}  3着内 {(t["fav_pos"] <= 3).mean():.1%}')
print(f'  序列1位が1番人気と同じ馬: {t["top_is_fav"].mean():.1%}')
nf = t[~t['top_is_fav']]
print(f'  一致しないレースでの序列1位  勝率 {(nf["top_pos"] == 1).mean():.1%}  '
      f'3着内 {(nf["top_pos"] <= 3).mean():.1%}  (n={len(nf)})')
