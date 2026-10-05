"""目線ごとの偏差値と、実際の3着内率の関係（市場あり・市場なしを除く）。

    python scripts/views_eval.py [開始日=2025-01-01]

どの値も、そのレースより前の情報だけで作る。レースレベルは週ごとに解き直す。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from keiba import figure, h2h, market, race_level, store, views

t0 = time.time()
start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else '2025-01-01')
races, results = store.load_for_features()
fig = figure.add_horse_figure_features(figure.add_figures(results, races))
fig['race_id'] = fig['race_id'].astype(str)
fig = fig[fig['race_date'] >= start]
base = fig[['race_id', 'horse_number', 'horse_id', 'race_date', 'finish_position', 'popularity',
            'fig_r3', 'fig_best', 'l3fig_r3']].copy()
base['horse_id'] = base['horse_id'].astype(str)
base['fin'] = pd.to_numeric(base['finish_position'], errors='coerce')
base['pop'] = pd.to_numeric(base['popularity'], errors='coerce')
base = base.dropna(subset=['fin', 'pop'])
base['n'] = base.groupby('race_id')['horse_number'].transform('size')
base = base[base['n'] >= 8].rename(columns={'fig_r3': '指数・近3走', 'fig_best': '指数・最高', 'l3fig_r3': '上がり指数・近3走'})
print(f'対象 {base["race_id"].nunique():,} レース  指数 {time.time() - t0:.0f}秒', flush=True)

# 市場の見込み
o = store.read_table('odds'); o['race_id'] = o['race_id'].astype(str)
o = o[o['race_id'].isin(set(base['race_id']))]
mk = []
for rid, g in o.groupby('race_id'):
    if g[['win', 'place_min', 'place_max', 'win_pop']].notna().all().all():
        mk.append(pd.DataFrame({'race_id': rid, 'horse_number': g['horse_number'],
                                '市場の見込み': market.market_top3(g['win'], g['place_min'], g['place_max'], g['win_pop'])}))
base = base.merge(pd.concat(mk), on=['race_id', 'horse_number'], how='left')

# レースレベル（週ごとに解き直す）
raw_r, raw_res = store.read_table('races'), store.read_table('results')
runs = race_level.prepare_runs(raw_r, raw_res)
base['week'] = base['race_date'].dt.to_period('W-SUN').dt.start_time
parts = []
for wk, g in base.groupby('week'):
    level, ability = race_level.fit(runs, wk)
    past = runs[(runs['date'] < wk) & runs['horse_id'].isin(set(g['horse_id']))]
    prev = past.sort_values('date').groupby('horse_id').tail(1).set_index('horse_id')
    perf = prev['race_id'].map(level) - prev['behind']
    parts.append(g[['race_id', 'horse_number']].assign(
        前走パフォーマンス=g['horse_id'].map(perf).to_numpy(), 能力=g['horse_id'].map(ability).to_numpy()))
base = base.merge(pd.concat(parts), on=['race_id', 'horse_number'], how='left')
print(f'レースレベル {time.time() - t0:.0f}秒', flush=True)

# 対戦比較（レースごと）
hist = h2h.History(raw_r, raw_res)
sc = []
for i, (rid, g) in enumerate(base.groupby('race_id')):
    table, _ = h2h.rank(hist, list(g['horse_id']), g['race_date'].iloc[0])
    s = table[table['known'] > 0].set_index('horse_id')['score']   # 比較できる相手がいない馬は空欄
    sc.append(g[['race_id', 'horse_number']].assign(対戦比較=g['horse_id'].map(s).to_numpy()))
    if i % 1000 == 0:
        hist.clear_cache(); print(f'  対戦比較 {i:,} レース {time.time() - t0:.0f}秒', flush=True)
base = base.merge(pd.concat(sc), on=['race_id', 'horse_number'], how='left')
base['top3'] = (base['fin'] <= 3).astype(int)

V = ['指数・近3走', '指数・最高', '上がり指数・近3走', '市場の見込み', '前走パフォーマンス', '能力', '対戦比較']
for v in V:
    base[f'{v}_dev'] = base.groupby('race_id')[v].transform(lambda x: views.deviation(x))
    base[f'{v}_rank'] = base.groupby('race_id')[v].rank(ascending=False, method='min')
base.to_pickle(sys.argv[2]) if len(sys.argv) > 2 else None

BANDS = [-np.inf, 40, 45, 50, 55, 60, 65, 70, np.inf]
LAB = ['〜40', '40〜45', '45〜50', '50〜55', '55〜60', '60〜65', '65〜70', '70〜']
print(f'\n全体の3着内率 {base["top3"].mean() * 100:.1f}%\n')
print('【1】偏差値の帯ごとの実際の3着内率（括弧は頭数）')
for v in V:
    x = base.dropna(subset=[f'{v}_dev'])
    t = x.groupby(pd.cut(x[f'{v}_dev'], BANDS, labels=LAB))['top3'].agg(['mean', 'size'])
    print(f"  {v:12s} AUC {roc_auc_score(x['top3'], x[v]):.3f}  値あり{len(x) / len(base) * 100:3.0f}%  "
          + '  '.join(f"{b}:{r['mean'] * 100:4.1f}%({int(r['size']):,})" for b, r in t.iterrows()), flush=True)

print('\n【2】目線の1番手・2番手・3番手の3着内率')
for v in V:
    x = base.dropna(subset=[f'{v}_rank'])
    print(f"  {v:12s} " + '  '.join(f"{k}番手 {x[x[f'{v}_rank'] == k]['top3'].mean() * 100:5.1f}%" for k in (1, 2, 3)))

print('\n【3】人気4番以下で、その目線3番手以内の馬（同じ人気の他の馬と比べる）')
pop4 = base[base['pop'] >= 4]
for v in V:
    s, o_ = pop4[pop4[f'{v}_rank'] <= 3], pop4[~(pop4[f'{v}_rank'] <= 3)]
    w = s['pop'].value_counts(normalize=True)
    exp = float((w * o_.groupby('pop')['top3'].mean().reindex(w.index)).sum())
    print(f"  {v:12s} {len(s):>6,}頭  3着内 {s['top3'].mean() * 100:5.1f}%  同じ人気の他の馬 {exp * 100:5.1f}%  差 {(s['top3'].mean() - exp) * 100:+.1f}pt")
print(f'\n合計 {time.time() - t0:.0f}秒')
