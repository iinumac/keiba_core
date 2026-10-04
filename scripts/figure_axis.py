"""軸の選び方：モデル1位の指数がメンバー下位なら、次点に替えると当たりやすいか。

    python scripts/figure_axis.py

対象は 2025・2026 年（モデルは2024年までで学習）。
"""

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from keiba import betting as bt, figure, segments as sg, stages, store
from keiba.features import build_features, C04_CONFIG

t0 = time.time()
races, results = store.load_for_features()
df = build_features(races, results, C04_CONFIG)
df['race_id'] = df['race_id'].astype(str)
df = df[df['year'] >= 2025].copy()
fig = figure.add_horse_figure_features(figure.add_figures(results, races))
fig['race_id'] = fig['race_id'].astype(str)
df = df.merge(fig[['race_id', 'horse_number', 'fig_r3_rank', 'fig_best_rank']],
              on=['race_id', 'horse_number'], how='left')
df['_over'] = sg.overvalued(df); df['_thin'] = sg.thin_record(df)
bundle = pickle.load(open(ROOT / 'models' / 'model_strategy.pkl', 'rb'))
df['score'] = bundle['model'].predict(stages.model_input(df, bundle['features']))
df['fin'] = pd.to_numeric(df['finish_position'], errors='coerce')
pay = store.read_table('payouts'); pay['race_id'] = pay['race_id'].astype(str)
IDX = bt.payout_index(pay)
print(f'準備 {time.time() - t0:.0f}秒', flush=True)

def pick_axis(g, col):
    """モデル順に見て、指数がメンバー6位以下の馬を飛ばした最初の馬。"""
    if col is None:
        return g.iloc[0]
    for _, r in g.iterrows():
        if not (r[col] >= 6):          # 指数が無い馬（デビュー戦など）は飛ばさない
            return r
    return g.iloc[0]

def run(col, rule_only, year):
    rec = []
    for rid, g in df[df['year'] == year].groupby('race_id'):
        if len(g) < 8:
            continue
        g = g.sort_values('score', ascending=False)
        if rule_only:
            top = g.head(3)
            if top['_over'].any() or top['_thin'].any() or g['score'].head(3).sum() < 1.55:
                continue
        ax = pick_axis(g, col)
        partners = [int(h) for h in g['horse_number'] if h != ax['horse_number']][:3]
        a = int(ax['horse_number'])
        combos = {frozenset([a, p]) for p in partners}
        ret = {b: sum(v for c, v in IDX[b].get(rid, []) if c in combos) for b in ('ワイド', '馬連')}
        rec.append(dict(changed=a != int(g['horse_number'].iloc[0]), win=ax['fin'] == 1, top3=ax['fin'] <= 3,
                        wide=ret['ワイド'] > 0, umaren=ret['馬連'] > 0, wide_ret=ret['ワイド'], umaren_ret=ret['馬連']))
    r = pd.DataFrame(rec)
    return (f"{len(r):>5,}R 替えた{r['changed'].mean()*100:4.1f}%  軸の勝率{r['win'].mean()*100:5.1f}% 3着内{r['top3'].mean()*100:5.1f}%"
            f"  ワイド的中{r['wide'].mean()*100:5.1f}% 回収{r['wide_ret'].sum()/(len(r)*300)*100:5.1f}%"
            f"  馬連的中{r['umaren'].mean()*100:5.1f}% 回収{r['umaren_ret'].sum()/(len(r)*300)*100:5.1f}%")

for rule_only, lab in ((False, '全レース（8頭以上）'), (True, '今のルールで馬連を買うレース（除外なし・確信度1.55以上）')):
    print(f'\n■ {lab}', flush=True)
    for name, col in (('A 今のまま', None), ('B 指数(近3走)6位以下なら次点', 'fig_r3_rank'),
                      ("B' 指数(最高)6位以下なら次点", 'fig_best_rank')):
        for y in (2025, 2026):
            print(f'  {name:24s} {y}  {run(col, rule_only, y)}', flush=True)
print(f'\n合計 {time.time() - t0:.0f}秒')
