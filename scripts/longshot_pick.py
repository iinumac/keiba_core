"""人気薄（7番人気以下）の中から、来る馬を1〜2頭に絞れるか。

    python scripts/longshot_pick.py

【1】人気薄の中での選び方を比べる（拾えた率・選んだ馬の3着内率・複勝回収率）
【2】選んだ人気薄を三連複の相手に入れたとき、同じ点数の今のやり方と比べた的中率・回収率
学習は2024年まで、検証は2025・2026年。
"""

import itertools
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import numpy as np
import pandas as pd

from keiba import betting as bt, figure, segments as sg, stages, store
from keiba.features import build_features, C04_CONFIG

t0 = time.time()
races, results = store.load_for_features()
df = build_features(races, results, C04_CONFIG)
df['race_id'] = df['race_id'].astype(str)
fig = figure.add_horse_figure_features(figure.add_figures(results, races))
fig['race_id'] = fig['race_id'].astype(str)
FIG = [c for c in fig.columns if c.startswith(('fig_', 'l3fig_'))]
df = df.merge(fig[['race_id', 'horse_number'] + FIG], on=['race_id', 'horse_number'], how='left')
df['pop'] = pd.to_numeric(df['popularity'], errors='coerce')
df['fin'] = pd.to_numeric(df['finish_position'], errors='coerce')
df['n'] = df.groupby('race_id')['horse_number'].transform('size')
df = df[df['n'] >= 8].copy()
# 人気薄の中での位置づけ
ls = df['pop'] >= 7
df['ls_odds_rank'] = df[ls].groupby('race_id')['odds'].rank(method='min')
df['ls_count'] = ls.groupby(df['race_id']).transform('sum')
print(f'準備 {time.time() - t0:.0f}秒', flush=True)

bundle = pickle.load(open(ROOT / 'models' / 'model_strategy.pkl', 'rb'))
df['s_model'] = bundle['model'].predict(stages.model_input(df, bundle['features']))
tr_all = df[df['year'] <= 2024]
F_free = [c for c in dict.fromkeys(stages.MARKET_FREE_FEATURES) if c in df.columns]
m_free = lgb.train(stages.STRATEGY_PARAMS, lgb.Dataset(stages.model_input(tr_all, F_free), tr_all['is_top3']),
                   num_boost_round=stages.STRATEGY_ROUNDS)
df['s_free'] = m_free.predict(stages.model_input(df, F_free))
F_ls = stages.strategy_feature_list(df.columns) + FIG + ['ls_odds_rank', 'ls_count']
tr_ls = df[(df['year'] <= 2024) & ls]
m_ls = lgb.train(stages.STRATEGY_PARAMS, lgb.Dataset(stages.model_input(tr_ls, F_ls), tr_ls['is_top3']),
                 num_boost_round=stages.STRATEGY_ROUNDS)
df['s_ls'] = np.where(ls, m_ls.predict(stages.model_input(df, F_ls)), np.nan)
df['s_fig'] = df['fig_r3_gap']
df['s_base'] = -df['odds']
imp = pd.Series(m_ls.feature_importance('gain'), index=F_ls).sort_values(ascending=False)
print(f'学習 {time.time() - t0:.0f}秒  人気薄専用モデルの重要度上位: ' + ', '.join(imp.index[:10]), flush=True)

te = df[df['year'] >= 2025].copy()
pay = store.read_table('payouts'); pay['race_id'] = pay['race_id'].astype(str)
IDX = bt.payout_index(pay)
FUKU = {(r, int(h[0])): v for r, h, v in zip(pay[pay['bet_type'] == '複勝']['race_id'],
                                            pay[pay['bet_type'] == '複勝']['horse_numbers'],
                                            pay[pay['bet_type'] == '複勝']['payout'])}
SCORERS = {'基準：人気薄の中で一番人気': 's_base', '市場ありモデル': 's_model', '市場なしモデル': 's_free',
           'スピード指数（トップとの差）': 's_fig', '人気薄専用モデル': 's_ls'}

def picks(g, col, k):
    c = g[g['pop'] >= 7].dropna(subset=[col]) if col != 's_fig' else g[g['pop'] >= 7].fillna({col: -99})
    return c.sort_values(col, ascending=False)['horse_number'].astype(int).tolist()[:k]

print('\n【1】人気薄から選ぶ力（「拾えた率」= 人気薄が3着内に来たレースのうち、選んだ馬がその馬だった割合）')
for y in (2025, 2026):
    t = te[te['year'] == y]
    print(f'  {y}年')
    for name, col in SCORERS.items():
        res = {1: [], 2: []}
        for rid, g in t.groupby('race_id'):
            came = set(g.loc[(g['pop'] >= 7) & (g['fin'] <= 3), 'horse_number'].astype(int))
            for k in (1, 2):
                p = picks(g, col, k)
                res[k].append(dict(any=bool(came), caught=bool(came & set(p)), n=len(p),
                                   hits=len(came & set(p)), fuku=sum(FUKU.get((rid, h), 0) for h in p)))
        line = f'    {name:18s}'
        for k in (1, 2):
            r = pd.DataFrame(res[k])
            line += (f'  {k}頭: 拾えた率 {r[r["any"]]["caught"].mean()*100:5.1f}%  選んだ馬の3着内率 {r["hits"].sum()/r["n"].sum()*100:5.1f}%'
                     f'  複勝回収 {r["fuku"].sum()/(r["n"].sum()*100)*100:5.1f}%')
        print(line, flush=True)

print('\n【2】三連複フォーメーション（軸＝市場ありモデル1位）')
def formation(g, k, col):
    g = g.sort_values('s_model', ascending=False)
    o = g['horse_number'].astype(int).tolist()
    axis = o[0]
    if col is None:
        partners = o[1:1 + 2 + k]
    else:
        ls_pick = [h for h in picks(g, col, k + 3) if h != axis][:k]
        partners = [h for h in o[1:] if h not in ls_pick][:2] + ls_pick
    return {frozenset([axis, a, b]) for a, b in itertools.combinations(partners, 2)}

for k, pts in ((1, 3), (2, 6)):
    print(f'  ■ 相手 上位人気2頭＋人気薄{k}頭（{pts}点） vs モデル2〜{2 + k + 1}位（{pts}点）')
    for name, col in [('今のやり方（モデル順）', None)] + [(n, c) for n, c in SCORERS.items()]:
        line = f'    {name:18s}'
        for y in (2025, 2026):
            hits = ret = n = 0
            for rid, g in te[te['year'] == y].groupby('race_id'):
                combos = formation(g, k, col)
                got = sum(v for c, v in IDX['三連複'].get(rid, []) if c in combos)
                hits += got > 0; ret += got; n += 1
            line += f'  {y} 的中 {hits/n*100:5.1f}% 回収 {ret/(n*pts*100)*100:5.1f}%'
        print(line, flush=True)
print(f'\n合計 {time.time() - t0:.0f}秒')
