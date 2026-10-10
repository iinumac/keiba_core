"""指数の距離の変換を学び、確かめる。

    python scripts/figure_transfer.py

同じ馬が同じ芝ダで距離を変えた2走の組（間隔365日以内）から、
「元の距離の指数 → 今回の距離の指数」の係数を2024年までで学び、models/figure_transfer.json に保存する。

2025年以降の出走で、指数（近3走・最高）の作り方を3通り比べる。

  全走歴      芝ダ・距離を問わず全過去走（2026-10-10 まで views で使っていた方式）
  ±400m      同じ芝ダで距離 ±400m 以内、無ければ同じ芝ダの全距離（2026-10-10 のつなぎ）
  距離変換    同じ芝ダの全過去走を今回の距離に変換してから集計（この方式）

比べるもの:
  【1】今回のレースの実際の指数との近さ（相関）
  【2】3着内との関係（AUC、レース内1位の3着内率）
  【3】単勝オッズからの見込みに足したときの上積み（2025年で式を当て、2026年で検証）
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from keiba import config, figure, market, store

t0 = time.time()
races, results = store.load_for_features()
fig = figure.add_figures(results, races)
fig['horse_id'] = fig['horse_id'].astype(str)
fig['race_id'] = fig['race_id'].astype(str)
print(f'指数 {time.time() - t0:.0f}秒', flush=True)

# 学習（今回の走りが2024年まで）
pairs = figure.transfer_pairs(fig[fig['race_date'] < '2025-01-01'])
transfer = figure.fit_distance_transfer(pairs)
out = config.MODEL_DIR / 'figure_transfer.json'
out.write_text(json.dumps({'train_end': '2024-12-31', 'max_gap_days': figure.TRANSFER_MAX_GAP_DAYS,
                           'dist_labels': figure.DIST_LABELS, **transfer},
                          ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(f'学習 {len(pairs):,}組  保存: {out}  {time.time() - t0:.0f}秒\n', flush=True)

for col in ('figure',):
    cell = pd.DataFrame(transfer[col]['cell'])
    for surf in ('芝', 'ダート'):
        c = cell[cell['surface'] == surf]
        lab = dict(enumerate(figure.DIST_LABELS))
        print(f'■ {surf}：元の指数がどれだけ当てになるか（b。1に近いほどそのまま使える）。行=元の距離帯、列=今回の距離帯')
        print(c.pivot(index='band_s', columns='band_t', values='b').rename(index=lab, columns=lab).round(2).to_string())
        print(f'  平均のずれ（元の指数0のときの今回の指数 a）')
        print(c.pivot(index='band_s', columns='band_t', values='a').rename(index=lab, columns=lab).round(2).to_string())
        print(f'  組数')
        print(c.pivot(index='band_s', columns='band_t', values='n').rename(index=lab, columns=lab).fillna(0).astype(int).to_string(), '\n')

# 検証（2025年以降の出走）
tg = fig[(fig['race_date'] >= '2025-01-01') & fig['surface'].isin(['芝', 'ダート'])].copy()
tg['fin'] = pd.to_numeric(tg['finish_position'], errors='coerce')
tg['n'] = tg.groupby('race_id')['horse_number'].transform('size')
tg = tg[tg['fin'].notna() & (tg['n'] >= 8)]
key = tg[['race_id', 'horse_number', 'horse_id', 'race_date', 'surface', 'distance', 'fin', 'odds', 'figure']]
key = key.rename(columns={'figure': 'actual'}).reset_index(drop=True)

conv = figure.converted_summary(key[['horse_id', 'race_date', 'surface', 'distance']], fig, transfer)
key['距離変換_r3'], key['距離変換_best'] = conv['fig_r3'].to_numpy(), conv['fig_best'].to_numpy()

# 全走歴・±400m
h = fig[['horse_id', 'race_date', 'surface', 'distance', 'figure']].dropna(subset=['figure'])
j = key[['horse_id', 'race_date', 'surface', 'distance']].reset_index().merge(
    h.rename(columns={'race_date': 'd', 'surface': 's', 'distance': 'dist'}), on='horse_id')
j = j[j['d'] < j['race_date']].sort_values(['index', 'd'])
same = j['s'] == j['surface']
near = same & ((pd.to_numeric(j['dist']) - pd.to_numeric(j['distance'])).abs() <= 400)


def agg(m):
    g = j[m].groupby('index')['figure']
    return g.apply(lambda x: x.tail(3).mean()), g.max()


r3a, ba = agg(same | ~same)
r3n, bn = agg(near)
r3s, bs = agg(same)
key['全走歴_r3'], key['全走歴_best'] = r3a.reindex(key.index), ba.reindex(key.index)
key['±400m_r3'] = r3n.reindex(key.index).fillna(r3s.reindex(key.index))
key['±400m_best'] = bn.reindex(key.index).fillna(bs.reindex(key.index))
key['top3'] = (key['fin'] <= 3).astype(int)
mk = []
for rid, g in key.groupby('race_id'):
    o = pd.to_numeric(g['odds'], errors='coerce').to_numpy(float)
    if np.isfinite(o).all() and (o > 0).all():
        mk.append(pd.Series(market.top3_from_support(o), index=g.index))
key['mkt'] = pd.concat(mk)
key = key[key['mkt'].notna()]
print(f'検証 {key["race_id"].nunique():,}レース {len(key):,}頭  {time.time() - t0:.0f}秒\n')

lg = lambda p: np.log(np.clip(p, 1e-4, .9999) / (1 - np.clip(p, 1e-4, .9999)))
tr, te = key['race_date'] < '2026-01-01', key['race_date'] >= '2026-01-01'
base = LogisticRegression().fit(lg(key.loc[tr, ['mkt']]), key.loc[tr, 'top3'])
ll0 = log_loss(key.loc[te, 'top3'], base.predict_proba(lg(key.loc[te, ['mkt']]))[:, 1])


def gain(col):
    z = key.groupby('race_id')[col].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)
    X = np.c_[lg(key['mkt']), z.fillna(0), z.isna()]
    m = LogisticRegression(C=1e4, max_iter=2000).fit(X[tr], key.loc[tr, 'top3'])
    return (ll0 - log_loss(key.loc[te, 'top3'], m.predict_proba(X[te])[:, 1])) * 1e5


print('                  値あり  実際の指数との相関  3着内AUC  レース内1位の3着内率  市場に足した上積み(×1e-5)')
for kind in ('全走歴', '±400m', '距離変換'):
    for v, lab in (('r3', '近3走'), ('best', '最高')):
        c = f'{kind}_{v}'
        x = key[key[c].notna()]
        both = x[x['actual'].notna()]
        rk = x.groupby('race_id')[c].rank(ascending=False, method='first')
        print(f'  {kind:6s} {lab:4s}  {key[c].notna().mean() * 100:4.0f}%   {np.corrcoef(both[c], both["actual"])[0, 1]:.3f}'
              f'            {roc_auc_score(x["top3"], x[c]):.4f}   {x.loc[rk == 1, "top3"].mean() * 100:5.1f}%'
              f'                {gain(c):+7.1f}')

print('\n全方式に値がある馬だけ（同じ頭で比べる）')
common = key[[f'{k}_{v}' for k in ('全走歴', '±400m', '距離変換') for v in ('r3', 'best')]].notna().all(axis=1)
kc = key[common]
for kind in ('全走歴', '±400m', '距離変換'):
    for v, lab in (('r3', '近3走'), ('best', '最高')):
        c = f'{kind}_{v}'
        both = kc[kc['actual'].notna()]
        rk = kc.groupby('race_id')[c].rank(ascending=False, method='first')
        print(f'  {kind:6s} {lab:4s}  {len(kc):,}頭  実際の指数との相関 {np.corrcoef(both[c], both["actual"])[0, 1]:.3f}'
              f'  3着内AUC {roc_auc_score(kc["top3"], kc[c]):.4f}  レース内1位 {kc.loc[rk == 1, "top3"].mean() * 100:5.1f}%')


def within_auc(df, col):
    """レースの中での AUC の平均（3着内の馬が、そうでない馬より高く評価されている割合）。"""
    out = []
    for _, g in df.groupby('race_id'):
        if 0 < g['top3'].sum() < len(g) and g[col].nunique() > 1:
            out.append(roc_auc_score(g['top3'], g[col]))
    return float(np.mean(out))


trc, tec = kc['race_date'] < '2026-01-01', kc['race_date'] >= '2026-01-01'
b0 = LogisticRegression().fit(lg(kc.loc[trc, ['mkt']]), kc.loc[trc, 'top3'])
l0 = log_loss(kc.loc[tec, 'top3'], b0.predict_proba(lg(kc.loc[tec, ['mkt']]))[:, 1])
print('  レースの中で比べる（同じ頭）: レース内AUC / 市場に足した上積み(×1e-5)')
for kind in ('全走歴', '±400m', '距離変換'):
    for v, lab in (('r3', '近3走'), ('best', '最高')):
        c = f'{kind}_{v}'
        z = kc.groupby('race_id')[c].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0).fillna(0)
        X = np.c_[lg(kc['mkt']), z]
        m = LogisticRegression(C=1e4, max_iter=2000).fit(X[trc], kc.loc[trc, 'top3'])
        g_ = (l0 - log_loss(kc.loc[tec, 'top3'], m.predict_proba(X[tec])[:, 1])) * 1e5
        print(f'    {kind:6s} {lab:4s}  {within_auc(kc, c):.4f}  {g_:+7.1f}')
print('\n距離を変えた馬だけ（前走から±300m以上）')
prev = j.groupby('index').tail(1).set_index('index')
moved = ((pd.to_numeric(prev['dist']) - pd.to_numeric(prev['distance'])).abs() >= 300).reindex(key.index).fillna(False)
km = key[moved.astype(bool) & common]
for kind in ('全走歴', '±400m', '距離変換'):
    for v, lab in (('r3', '近3走'), ('best', '最高')):
        c = f'{kind}_{v}'
        both = km[km[c].notna() & km['actual'].notna()]
        print(f'  {kind:6s} {lab:4s}  {len(both):,}頭  実際の指数との相関 {np.corrcoef(both[c], both["actual"])[0, 1]:.3f}'
              f'  3着内AUC {roc_auc_score(both["top3"], both[c]):.4f}')
print(f'\n合計 {time.time() - t0:.0f}秒')
