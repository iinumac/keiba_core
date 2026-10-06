"""接戦（ハナ・アタマ・クビ差）での勝負強さは、騎手・厩舎の持ち続ける力か。

    python scripts/clutch_eval.py

隣り合った着順の2頭で、後ろの馬の着差がハナ・アタマ・クビなら「接戦」。
前に出た側の騎手・厩舎に1勝、後ろの側に1敗。事前の実力差（単勝支持率の比）から
見込まれる勝率との差を数える。2024年までで評価を作り、2025年以降の接戦で当たるかを見る。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from keiba import store

CLOSE = {'ハナ', 'アタマ', 'クビ'}
races, res = store.read_table('races'), store.read_table('results')
res['race_id'] = res['race_id'].astype(str); races['race_id'] = races['race_id'].astype(str)
res = res.merge(races[['race_id', 'date', 'race_name']], on='race_id')
res = res[~res['race_name'].astype(str).str.contains('障害')]
res['fin'] = pd.to_numeric(res['finish_position'], errors='coerce')
res['odds'] = pd.to_numeric(res['odds'], errors='coerce')
res = res.dropna(subset=['fin', 'odds'])
inv = 1 / res['odds']; res['sup'] = inv / inv.groupby(res['race_id']).transform('sum')
res = res.sort_values(['race_id', 'fin'])
prev = res.groupby('race_id').shift(1)
close = res['margin'].astype(str).isin(CLOSE) & prev['fin'].notna() & (res['fin'] == prev['fin'] + 1)
P = pd.DataFrame({
    'race_id': res.loc[close, 'race_id'], 'date': pd.to_datetime(res.loc[close, 'date']),
    'fin_ahead': prev.loc[close, 'fin'],
    'j_a': prev.loc[close, 'jockey_id'], 'j_b': res.loc[close, 'jockey_id'],
    't_a': prev.loc[close, 'trainer_id'], 't_b': res.loc[close, 'trainer_id'],
    'lr': np.log(prev.loc[close, 'sup'] / res.loc[close, 'sup'])})
print(f'接戦 {len(P):,} 組（{P["race_id"].nunique():,} レース）  うち3着以内を争った組 {(P["fin_ahead"] <= 2).sum():,}')

# 向きをランダムに入れ替えて「見込み勝率」を学習（前に出た側が必ず勝つので、片側だけでは学べない）
rng = np.random.default_rng(0)
flip = rng.random(len(P)) < 0.5
X = np.where(flip, -P['lr'], P['lr']).reshape(-1, 1); y = (~flip).astype(int)
tr = (P['date'].dt.year <= 2024).to_numpy()
lr = LogisticRegression().fit(X[tr], y[tr])
P['exp_a'] = lr.predict_proba(P[['lr']].to_numpy())[:, 1]       # 前に出た側が勝つ見込み
print(f'見込み: 支持率が同じなら50%、2倍なら {lr.predict_proba([[np.log(2)]])[0, 1] * 100:.0f}%')

def rating(P_, a, b, k=30):
    """騎手（厩舎）ごとの (実際 − 見込み) の合計 ÷ (回数 + k)。少数は0に寄せる。"""
    w = pd.concat([pd.DataFrame({'id': P_[a], 'd': 1 - P_['exp_a']}),
                   pd.DataFrame({'id': P_[b], 'd': -(1 - P_['exp_a'])})])
    # 前に出た側: 実際1 − 見込みexp_a = 1−exp_a。後ろの側: 実際0 − 見込み(1−exp_a)
    g = w.groupby('id')['d'].agg(['sum', 'size'])
    return g['sum'] / (g['size'] + k), g['size']

for who, a, b in (('騎手', 'j_a', 'j_b'), ('厩舎', 't_a', 't_b')):
    r, n = rating(P[tr], a, b)
    te = P[~tr].copy()
    te['ra'] = te[a].map(r); te['rb'] = te[b].map(r)
    te = te.dropna(subset=['ra', 'rb'])
    te['diff'] = te['ra'] - te['rb']
    q = pd.qcut(te['diff'], 5, labels=['①A側が弱い', '②', '③', '④', '⑤A側が強い'])
    print(f'\n■ {who}（2024年までで評価 {len(r):,} 人・組、2025年以降の接戦 {len(te):,} 組）')
    print('  勝負強さの差で5分割:  前に出た側が勝った…ではなく、2頭のうち勝負強さの高い側が先着した割合を見る')
    # 2頭のうち評価の高い側が先着した割合 vs 見込み
    hi_is_a = te['ra'] > te['rb']
    actual = np.where(hi_is_a, 1, 0)
    expect = np.where(hi_is_a, te['exp_a'], 1 - te['exp_a'])
    gap = (te['ra'] - te['rb']).abs()
    for lo, hi in ((0, .01), (.01, .03), (.03, .06), (.06, 1)):
        m = (gap >= lo) & (gap < hi) & (te['ra'] != te['rb'])
        print(f'    評価の差 {lo:.2f}〜{hi:.2f}: {m.sum():>6,}組  評価の高い側の先着 {actual[m].mean() * 100:5.1f}%  見込み {expect[m].mean() * 100:5.1f}%  差 {(actual[m].mean() - expect[m].mean()) * 100:+.1f}pt')
    top = r[n >= 100].sort_values(ascending=False)
    print(f'  2024年までの上位（100回以上）: ' + ', '.join(f'{i}:{v:+.3f}' for i, v in top.head(5).items()))
