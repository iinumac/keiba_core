"""人気薄なのに3着内に来た馬の特徴を探す（ピックアップ要素の候補）。

    python scripts/flop_explore.py

2025〜2026年の 1〜3番人気で6着以下の馬から1,000頭、3着以内の馬から人気の内訳を
そろえて1,000頭を選び、レース前に分かる要素の差を比べる。差の大きい候補は、
2024年までの全データでも同じ向きか（凡走率の差）を確かめる。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from keiba import figure, market, pace, store
from keiba.features import build_features, C04_CONFIG

t0 = time.time()
races, results = store.load_for_features()
df = build_features(races, results, C04_CONFIG)
df['race_id'] = df['race_id'].astype(str)
fig = figure.add_horse_figure_features(figure.add_figures(results, races)); fig['race_id'] = fig['race_id'].astype(str)
pc = pace.add_horse_pace_features(results, races); pc['race_id'] = pc['race_id'].astype(str)
df = df.merge(fig[['race_id', 'horse_number', 'fig_r3', 'fig_r3_rank', 'fig_r3_gap', 'fig_best_rank', 'l3fig_r3_rank', 'fig_p1']],
              on=['race_id', 'horse_number'], how='left')
df = df.merge(pc[['race_id', 'horse_number', 'style', 'field_front', 'p1_pace', 'p1_pos', 'p1_l3f_vs_pos']],
              on=['race_id', 'horse_number'], how='left')
r = races.copy(); r['race_id'] = r['race_id'].astype(str)
df = df.merge(r[['race_id', 'venue_code']].rename(columns={'venue_code': 'venue'}), on='race_id', how='left', suffixes=('', '_r'))
print(f'準備 {time.time() - t0:.0f}秒', flush=True)

df['pop'] = pd.to_numeric(df['popularity'], errors='coerce')
df['fin'] = pd.to_numeric(df['finish_position'], errors='coerce')
df['n'] = df.groupby('race_id')['horse_number'].transform('size')
df = df.sort_values(['horse_id', 'race_date', 'race_id'])
h = df.groupby('horse_id')
for c in ('distance', 'surface', 'level_score', 'venue', 'horse_weight', 'impost', 'fin', 'n'):
    df[f'prev_{c}'] = h[c].shift(1)
odds = pd.to_numeric(df['odds'], errors='coerce')
inv = 1 / odds
ws = inv / inv.groupby(df['race_id']).transform('sum')
df['n_eff'] = 1 / (ws ** 2).groupby(df['race_id']).transform('sum')
df['odds_vs_pop'] = odds   # 同じ人気順位でのオッズの高さ
df['gap_to_fav'] = odds / odds.groupby(df['race_id']).transform('min')

# 単勝と複勝の支持の食い違い（2022年以降）
o = store.read_table('odds'); o['race_id'] = o['race_id'].astype(str)
mk = []
for rid, g in o.groupby('race_id'):
    if g[['win', 'place_min', 'place_max']].notna().all().all():
        mk.append(pd.DataFrame({'race_id': rid, 'horse_number': g['horse_number'],
                                'place_vs_win': np.log(market.place_support(g['place_min'], g['place_max'])
                                                       / np.clip(market.top3_from_support(g['win'].to_numpy(float)), 1e-4, 1))}))
df = df.merge(pd.concat(mk), on=['race_id', 'horse_number'], how='left')

F = {
    # 連続値
    '前走着順': df['prev_fin'], '前走 着順÷頭数': df['prev_fin'] / df['prev_n'],
    '前走の人気': df['prev_popularity'], '前走オッズ': df['prev_odds'],
    '前走 人気−着順のズレ': df['p1_gap'], '前走着差（馬身）': df['p1_margin'],
    '休養日数': df['days_since_last'], '馬体重': df['horse_weight'], '馬体重の増減': df['horse_weight'] - df['prev_horse_weight'],
    '斤量': df['impost'], '斤量の増減': df['impost'] - df['prev_impost'], '馬番': df['horse_number'], '頭数': df['n'], '年齢': df['age'],
    '距離の変化（m）': df['distance'] - df['prev_distance'], 'クラスの変化': df['level_score'] - df['prev_level_score'],
    '指数(近3走)': df['fig_r3'], '指数(近3走)のメンバー内順位': df['fig_r3_rank'], '指数(近3走)のトップとの差': df['fig_r3_gap'],
    '指数(最高)のメンバー内順位': df['fig_best_rank'], '上がり指数(近3走)のメンバー内順位': df['l3fig_r3_rank'],
    '前走の指数': df['fig_p1'], '脚質（0前〜1後）': df['style'], 'メンバーの先行馬の数': df['field_front'],
    '前走 位置の割の上がり': df['p1_l3f_vs_pos'], '有力馬の数（混戦度）': df['n_eff'],
    '1番人気とのオッズ倍率': df['gap_to_fav'], '単勝オッズ': df['odds_vs_pop'],
    '複勝の支持÷単勝からの見込み（log）': df['place_vs_win'],
    '騎手の上積み': df['jockey_added_value'], '厩舎の上積み': df['trainer_added_value'],
    '通算3着内率': df['horse_prev_top3_rate'], '連続3着内': df['streak_top3'],
    # 該当するかどうか
    '乗り替わり': df['is_jockey_changed'].astype(float),
    '芝ダ替わり': (df['surface'] != df['prev_surface']).astype(float).where(df['prev_surface'].notna()),
    '競馬場替わり': (df['venue'] != df['prev_venue']).astype(float).where(df['prev_venue'].notna()),
    '昇級': (df['level_score'] > df['prev_level_score']).astype(float).where(df['prev_level_score'].notna()),
    '牝馬': (df['sex'] == '牝').astype(float),
    '前走ハイペースを前で': ((df['p1_pace'] == 'ハイ') & (df['p1_pos'] <= .25)).astype(float).where(df['p1_pace'].notna()),
    '重・不良馬場': df['track_condition'].isin(['重', '不良']).astype(float),
}

# 前走の勝ち方：前走で5着から何秒離していたか（前走1〜3着のときだけ）
t_ = pd.to_numeric(df['time_seconds'], errors='coerce')
t5 = df[df['fin'] == 5].groupby('race_id')['time_seconds'].min()
df['spread5'] = df['race_id'].map(t5) - t_.groupby(df['race_id']).transform('min')
df = df.sort_values(['horse_id', 'race_date', 'race_id'])
df['p1_spread5'] = df.groupby('horse_id')['spread5'].shift(1)
F['前走1〜3着で5着から離した秒数'] = df['p1_spread5'].where(df['prev_fin'] <= 3)
F['前走1〜3着'] = (df['prev_fin'] <= 3).astype(float).where(df['prev_fin'].notna())
F['前走1着'] = (df['prev_fin'] == 1).astype(float).where(df['prev_fin'].notna())
F['降級'] = (df['level_score'] < df['prev_level_score']).astype(float).where(df['prev_level_score'].notna())
F['距離短縮'] = ((df['distance'] - df['prev_distance']) <= -200).astype(float).where(df['prev_distance'].notna())
F['距離延長'] = ((df['distance'] - df['prev_distance']) >= 200).astype(float).where(df['prev_distance'].notna())
F['差し・追込型'] = df['style']

X = pd.DataFrame(F)
grp = (df['pop'] >= 4) & (df['n'] >= 8) & df['fin'].notna()
X = X[grp]; meta = df.loc[grp, ['race_id', 'pop', 'fin', 'year']]
from sklearn.linear_model import LogisticRegression
M = meta.assign(lo=np.log(df.loc[grp, 'odds'].astype(float)), ne=np.log(df.loc[grp, 'n_eff']), nn=np.log(df.loc[grp, 'n']),
                hit=(meta['fin'] <= 3).astype(int)).dropna(subset=['lo', 'ne', 'nn'])
base = ['lo', 'ne', 'nn']
old = M['year'] <= 2024
lr = LogisticRegression().fit(M.loc[old, base], M.loc[old, 'hit'])
M['resid'] = M['hit'] - lr.predict_proba(M[base])[:, 1]
print(f"人気4番以下 {len(M):,}頭  3着内率 〜2024 {M.loc[old, 'hit'].mean() * 100:.1f}% / 2025〜 {M.loc[~old, 'hit'].mean() * 100:.1f}%")
print('\n【オッズ・混戦度・頭数で見込める3着内率を差し引いた後の差】 上位25%（該当）− 下位25%（非該当）の3着内率の差（pt）')
out = []
for c in X.columns:
    x = X.loc[M.index, c]
    binary = set(x.dropna().unique()) <= {0.0, 1.0}
    row = {'要素': c, '種類': '該当' if binary else '連続'}
    for lab, m in (('〜2024', old), ('2025〜', ~old)):
        xx = x[m]
        if binary:
            hi, lo = xx == 1, xx == 0
        else:
            q1, q3 = x[old].quantile([.25, .75]); hi, lo = xx >= q3, xx <= q1
        row[lab] = (M.loc[hi[hi].index, 'resid'].mean() - M.loc[lo[lo].index, 'resid'].mean()) * 100
    out.append(row)
O = pd.DataFrame(out)
O['一致'] = np.sign(O['〜2024']) == np.sign(O['2025〜'])
O = O.reindex(O['〜2024'].abs().sort_values(ascending=False).index)
pd.set_option('display.width', 250); pd.set_option('display.unicode.east_asian_width', True)
print(O.round(1).to_string(index=False))
print(f'\n合計 {time.time() - t0:.0f}秒')
