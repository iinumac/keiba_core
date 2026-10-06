"""上位人気なのに凡走した馬の特徴を探す（消す要素の候補）。

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
X = pd.DataFrame(F)
fav = (df['pop'] <= 3) & (df['n'] >= 8) & df['fin'].notna()
X = X[fav]; meta = df.loc[fav, ['race_id', 'pop', 'fin', 'year']]
flop, good = meta['fin'] >= 6, meta['fin'] <= 3
recent = meta['year'] >= 2025
rng = np.random.default_rng(0)
f_idx = rng.choice(meta.index[flop & recent], 1000, replace=False)
# 好走側は、凡走側の人気の内訳にそろえて選ぶ
mix = meta.loc[f_idx, 'pop'].value_counts()
g_idx = np.concatenate([rng.choice(meta.index[good & recent & (meta['pop'] == p)], k, replace=False) for p, k in mix.items()])
print(f'凡走 1,000頭（人気の内訳 {mix.sort_index().to_dict()}）と好走 1,000頭を比較\n', flush=True)

rows = []
for c in X.columns:
    a, b = X.loc[f_idx, c].dropna(), X.loc[g_idx, c].dropna()
    if len(a) < 100 or len(b) < 100:
        continue
    binary = set(X[c].dropna().unique()) <= {0.0, 1.0}
    sd = np.sqrt((a.var() + b.var()) / 2) or np.nan
    d = (a.mean() - b.mean()) / sd                       # 正なら凡走側で大きい
    # 2024年までの全データで、上位25%（二値なら該当）と下位25%の凡走率の差
    old = X[(meta['year'] <= 2024)][c]; m_old = meta[meta['year'] <= 2024]
    if binary:
        hi, lo = old == 1, old == 0
    else:
        q1, q3 = old.quantile([.25, .75]); hi, lo = old >= q3, old <= q1
    fr = lambda m: (m_old.loc[m[m].index, 'fin'] >= 6).mean() * 100
    rows.append(dict(要素=c, 種類='該当' if binary else '連続', 凡走側=a.mean(), 好走側=b.mean(), 効果量=d,
                     旧データ_高い側の凡走率=fr(hi), 旧データ_低い側の凡走率=fr(lo), 値あり=len(a)))
R = pd.DataFrame(rows)
R['旧データの差'] = R['旧データ_高い側の凡走率'] - R['旧データ_低い側の凡走率']
R['向きが一致'] = np.sign(R['効果量']) == np.sign(R['旧データの差'])
R = R.reindex(R['効果量'].abs().sort_values(ascending=False).index)
pd.set_option('display.width', 250); pd.set_option('display.unicode.east_asian_width', True)
print(R.round(3).to_string(index=False))
R.to_csv(sys.argv[1] if len(sys.argv) > 1 else 'flop_explore.csv', index=False)

# オッズ・混戦度・頭数から見込める凡走率を差し引いても、なお残る差
from sklearn.linear_model import LogisticRegression
M = meta.assign(lo=np.log(df.loc[fav, 'odds'].astype(float)), ne=np.log(df.loc[fav, 'n_eff']), nn=np.log(df.loc[fav, 'n']),
                flop=(meta['fin'] >= 6).astype(int)).dropna(subset=['lo', 'ne', 'nn'])
base_cols = ['lo', 'ne', 'nn']
old_m = M['year'] <= 2024
lrm = LogisticRegression().fit(M.loc[old_m, base_cols], M.loc[old_m, 'flop'])
M['resid'] = M['flop'] - lrm.predict_proba(M[base_cols])[:, 1]
print('\n【オッズ・混戦度・頭数で見込める凡走率を差し引いた後の差】 上位25%（該当）− 下位25%（非該当）の凡走率の差（pt）')
out = []
for c in X.columns:
    x = X.loc[M.index, c]
    binary = set(x.dropna().unique()) <= {0.0, 1.0}
    row = {'要素': c}
    for lab, m in (('〜2024', M['year'] <= 2024), ('2025〜', M['year'] >= 2025)):
        xx = x[m]
        if binary:
            hi, lo = xx == 1, xx == 0
        else:
            q1, q3 = x[M['year'] <= 2024].quantile([.25, .75]); hi, lo = xx >= q3, xx <= q1
        row[lab] = (M.loc[hi[hi].index, 'resid'].mean() - M.loc[lo[lo].index, 'resid'].mean()) * 100
    out.append(row)
O = pd.DataFrame(out)
O['一致'] = np.sign(O['〜2024']) == np.sign(O['2025〜'])
O = O.reindex(O['〜2024'].abs().sort_values(ascending=False).index)
print(O.round(1).to_string(index=False))
print(f'\n合計 {time.time() - t0:.0f}秒')
