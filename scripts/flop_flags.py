"""消す要素のフラグ：人気馬に付いたフラグの数ごとの凡走率。

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

fav = (df['pop'] <= 3) & (df['n'] >= 8) & df['fin'].notna()
d = df[fav].copy()
d['flop'] = d['fin'] >= 6; d['top3'] = d['fin'] <= 3
old = d['year'] <= 2024
q = lambda c, p: d.loc[old, c].quantile(p)
d['p1_gap_v'] = d['p1_gap']
FLAGS = {
    '指数(最高)がメンバー内で下位':        d['fig_best_rank'] >= q('fig_best_rank', .75),
    '上がり指数(近3走)がメンバー内で下位': d['l3fig_r3_rank'] >= q('l3fig_r3_rank', .75),
    '指数(近3走)がトップから離れている':   d['fig_r3_gap'] <= q('fig_r3_gap', .25),
    '芝ダ替わり':                          (d['surface'] != d['prev_surface']) & d['prev_surface'].notna(),
    '複勝の支持が単勝の見込みより低い':     d['place_vs_win'] <= d.loc[d['year'] >= 2022, 'place_vs_win'].quantile(.25),
    '先行型':                              d['style'] <= q('style', .25),
    '前走 位置の割に上がりが遅い':         d['p1_l3f_vs_pos'] <= q('p1_l3f_vs_pos', .25),
    '休み明け':                            d['days_since_last'] >= q('days_since_last', .75),
    '昇級':                                (d['level_score'] > d['prev_level_score']) & d['prev_level_score'].notna(),
    '前走 人気より着順が悪い':             d['p1_gap_v'] >= q('p1_gap_v', .75),
}
print('境目（2024年まで）: 指数(最高)順位≥%d / 上がり指数順位≥%d / 指数トップとの差≤%.2f秒 / 先行≤%.2f / 前走位置の割の上がり≤%.2f / 休養≥%d日 / 前走ズレ≥%d'
      % (q('fig_best_rank', .75), q('l3fig_r3_rank', .75), q('fig_r3_gap', .25), q('style', .25), q('p1_l3f_vs_pos', .25),
         q('days_since_last', .75), q('p1_gap_v', .75)))
Fm = pd.DataFrame({k: v.fillna(False).astype(bool) for k, v in FLAGS.items()})
d['n_flags'] = Fm.sum(axis=1)
print(f"\n1〜3番人気 {len(d):,}頭（〜2024 {old.sum():,} / 2025〜 {(~old).sum():,}）  凡走率 〜2024 {d.loc[old, 'flop'].mean() * 100:.1f}% / 2025〜 {d.loc[~old, 'flop'].mean() * 100:.1f}%")
print('\n【フラグ1つずつ】該当した人気馬の凡走率（該当なしの馬）')
for k in Fm:
    line = f'  {k:20s}'
    for lab, m in (('〜2024', old), ('2025〜', ~old)):
        on, off = d[m & Fm[k]], d[m & ~Fm[k]]
        line += f'  {lab} {len(on):>6,}頭 {on["flop"].mean() * 100:5.1f}%（{off["flop"].mean() * 100:5.1f}%）'
    print(line)
for title, sub in (('1〜3番人気', d.index), ('1番人気だけ', d.index[d['pop'] == 1])):
    x = d.loc[sub]
    print(f'\n【フラグの数ごと】{title}')
    for lab, m in (('〜2024', x['year'] <= 2024), ('2025〜', x['year'] >= 2025)):
        y = x[m]; t = y.groupby(y['n_flags'].clip(upper=5)).agg(頭数=('flop', 'size'), 凡走率=('flop', 'mean'), 三着内率=('top3', 'mean'))
        print(f'  {lab}: ' + '  '.join(f"{int(k)}{'個以上' if k == 5 else '個'} {r.頭数:,}頭 凡走{r.凡走率 * 100:.0f}%/3着内{r.三着内率 * 100:.0f}%" for k, r in t.iterrows()))
print(f'\n合計 {time.time() - t0:.0f}秒')
