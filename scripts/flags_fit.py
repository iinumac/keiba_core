"""ピックアップ・凡走フラグの数と、補正スコア（市場の見込みをフラグ数で補正した3着内確率）。

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


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
d = df[(df['n'] >= 8) & df['fin'].notna() & df['pop'].notna()].copy()
d['hit'] = d['fin'] <= 3; d['flop'] = d['fin'] >= 6
old = d['year'] <= 2024
fav_old, ls_old = old & (d['pop'] <= 3), old & (d['pop'] >= 4)
qf = lambda c, p: float(d.loc[fav_old, c].quantile(p))      # 凡走フラグの境目は1〜3番人気で決める
qp = lambda c, p: float(d.loc[ls_old, c].quantile(p))       # ピックアップの境目は人気4番以下で決める
d['p1_gap_v'] = d['p1_gap']
T = dict(
    flop_fig_best_rank=qf('fig_best_rank', .75), flop_l3fig_rank=qf('l3fig_r3_rank', .75), flop_fig_gap=qf('fig_r3_gap', .25),
    flop_style=qf('style', .25), flop_p1_l3f=qf('p1_l3f_vs_pos', .25), flop_rest=qf('days_since_last', .75), flop_p1_gap=qf('p1_gap_v', .75),
    flop_place_vs_win=float(d.loc[old & (d['pop'] <= 3) & (d['year'] >= 2022), 'place_vs_win'].quantile(.25)),
    pick_fig_best_rank=qp('fig_best_rank', .25), pick_fig_gap=qp('fig_r3_gap', .75), pick_l3fig_rank=qp('l3fig_r3_rank', .25),
    pick_fig_p1=qp('fig_p1', .75), pick_p1_l3f=qp('p1_l3f_vs_pos', .75), pick_rest=qp('days_since_last', .25),
    pick_p1_gap=qp('p1_gap_v', .25), pick_spread5=float(d.loc[ls_old & (d['prev_fin'] <= 3), 'p1_spread5'].quantile(.75)))
print('境目（2024年まで）:', {k: round(v, 3) for k, v in T.items()})
surf_change = (d['surface'] != d['prev_surface']) & d['prev_surface'].notna()
FLOP = {
    '指数(最高)がメンバー内で下位': d['fig_best_rank'] >= T['flop_fig_best_rank'],
    '上がり指数(近3走)がメンバー内で下位': d['l3fig_r3_rank'] >= T['flop_l3fig_rank'],
    '指数(近3走)がトップから離れている': d['fig_r3_gap'] <= T['flop_fig_gap'],
    '芝ダ替わり': surf_change,
    '複勝の支持が単勝の見込みより低い': d['place_vs_win'] <= T['flop_place_vs_win'],
    '先行型': d['style'] <= T['flop_style'],
    '前走 位置の割に上がりが遅い': d['p1_l3f_vs_pos'] <= T['flop_p1_l3f'],
    '休み明け': d['days_since_last'] >= T['flop_rest'],
    '昇級': (d['level_score'] > d['prev_level_score']) & d['prev_level_score'].notna(),
    '前走 人気より着順が悪い': d['p1_gap_v'] >= T['flop_p1_gap'],
}
PICK = {
    '指数(最高)がメンバー内で上位': d['fig_best_rank'] <= T['pick_fig_best_rank'],
    '指数(近3走)がトップに近い': d['fig_r3_gap'] >= T['pick_fig_gap'],
    '上がり指数(近3走)がメンバー内で上位': d['l3fig_r3_rank'] <= T['pick_l3fig_rank'],
    '前走の指数が高い': d['fig_p1'] >= T['pick_fig_p1'],
    '前走 位置の割に上がりが速い': d['p1_l3f_vs_pos'] >= T['pick_p1_l3f'],
    '前走1〜3着で5着から離した': (d['prev_fin'] <= 3) & (d['p1_spread5'] >= T['pick_spread5']),
    '使い詰め（休養が短い）': d['days_since_last'] <= T['pick_rest'],
    '前走 人気より着順が良い': d['p1_gap_v'] <= T['pick_p1_gap'],
}
d['n_flop'] = pd.DataFrame({k: v.fillna(False) for k, v in FLOP.items()}).sum(axis=1)
for k, v in {**PICK, **FLOP}.items():
    d[k] = v.fillna(False).astype(float)

# 市場の見込み（2022年以降）
o = store.read_table('odds'); o['race_id'] = o['race_id'].astype(str)
mk = []
for rid, g in o.groupby('race_id'):
    if g[['win', 'place_min', 'place_max', 'win_pop']].notna().all().all():
        mk.append(pd.DataFrame({'race_id': rid, 'horse_number': g['horse_number'],
                                'mkt': market.market_top3(g['win'], g['place_min'], g['place_max'], g['win_pop'])}))
d = d.merge(pd.concat(mk), on=['race_id', 'horse_number'], how='left')
e = d.dropna(subset=['mkt'])
lgt = lambda p: np.log(np.clip(p, 1e-4, .9999) / (1 - np.clip(p, 1e-4, .9999)))
tr, te = e[e['year'] <= 2024], e[e['year'] >= 2025]

# ピックアップ候補1つずつの重さ。単独で効いて見えても、市場の見込みと他の候補を入れると
# ほとんど残らないものがある（2026-10-07: 残ったのは指数の2つだけ）
full = LogisticRegression(C=1e4, max_iter=2000).fit(
    np.column_stack([lgt(tr['mkt'])] + [tr[k] for k in PICK]), tr['hit'])
print('\n【ピックアップ候補の重さ】市場の見込みと全候補を入れたときの係数（logit）')
for k, b in sorted(zip(PICK, full.coef_[0][1:]), key=lambda x: -x[1]):
    print(f'  {k:22s} {b:+.3f}')
USED = ['指数(最高)がメンバー内で上位', '指数(近3走)がトップに近い']

# 凡走フラグの重さ。1〜5番人気が6着以下になるかを、市場の見込み・ピックアップ・凡走候補で当てる。
# 指数(近3走)がトップから離れている（指数(最高)と重なる）・昇級・前走 人気より着順が悪い は
# 重みがほぼ残らなかったので外した（2026-10-08）
FLOP_USED = ['休み明け', '上がり指数(近3走)がメンバー内で下位', '芝ダ替わり', '先行型',
             '複勝の支持が単勝の見込みより低い', '前走 位置の割に上がりが遅い', '指数(最高)がメンバー内で下位']
fav_tr = tr[tr['pop'] <= 5]
fm = LogisticRegression(C=1e4, max_iter=2000).fit(
    np.column_stack([lgt(fav_tr['mkt'])] + [fav_tr[k] for k in USED + FLOP_USED]), fav_tr['flop'])
FLOP_W = dict(zip(FLOP_USED, fm.coef_[0][1 + len(USED):]))
print('\n【凡走フラグの重さ】1〜5番人気の6着以下（logit）')
for k, w in sorted(FLOP_W.items(), key=lambda x: -x[1]):
    print(f'  {k:24s} {w:+.3f}')
for x in (tr, te):
    x['flop_score'] = sum(x[k] * w for k, w in FLOP_W.items())
top_w = max(FLOP_W.values())
fav_te = te[te['pop'] <= 5]
pts = (fav_te['flop_score'] / top_w).round(1)
print('2025〜 1〜5番人気：凡走点（いちばん重いフラグ=1点）ごとの6着以下率')
for lo_, hi_ in ((0, .5), (.5, 1.5), (1.5, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 99)):
    g = fav_te[(pts >= lo_) & (pts < hi_)]
    print(f'  {lo_:.1f}〜{hi_:.1f}点 {len(g):>6,}頭 {g["flop"].mean() * 100:4.1f}%')

# 補正スコア：市場の見込み＋使うピックアップ（それぞれの重み）＋凡走の重みの合計
Xs = lambda x: np.column_stack([lgt(x['mkt'])] + [x[k] for k in USED] + [x['flop_score']])
cal = LogisticRegression(C=1e4, max_iter=2000).fit(Xs(tr), tr['hit'])
c = cal.coef_[0]
print('\n補正スコアの式: 切片 %.4f / 市場の見込み(logit) %.4f / %s / 凡走の重みの合計 %+.4f'
      % (cal.intercept_[0], c[0], ' / '.join(f'{k} {w:+.4f}' for k, w in zip(USED, c[1:-1])), c[-1]))
p_cal, p_mkt = cal.predict_proba(Xs(te))[:, 1], te['mkt'].to_numpy()
print(f"2025〜 市場の見込みだけ: 対数損失 {log_loss(te['hit'], p_mkt):.4f} AUC {roc_auc_score(te['hit'], p_mkt):.4f}"
      f"  →  補正スコア: 対数損失 {log_loss(te['hit'], p_cal):.4f} AUC {roc_auc_score(te['hit'], p_cal):.4f}")
lo = te[te['pop'] >= 4]
print('2025〜 人気4番以下（指数のピックアップの立ち方ごと。3着内率 / 市場の見込み）')
for (a, b), g in lo.groupby([lo[USED[0]], lo[USED[1]]]):
    print(f"  指数(最高){'○' if a else '×'} 指数(近3走){'○' if b else '×'}  {len(g):>6,}頭  "
          f"{g['hit'].mean() * 100:4.1f}% / {g['mkt'].mean() * 100:4.1f}%")
te = te.assign(cal=p_cal)
for k in ('mkt', 'cal'):
    te[f'{k}_rank'] = te.groupby('race_id')[k].rank(ascending=False, method='first')
print('2025〜 レース内1位の3着内率: 市場の見込み %.1f%% / 補正スコア %.1f%%' % (
    te[te['mkt_rank'] == 1]['hit'].mean() * 100, te[te['cal_rank'] == 1]['hit'].mean() * 100))
import json
json.dump({'thresholds': T, 'calibration': [float(cal.intercept_[0]), float(c[0]), float(c[-1])],
           'pick_weights': dict(zip(USED, map(float, c[1:-1]))),
           'flop_weights': {k: float(w) for k, w in FLOP_W.items()}},
          open(sys.argv[1] if len(sys.argv) > 1 else 'flags_fit.json', 'w'), ensure_ascii=False, indent=1)
print(f'\n合計 {time.time() - t0:.0f}秒')
