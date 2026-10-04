"""スピード指数（馬場・ペース補正タイム）の検証。

    python scripts/figure_eval.py [出力CSV]

1. 指数の当たり具合：前走までの指数がどれだけ着順を言い当てるか（レース内のAUC）
2. 人気馬の凡走・人気薄の激走：指数のメンバー内順位で率がどれだけ変わるか
   （同じ人気の馬、および支持率の式による市場の見込みと比べる）
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

from keiba import figure, market, segments as sg, store

t0 = time.time()
races, results = store.load_for_features()
df = figure.add_horse_figure_features(figure.add_figures(results, races))
df['pop'] = pd.to_numeric(df['popularity'], errors='coerce')
df['fin'] = pd.to_numeric(df['finish_position'], errors='coerce')
df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
df = df.dropna(subset=['pop', 'fin'])
df['race_id'] = df['race_id'].astype(str)
df['top3'] = df['fin'] <= 3
df['year'] = df['race_date'].dt.year
df['period'] = np.where(df['year'] <= 2024, '〜2024', df['year'].astype(str))
place = sg.attach_place_payout(df[['race_id', 'horse_number']], store.read_table('payouts'))
df = df.merge(place[['race_id', 'horse_number', 'fuku']], on=['race_id', 'horse_number'], how='left')
mk = np.full(len(df), np.nan)
for idx in df.groupby('race_id').indices.values():
    o = df['odds'].iloc[idx].to_numpy(float)
    if np.isfinite(o).all() and (o > 0).all():
        mk[idx] = market.top3_from_support(o)
df['mkt'] = mk

# 比較用：レース内のタイム偏差の直近3走平均（今のモデルの最重要特徴量と同じ考え方）
t = pd.to_numeric(df['time_seconds'], errors='coerce')
g = t.groupby(df['race_id'])
df['time_z'] = -(t - g.transform('mean')) / g.transform('std').replace(0, np.nan)
df = df.sort_values(['horse_id', 'race_date'])
df['tz_r3'] = df.groupby('horse_id')['time_z'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
print(f'データ {len(df):,} 頭  {time.time() - t0:.0f}秒', flush=True)

print('\n【1】前走までの値で3着内を言い当てる力（AUC、値が無い馬は除く）')
for c in ('tz_r3', 'fig_p1', 'fig_r3', 'fig_best', 'l3fig_r3', 'l3fig_best'):
    line = f'  {c:12s}'
    for p in ('〜2024', '2025', '2026'):
        s = df[(df['period'] == p) & df[c].notna()]
        line += f'  {p} {roc_auc_score(s["top3"], s[c]):.4f}'
    print(line, flush=True)

SIGNALS = {
    '指数(近3走) メンバー1位': df['fig_r3_rank'] == 1,
    '指数(近3走) メンバー3位以内': df['fig_r3_rank'] <= 3,
    '指数(近3走) メンバー6位以下': df['fig_r3_rank'] >= 6,
    '指数(最高) メンバー3位以内': df['fig_best_rank'] <= 3,
    '指数(最高) メンバー6位以下': df['fig_best_rank'] >= 6,
    '指数(前走) メンバー3位以内': df['fig_p1_rank'] <= 3,
    '上がり指数(近3走) メンバー3位以内': df['l3fig_r3_rank'] <= 3,
    '上がり指数(近3走) メンバー6位以下': df['l3fig_r3_rank'] >= 6,
    '指数(近3走) トップとの差 0.3秒以内': df['fig_r3_gap'] >= -0.3,
    '指数(近3走) トップとの差 1秒以上': df['fig_r3_gap'] <= -1.0,
}
TARGETS = {
    '凡走（1〜3番人気が6着以下）': (df['pop'] <= 3, df['fin'] >= 6),
    '激走（7番人気以下が3着以内）': (df['pop'] >= 7, df['fin'] <= 3),
}
rows = []
for tname, (group, hit) in TARGETS.items():
    print(f'\n【2】{tname}')
    for sname, cond in SIGNALS.items():
        cond = cond.fillna(False).astype(bool)
        line = f'  {sname:26s}'
        for p in ('〜2024', '2025', '2026'):
            base = group & (df['period'] == p)
            s, o = df[base & cond], df[base & ~cond]
            if len(s) < 50:
                line += '   （少数）' + ' ' * 22
                continue
            w = s['pop'].value_counts(normalize=True)
            exp = float((w * hit[o.index].groupby(o['pop']).mean().reindex(w.index)).sum())
            exp_fk = float((w * o['fuku'].groupby(o['pop']).mean().reindex(w.index)).sum())
            rate = hit[s.index].mean()
            line += f'  {len(s):>6,}頭 {rate*100:5.1f}% {(rate-exp)*100:+5.1f}pt 複{s["fuku"].mean():3.0f}({exp_fk:3.0f})'
            rows.append(dict(target=tname, signal=sname, period=p, n=len(s), rate=rate, expected=exp,
                             fuku=s['fuku'].mean(), fuku_expected=exp_fk,
                             top3_vs_market=(s['top3'].mean() - s['mkt'].mean())))
        print(line, flush=True)

out = sys.argv[1] if len(sys.argv) > 1 else 'figure_eval.csv'
pd.DataFrame(rows).to_csv(out, index=False)
print(f'\n保存: {out}  合計 {time.time() - t0:.0f}秒')
