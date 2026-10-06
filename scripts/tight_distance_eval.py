"""前走の混戦具合と、距離の得意不得意。

    python scripts/tight_distance_eval.py

各条件に当てはまる馬の3着内率・複勝回収率を、同じ人気で当てはまらない馬
（人気の構成をそろえた期待値）と比べる。期間は 〜2024 / 2025 / 2026。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from keiba import figure, segments as sg, store

t0 = time.time()
races, results = store.load_for_features()
df = figure.add_figures(results, races)
df['race_id'] = df['race_id'].astype(str)
df['fin'] = pd.to_numeric(df['finish_position'], errors='coerce')
df['pop'] = pd.to_numeric(df['popularity'], errors='coerce')
t = pd.to_numeric(df['time_seconds'], errors='coerce')
df['behind'] = t - t.groupby(df['race_id']).transform('min')               # 勝ち馬との差（秒）
t5 = df[df['fin'] == 5].groupby('race_id')['time_seconds'].min()   # 5着同着のレースもある
df['spread5'] = df['race_id'].map(t5) - t.groupby(df['race_id']).transform('min')   # 5着までの詰まり具合（秒）
df['dist'] = pd.to_numeric(df['distance'], errors='coerce')

df = df.sort_values(['horse_id', 'race_date', 'race_id'])
h = df.groupby('horse_id')
for c in ('fin', 'behind', 'spread5', 'dist'):
    df[f'p1_{c}'] = h[c].shift(1)

# 距離の得意不得意：今回の距離 ±200m で走った過去の指数の平均 − 過去全体の平均
def dist_aptitude(g):
    d, f = g['dist'].to_numpy(), g['figure'].to_numpy()
    near_mean, near_n, all_mean = [], [], []
    for i in range(len(g)):
        past_d, past_f = d[:i], f[:i]
        ok = ~np.isnan(past_f)
        near = ok & (np.abs(past_d - d[i]) <= 200)
        near_n.append(int((np.abs(past_d - d[i]) <= 200).sum()))
        near_mean.append(past_f[near].mean() if near.sum() else np.nan)
        all_mean.append(past_f[ok].mean() if ok.sum() else np.nan)
    return pd.DataFrame({'near_n': near_n, 'apt': np.array(near_mean) - np.array(all_mean)}, index=g.index)

apt = df.groupby('horse_id', group_keys=False)[['dist', 'figure']].apply(dist_aptitude)
df = df.join(apt)
df['n_past'] = h.cumcount()
print(f'準備 {time.time() - t0:.0f}秒', flush=True)

df = df.dropna(subset=['fin', 'pop'])
df = df[df.groupby('race_id')['horse_number'].transform('size') >= 8]
place = sg.attach_place_payout(df[['race_id', 'horse_number']], store.read_table('payouts'))
df = df.merge(place[['race_id', 'horse_number', 'fuku']], on=['race_id', 'horse_number'], how='left')
df['top3'] = df['fin'] <= 3
df['year'] = df['race_date'].dt.year
df['period'] = np.where(df['year'] <= 2024, '〜2024', df['year'].astype(str))

mid = df['p1_fin'].between(4, 8)
has2 = (df['near_n'] >= 2) & (df['n_past'] >= 3)
SIGNALS = {
    '前走4〜8着だが勝ち馬と0.3秒以内': mid & (df['p1_behind'] <= 0.3),
    '前走4〜8着でレースが詰まっていた（5着まで0.3秒以内）': mid & (df['p1_spread5'] <= 0.3),
    '前走4〜8着で1秒以上の大差': mid & (df['p1_behind'] >= 1.0),
    '前走1〜3着でレースが詰まっていた': df['p1_fin'].le(3) & (df['p1_spread5'] <= 0.3),
    '前走1〜3着で5着まで1秒以上離れていた': df['p1_fin'].le(3) & (df['p1_spread5'] >= 1.0),
    '距離延長（前走から+400m以上）': (df['dist'] - df['p1_dist']) >= 400,
    '距離短縮（前走から−400m以上）': (df['dist'] - df['p1_dist']) <= -400,
    '初めての距離（±200m以内の経験なし、3走以上）': (df['near_n'] == 0) & (df['n_past'] >= 3),
    '近い距離が得意（指数 +0.3秒以上）': has2 & (df['apt'] >= 0.3),
    '近い距離が苦手（指数 −0.3秒以下）': has2 & (df['apt'] <= -0.3),
}
GROUPS = {'全体': df['pop'] >= 1, '人気4番以下': df['pop'] >= 4}
for gname, gmask in GROUPS.items():
    print(f'\n■ {gname}（条件に当てはまる馬の3着内率 / 同じ人気の他の馬との差 / 複勝回収率（他の馬））')
    for sname, cond in SIGNALS.items():
        cond = cond.fillna(False).astype(bool)
        line = f'  {sname:34s}'
        for p in ('〜2024', '2025', '2026'):
            base = gmask & (df['period'] == p)
            s, o = df[base & cond], df[base & ~cond]
            if len(s) < 100:
                line += '   （少数）                       '
                continue
            w = s['pop'].value_counts(normalize=True)
            exp = float((w * o.groupby('pop')['top3'].mean().reindex(w.index)).sum())
            efk = float((w * o.groupby('pop')['fuku'].mean().reindex(w.index)).sum())
            line += f'  {len(s):>7,}頭 {s["top3"].mean() * 100:5.1f}% {(s["top3"].mean() - exp) * 100:+5.1f}pt 複{s["fuku"].mean():3.0f}({efk:3.0f})'
        print(line, flush=True)
print(f'\n合計 {time.time() - t0:.0f}秒')
