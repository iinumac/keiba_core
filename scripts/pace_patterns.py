"""展開・馬場の特徴で、人気馬の凡走と人気薄の激走がどれだけ変わるか。

    python scripts/pace_patterns.py [出力CSV]

凡走: 1〜3番人気で6着以下 / 激走: 7番人気以下で3着以内。
各条件に当てはまる馬の率を、同じ人気で条件に当てはまらない馬の率
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

from keiba import pace, segments as sg, store

t0 = time.time()
races, results = store.load_for_features()
df = pace.add_horse_pace_features(results, races)
df['pop'] = pd.to_numeric(df['popularity'], errors='coerce')
df['fin'] = pd.to_numeric(df['finish_position'], errors='coerce')
df = df.dropna(subset=['pop', 'fin'])
df = df[~df['race_name'].astype(str).str.contains('障害')]
df['race_id'] = df['race_id'].astype(str)
place = sg.attach_place_payout(df[['race_id', 'horse_number']], store.read_table('payouts'))
df = df.merge(place[['race_id', 'horse_number', 'fuku']], on=['race_id', 'horse_number'], how='left')
df['year'] = df['race_date'].dt.year
df['period'] = np.where(df['year'] <= 2024, '〜2024', df['year'].astype(str))
print(f'データ {len(df):,} 頭  {time.time() - t0:.0f}秒', flush=True)

heavy_today = df['track_condition'].isin(['重', '不良'])
front, back = df['style'] <= pace.FRONT, df['style'] >= pace.BACK
q = df['p1_l3f_vs_pos'].quantile([.2, .8]).to_numpy()
he = df['heavy_edge'].quantile([.2, .8]).to_numpy()
SIGNALS = {
    '前走 展開が向かなかった':            df['p1_against'],
    '前走 展開が向いた':                  df['p1_with'],
    '前走 ハイを前で':                    (df['p1_pace'] == 'ハイ') & (df['p1_pos'] <= pace.FRONT),
    '前走 スローを後ろで':                (df['p1_pace'] == 'スロー') & (df['p1_pos'] >= pace.BACK),
    '前走 位置の割に上がりが速い（上位20%）': df['p1_l3f_vs_pos'] >= q[1],
    '前走 位置の割に上がりが遅い（下位20%）': df['p1_l3f_vs_pos'] <= q[0],
    '前の馬 × 前の馬が多い（4頭以上）':    front & (df['field_front'] >= 4),
    '前の馬 × 前の馬が少ない（1頭以下）':  front & (df['field_front'] <= 1),
    '後ろの馬 × 前の馬が多い（4頭以上）':  back & (df['field_front'] >= 4),
    '後ろの馬 × 前の馬が少ない（1頭以下）': back & (df['field_front'] <= 1),
    '重・不良が得意 × 今日 重・不良':      (df['heavy_edge'] >= he[1]) & heavy_today,
    '重・不良が苦手 × 今日 重・不良':      (df['heavy_edge'] <= he[0]) & heavy_today,
}
TARGETS = {
    '凡走（1〜3番人気が6着以下）': (df['pop'] <= 3, df['fin'] >= 6),
    '激走（7番人気以下が3着以内）': (df['pop'] >= 7, df['fin'] <= 3),
}

rows = []
for tname, (group, hit) in TARGETS.items():
    print(f'\n■ {tname}')
    print(f"{'条件':34s}" + ''.join(f'{p:>26s}' for p in ('〜2024', '2025', '2026')))
    for sname, cond in SIGNALS.items():
        cond = cond.fillna(False).astype(bool)
        line = f'{sname:34s}'
        for p in ('〜2024', '2025', '2026'):
            base = group & (df['period'] == p)
            s, o = df[base & cond], df[base & ~cond]
            if len(s) < 50:
                line += f"{'（少数）':>26s}"
                continue
            h_s, h_o = hit[s.index], hit[o.index]
            w = s['pop'].value_counts(normalize=True)
            rate_o = h_o.groupby(o['pop']).mean()
            exp = float((w * rate_o.reindex(w.index)).sum())
            fk_o = o['fuku'].groupby(o['pop']).mean()
            exp_fk = float((w * fk_o.reindex(w.index)).sum())
            diff = (h_s.mean() - exp) * 100
            line += f'{len(s):>7,}頭 {h_s.mean() * 100:5.1f}% {diff:+5.1f}pt'
            if tname.startswith('激走'):
                line += f' 複{s["fuku"].mean():3.0f}'
            rows.append(dict(target=tname, signal=sname, period=p, n=len(s), rate=h_s.mean(),
                             expected=exp, diff_pt=diff, fuku=s['fuku'].mean(), fuku_expected=exp_fk))
        print(line, flush=True)

out = sys.argv[1] if len(sys.argv) > 1 else 'pace_patterns.csv'
pd.DataFrame(rows).to_csv(out, index=False)
print(f'\n保存: {out}  合計 {time.time() - t0:.0f}秒')
