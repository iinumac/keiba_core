"""直線の長さと、差し・追込馬の成績。

    python scripts/straight_closers.py

直線が長いコースほど、後ろから行く馬が3着内に来やすいか。3つの面から見る。

  【1】実際の走り   4コーナーの位置ごとの3着内率（結果論。どれだけ効くかの上限の目安）
  【2】レース前     過去の走りから見た脚質ごとの3着内率と、単勝オッズからの見込みとの差
                    （差があれば、市場が直線の長さを織り込みきれていない）
  【3】モデル       市場あり・市場なしのモデル（2024年まで学習）の2025年以降の確率との差
                    （差があれば、モデルが取りこぼしている）

直線の長さは JRA 公表のコース紹介の値（Aコース）。障害と新潟の直線1000mは除く。
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from keiba import config, market, pace, stages, store
from keiba.features import build_features, C04_CONFIG

STRAIGHT = {   # (競馬場, 芝/ダート, 最後の直線が内回りか外回りか) -> m
    ('札幌', '芝', '内'): 266.1, ('函館', '芝', '内'): 262.1, ('福島', '芝', '内'): 292.0,
    ('新潟', '芝', '内'): 358.7, ('新潟', '芝', '外'): 658.7, ('東京', '芝', '内'): 525.9,
    ('中山', '芝', '内'): 310.0, ('中山', '芝', '外'): 310.0, ('中京', '芝', '内'): 412.5,
    ('京都', '芝', '内'): 328.4, ('京都', '芝', '外'): 403.7, ('阪神', '芝', '内'): 356.5,
    ('阪神', '芝', '外'): 473.6, ('小倉', '芝', '内'): 293.0,
    ('札幌', 'ダート', '内'): 264.3, ('函館', 'ダート', '内'): 260.3, ('福島', 'ダート', '内'): 295.7,
    ('新潟', 'ダート', '内'): 353.9, ('東京', 'ダート', '内'): 501.6, ('中山', 'ダート', '内'): 308.0,
    ('中京', 'ダート', '内'): 410.7, ('京都', 'ダート', '内'): 329.1, ('阪神', 'ダート', '内'): 352.7,
    ('小倉', 'ダート', '内'): 291.3,
}
# course_type の表記 → 最後の直線（「外-内」は外回りから入って内回りでゴール）
FINISH = {'': '内', '外': '外', '内-外': '外', '外-内': '内', '内2周': '内'}
BANDS = [0, 300, 340, 380, 450, 520, 700]
BAND_LAB = ['〜300m', '300〜340m', '340〜380m', '380〜450m', '450〜520m', '520m〜']

t0 = time.time()
races, results = store.load_for_features()
races = races.copy(); races['race_id'] = races['race_id'].astype(str)
rr = store.read_table('races'); rr['race_id'] = rr['race_id'].astype(str)
if 'course_type' not in races:
    races = races.merge(rr[['race_id', 'course_type']], on='race_id', how='left')
races['finish_side'] = races['course_type'].fillna('').map(FINISH)
races['straight'] = [STRAIGHT.get((v, s, f)) for v, s, f in zip(races['venue_name'], races['surface'], races['finish_side'])]
jump = races['race_name'].astype(str).str.contains('障害')
races = races[~jump & races['straight'].notna()]

pc = pace.add_horse_pace_features(results, store.load_for_features()[0])
pc['race_id'] = pc['race_id'].astype(str)
d = pc.merge(races[['race_id', 'straight']], on='race_id', how='inner')
d['fin'] = pd.to_numeric(d['finish_position'], errors='coerce')
d = d[d['fin'].notna() & (d['n'] >= 8)].copy()
d['top3'] = (d['fin'] <= 3).astype(int)
d['last_pos'] = pd.to_numeric(d['last_corner'], errors='coerce') / d['n']
# レース前の脚質：過去の4コーナー位置の平均（0 前 〜 1 後ろ）。pace.style は1コーナー
d = d.sort_values(['horse_id', 'race_date', 'race_id'])
d['style4'] = d.groupby('horse_id')['last_pos'].transform(lambda x: x.shift().expanding().mean())
d['n_past'] = d.groupby('horse_id').cumcount()
d['band'] = pd.cut(d['straight'], BANDS, labels=BAND_LAB)
# 単勝オッズからの3着内見込み（全期間で使える）
mk = []
for rid, g in d.groupby('race_id'):
    o = pd.to_numeric(g['odds'], errors='coerce').to_numpy(float)
    if np.isfinite(o).all() and (o > 0).all():
        mk.append(pd.Series(market.top3_from_support(o), index=g.index))
d['mkt'] = pd.concat(mk)
print(f'準備 {time.time() - t0:.0f}秒  {d["race_id"].nunique():,}レース {len(d):,}頭\n', flush=True)


def grp(x, col, cuts, labs):
    return pd.cut(x[col], cuts, labels=labs, include_lowest=True)


POS_CUT, POS_LAB = [0, .25, .5, .75, 1.0001], ['前（〜25%）', '中団前', '中団後', '後方（75%〜）']
for surf in ('芝', 'ダート'):
    x = d[d['surface'] == surf]
    print(f'━━ {surf} ━━')
    print('【1】実際の4コーナー位置ごとの3着内率（結果論）')
    t = x.assign(pos4=grp(x, 'last_pos', POS_CUT, POS_LAB)).pivot_table(index='band', columns='pos4', values='top3', aggfunc='mean') * 100
    n = x.groupby('band')['race_id'].nunique()
    t['レース数'] = n
    print(t.round(1).to_string(), '\n')

    y = x[(x['n_past'] >= 3) & x['mkt'].notna()]
    y = y.assign(sty=grp(y, 'style4', POS_CUT, POS_LAB))
    print('【2】レース前の脚質（過去の4コーナー位置の平均、3走以上）ごとの 3着内率 − 単勝オッズからの見込み（pt）')
    a = y.pivot_table(index='band', columns='sty', values='top3', aggfunc='mean')
    m = y.pivot_table(index='band', columns='sty', values='mkt', aggfunc='mean')
    c = y.pivot_table(index='band', columns='sty', values='top3', aggfunc='size')
    print(((a - m) * 100).round(1).astype(str).add(' (') .add(c.astype(str)).add(')').to_string())
    print('  参考: 3着内率そのもの')
    print((a * 100).round(1).to_string(), '\n')

# 【3】モデル（2025年以降）
df = build_features(*store.load_for_features(), C04_CONFIG)
df['race_id'] = df['race_id'].astype(str)
df = df[pd.to_datetime(df['race_date']) >= '2025-01-01']
for name, file in (('市場あり', 'model_strategy.pkl'), ('市場なし', 'model_free.pkl')):
    b = pickle.load(open(config.MODEL_DIR / file, 'rb'))
    df[name] = b['model'].predict(stages.model_input(df, b['features']))
df['市場なし'] = df['市場なし'] * 3 / df.groupby('race_id')['市場なし'].transform('sum')   # 目線と同じく合計3に
z = d.merge(df[['race_id', 'horse_number', '市場あり', '市場なし']], on=['race_id', 'horse_number'])
z = z[z['n_past'] >= 3]
z = z.assign(sty=grp(z, 'style4', POS_CUT, POS_LAB))
print('【3】2025年以降：3着内率 − モデルの確率（pt）。括弧は頭数')
for surf in ('芝', 'ダート'):
    w = z[z['surface'] == surf]
    for name in ('市場あり', '市場なし'):
        a = w.pivot_table(index='band', columns='sty', values='top3', aggfunc='mean')
        m = w.pivot_table(index='band', columns='sty', values=name, aggfunc='mean')
        c = w.pivot_table(index='band', columns='sty', values='top3', aggfunc='size')
        print(f'  {surf}・{name}')
        print(((a - m) * 100).round(1).astype(str).add(' (').add(c.astype(str)).add(')').to_string(), '\n')
print(f'合計 {time.time() - t0:.0f}秒')
