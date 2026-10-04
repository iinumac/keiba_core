"""展開（ペース・位置取り）と馬場から、人気馬の凡走・人気薄の激走を探す特徴量。

## レースのペース（ハイ / ミドル / スロー）

前半3F − 後半3F は、距離や馬場で水準がまるで違う（短距離は前半が速いのが普通）。
そこで同じ 競馬場・芝ダ・距離・馬場状態 のレースの平均と比べた偏差で分ける。
前半が普段より速ければハイ、遅ければスロー。全体がおおむね3等分になる境界
（±0.43σ）で区切る。

## 馬ごとの特徴量（すべて前走まで。当該レースは含めない）

    style              脚質。最初のコーナーの位置÷頭数の過去平均（0=先頭、1=最後方）
    p1_pace            前走のペース
    p1_pos             前走の最初のコーナーの位置÷頭数
    p1_against         前走で展開が向かなかった（ハイを前で / スローを後ろで）
    p1_with            前走で展開が向いた（ハイを後ろで / スローを前で）
    p1_l3f_vs_pos      前走で、4角の位置の割に上がりが速かったか（正ほど速い）
    heavy_edge         重・不良でのタイム偏差 − 良でのタイム偏差（過去平均）
    field_front        今回のメンバーで、脚質が前（style ≤ 0.25）の頭数
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PACE_LABELS = ('ハイ', 'ミドル', 'スロー')
PACE_Z_CUT = 0.43
"""標準正規分布をおおむね3等分する境界。"""

FRONT, BACK = 0.25, 0.6
"""位置取りの区分。最初のコーナーの位置÷頭数がこれ以下なら前、以上なら後ろ。"""


def add_race_pace(races: pd.DataFrame, min_group: int = 30) -> pd.DataFrame:
    """races に pace_diff / pace_z / pace を付ける。"""
    r = races.copy()
    r['pace_diff'] = pd.to_numeric(r['pace_first3f'], errors='coerce') - \
        pd.to_numeric(r['pace_last3f'], errors='coerce')
    fine = ['venue_code', 'surface', 'distance', 'track_condition']
    coarse = ['surface', 'distance']
    z = pd.Series(np.nan, index=r.index)
    for keys in (fine, coarse):
        g = r.groupby(keys)['pace_diff']
        n, mu, sd = g.transform('count'), g.transform('mean'), g.transform('std')
        ok = z.isna() & (n >= min_group) & (sd > 0)
        z[ok] = ((r['pace_diff'] - mu) / sd)[ok]
    r['pace_z'] = z
    r['pace'] = np.select([z < -PACE_Z_CUT, z > PACE_Z_CUT], ['ハイ', 'スロー'], 'ミドル')
    r.loc[z.isna(), 'pace'] = None
    return r


def add_horse_pace_features(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """results（read_results_enriched の形）に展開まわりの特徴量を付ける。"""
    rp = add_race_pace(races)[['race_id', 'pace', 'pace_z']]
    df = results.merge(rp, on='race_id', how='left')
    df['race_date'] = pd.to_datetime(df['race_date'])
    fin = pd.to_numeric(df['finish_position'], errors='coerce')
    df['n'] = df.groupby('race_id')['horse_number'].transform('size')
    df['pos'] = pd.to_numeric(df['first_corner'], errors='coerce') / df['n']
    last_pos = pd.to_numeric(df['last_corner'], errors='coerce') / df['n']

    # 4角の位置の割に上がりが速いか：上がりの順位（比）と4角の位置（比）の差
    l3f = pd.to_numeric(df['last_3f'], errors='coerce')
    l3f_rank = l3f.groupby(df['race_id']).rank(method='min') / df['n']
    df['l3f_vs_pos'] = last_pos - l3f_rank

    # レース内のタイム偏差（速いほど大きい）
    t = pd.to_numeric(df['time_seconds'], errors='coerce')
    g = t.groupby(df['race_id'])
    df['time_z'] = -(t - g.transform('mean')) / g.transform('std').replace(0, np.nan)

    df = df.sort_values(['horse_id', 'race_date', 'race_id'])
    h = df.groupby('horse_id')
    df['style'] = h['pos'].transform(lambda x: x.shift().expanding().mean())
    df['p1_pace'] = h['pace'].shift(1)
    df['p1_pos'] = h['pos'].shift(1)
    df['p1_fin_ratio'] = (fin / df['n']).groupby(df['horse_id']).shift(1)
    df['p1_l3f_vs_pos'] = h['l3f_vs_pos'].shift(1)
    front, back = df['p1_pos'] <= FRONT, df['p1_pos'] >= BACK
    df['p1_against'] = ((df['p1_pace'] == 'ハイ') & front) | ((df['p1_pace'] == 'スロー') & back)
    df['p1_with'] = ((df['p1_pace'] == 'ハイ') & back) | ((df['p1_pace'] == 'スロー') & front)

    heavy = df['track_condition'].isin(['重', '不良'])
    good = df['track_condition'] == '良'
    for name, mask in (('_tz_heavy', heavy), ('_tz_good', good)):
        v = df['time_z'].where(mask)
        df[name] = v.groupby(df['horse_id']).transform(lambda x: x.shift().expanding().mean())
    df['heavy_edge'] = df['_tz_heavy'] - df['_tz_good']

    df['field_front'] = (df['style'] <= FRONT).groupby(df['race_id']).transform('sum')
    return df.drop(columns=['_tz_heavy', '_tz_good'])
