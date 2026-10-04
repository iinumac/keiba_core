"""馬場とペースを補正したタイム（スピード指数）。

走破タイムは、距離・コース・その日の馬場・ペースで大きく変わる。重馬場で
このペースならこのタイム、という結果を「良馬場・平均ペースならどのくらいか」
に直し、レースをまたいで比べられるようにする。

すべて **1000mあたりの秒** で扱う（距離で水準が違うため）。

    レースの偏差 = 上位3頭の平均タイム − コース（競馬場×芝ダ×距離）の中央値
                 = 馬場差（日×競馬場×芝ダ） + クラス差 + ペースの影響 + 誤差

この3つを交互に推定して切り分ける。クラス差は推定のためだけに使い、
馬の指数からは引かない（強いクラスの速い時計は強さそのもの）。

    指数（figure） = −（走破タイム − コース中央値 − 馬場差 − ペースの影響）
                     正ほど速い。単位は 1000mあたりの秒
    上がり指数     = 上がり3F について同じことをしたもの（正ほど速い）

ペースの影響は位置取り（前・中・後ろ）で違うので、位置別に係数を持つ。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import pace as pace_mod

POS_BINS = (0.25, 0.6)          # 最初のコーナーの位置÷頭数で 前 / 中 / 後ろ
FIG_RANGE = (-5.0, 5.0)         # 1000mあたり±5秒を超える値は外れ値として捨てる


def _age_class(name: pd.Series) -> pd.Series:
    n = name.astype(str)
    return np.select([n.str.contains('2歳'), n.str.contains('3歳以上|4歳以上'), n.str.contains('3歳')],
                     ['2歳', '古馬', '3歳'], '古馬')


def _alternate(dev: pd.Series, day: pd.Series, cls: pd.Series, pz: pd.Series,
               iters: int = 6):
    """dev = 馬場差[day] + クラス差[cls] + beta*pz を交互に推定する。"""
    variant = pd.Series(0.0, index=dev.index)
    beta = 0.0
    for _ in range(iters):
        r = dev - variant - beta * pz
        cls_eff = r.groupby(cls).transform('mean')
        r2 = dev - variant - cls_eff
        ok = pz.notna() & r2.notna()
        beta = float(np.polyfit(pz[ok], r2[ok], 1)[0]) if ok.sum() > 100 else 0.0
        r3 = dev - cls_eff - beta * pz.fillna(0)
        variant = r3.groupby(day).transform('median')
    return variant, cls_eff, beta


def add_figures(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """results（read_results_enriched の形）に figure / l3f_figure と内訳を付ける。"""
    rp = pace_mod.add_race_pace(races)[['race_id', 'pace_z']]
    df = results.merge(rp, on='race_id', how='left')
    df = df[~df['race_name'].astype(str).str.contains('障害')].copy()
    df['race_date'] = pd.to_datetime(df['race_date'])
    km = pd.to_numeric(df['distance'], errors='coerce') / 1000.0
    t = pd.to_numeric(df['time_seconds'], errors='coerce') / km
    l3 = pd.to_numeric(df['last_3f'], errors='coerce')
    fin = pd.to_numeric(df['finish_position'], errors='coerce')
    n = df.groupby('race_id')['horse_number'].transform('size')
    pos = pd.to_numeric(df['first_corner'], errors='coerce') / n
    df['pos_group'] = np.select([pos <= POS_BINS[0], pos >= POS_BINS[1]], ['前', '後ろ'], '中')
    df.loc[pos.isna(), 'pos_group'] = '中'

    course = [df['venue_code'], df['surface'], df['distance']]
    df['_t_dev'] = t - t.groupby(course).transform('median')
    df['_l_dev'] = l3 - l3.groupby(course).transform('median')

    # レース単位の代表値（上位3頭の平均。上がりは全頭の中央値）
    top = df[fin <= 3]
    race = pd.DataFrame({
        't': top.groupby('race_id')['_t_dev'].mean(),
        'l': df.groupby('race_id')['_l_dev'].median()})
    meta = df.drop_duplicates('race_id').set_index('race_id')
    race['day'] = (meta['race_date'].dt.strftime('%Y%m%d') + meta['venue_code'].astype(str)
                   + meta['surface'].astype(str)).reindex(race.index)
    race['cls'] = (meta['level_score'].fillna(0).astype(int).astype(str) + '_'
                   + pd.Series(_age_class(meta['race_name']), index=meta.index)
                   + meta['surface'].astype(str)).reindex(race.index)
    race['pz'] = meta['pace_z'].reindex(race.index)
    race = race.dropna(subset=['t'])

    v_t, c_t, b_t = _alternate(race['t'], race['day'], race['cls'], race['pz'])
    v_l, c_l, b_l = _alternate(race['l'], race['day'], race['cls'], race['pz'])
    df['track_variant'] = df['race_id'].map(v_t)
    df['l3f_variant'] = df['race_id'].map(v_l)

    # ペースの影響は位置取りで違う。馬場差を除いた馬ごとの偏差で、位置別に係数を推定
    pz = df['pace_z']
    df['_t_res'] = df['_t_dev'] - df['track_variant']
    df['_l_res'] = df['_l_dev'] - df['l3f_variant']
    df['pace_adj'] = 0.0
    df['l3f_pace_adj'] = 0.0
    for g, sub in df.groupby('pos_group'):
        ok = sub['pace_z'].notna() & sub['_t_res'].notna() & sub['_l_res'].notna()
        bt = np.polyfit(sub.loc[ok, 'pace_z'], sub.loc[ok, '_t_res'], 1)[0]
        bl = np.polyfit(sub.loc[ok, 'pace_z'], sub.loc[ok, '_l_res'], 1)[0]
        df.loc[sub.index, 'pace_adj'] = bt * pz[sub.index].fillna(0)
        df.loc[sub.index, 'l3f_pace_adj'] = bl * pz[sub.index].fillna(0)

    df['figure'] = -(df['_t_res'] - df['pace_adj'])
    df['l3f_figure'] = -(df['_l_res'] - df['l3f_pace_adj'])
    # 大差のしんがり・競走中止に近いものは指数として意味がない
    for c in ('figure', 'l3f_figure'):
        df.loc[(df[c] < FIG_RANGE[0]) | (df[c] > FIG_RANGE[1]), c] = np.nan
    df.attrs['beta'] = {'race_time': b_t, 'race_l3f': b_l}
    return df.drop(columns=['_t_dev', '_l_dev', '_t_res', '_l_res'])


def add_horse_figure_features(df: pd.DataFrame) -> pd.DataFrame:
    """前走までの指数を馬ごとに集計し、メンバー内の順位も付ける。"""
    df = df.sort_values(['horse_id', 'race_date', 'race_id']).copy()
    h = df.groupby('horse_id')
    for col, name in (('figure', 'fig'), ('l3f_figure', 'l3fig')):
        s = h[col]
        df[f'{name}_p1'] = s.shift(1)
        df[f'{name}_r3'] = s.transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
        df[f'{name}_best'] = s.transform(lambda x: x.shift().expanding().max())
        for k in ('p1', 'r3', 'best'):
            c = f'{name}_{k}'
            df[f'{c}_rank'] = df.groupby('race_id')[c].rank(ascending=False, method='min')
            df[f'{c}_gap'] = df[c] - df.groupby('race_id')[c].transform('max')
    return df
