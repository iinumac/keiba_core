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


# ---------------------------------------------------------------------------
# 距離の変換
#
# 指数は「その距離を走る馬の中央値より何秒速いか」なので、距離が違うと同じ馬でも
# 値が変わる（スプリンターは距離が延びると相手より大きく失速する）。そこで同じ馬が
# 同じ芝ダで距離を変えた2走の組から「元の距離の指数 → 今回の距離の指数」を学び、
# 過去の全走を「今回の距離で走ったらこのくらい」という値に直してから集計する。
#
#   今回の距離での指数 ≈ a + b × 元の距離での指数      （芝ダ × 元の距離帯 × 今回の距離帯ごと）
#
# b は元の指数がどれだけ当てになるか。距離が離れるほど小さくなり、平均の方へ寄せて割り引く。
# 学習は scripts/figure_transfer.py。係数は models/figure_transfer.json。
DIST_EDGES = (0, 1250, 1450, 1650, 1850, 2050, 2450, 99999)
DIST_LABELS = ('〜1200', '1300〜1400', '1500〜1600', '1700〜1800', '1900〜2000', '2100〜2400', '2500〜')
TRANSFER_MAX_GAP_DAYS = 365
"""学習に使う2走の間隔の上限（日）。間が空くと成長・衰えが混ざる。"""
TRANSFER_MIN_PAIRS = 300
"""組ごとに係数を当てる最低の組数。足りない組は「距離帯がいくつ離れているか」でまとめた係数を使う。"""


def distance_band(distance) -> np.ndarray:
    """距離帯の番号（0〜6）。"""
    d = pd.to_numeric(pd.Series(np.asarray(distance)), errors='coerce')
    return (pd.cut(d, DIST_EDGES, labels=False, right=False)).to_numpy()


def transfer_pairs(fig: pd.DataFrame, max_gap_days: int = TRANSFER_MAX_GAP_DAYS) -> pd.DataFrame:
    """同じ馬・同じ芝ダの2走の組（元 → 今回）。fig は add_figures の結果。"""
    cols = ['horse_id', 'race_id', 'race_date', 'surface', 'distance', 'figure', 'l3f_figure']
    f = fig[cols].dropna(subset=['figure']).copy()
    f['horse_id'] = f['horse_id'].astype(str)
    f['band'] = distance_band(f['distance'])
    p = f.merge(f, on=['horse_id', 'surface'], suffixes=('_s', '_t'))
    gap = (p['race_date_t'] - p['race_date_s']).dt.days
    return p[(gap > 0) & (gap <= max_gap_days)]


def _fit_cells(p: pd.DataFrame, keys: list, col: str) -> pd.DataFrame:
    x, y = p[f'{col}_s'], p[f'{col}_t']
    ok = x.notna() & y.notna()
    g = pd.DataFrame({'x': x[ok], 'y': y[ok], 'xy': x[ok] * y[ok], 'xx': x[ok] ** 2, 'yy': y[ok] ** 2,
                      **{k: p.loc[ok, k] for k in keys}}).groupby(keys)
    s = g.agg(n=('x', 'size'), mx=('x', 'mean'), my=('y', 'mean'), mxy=('xy', 'mean'),
              mxx=('xx', 'mean'), myy=('yy', 'mean'))
    vx, vy, cxy = s['mxx'] - s['mx'] ** 2, s['myy'] - s['my'] ** 2, s['mxy'] - s['mx'] * s['my']
    s['b'] = cxy / vx
    s['a'] = s['my'] - s['b'] * s['mx']
    s['sd'] = np.sqrt(np.clip(vy - s['b'] ** 2 * vx, 0, None))
    s['r'] = cxy / np.sqrt(vx * vy)
    return s[['n', 'a', 'b', 'sd', 'r']].reset_index()


def fit_distance_transfer(pairs: pd.DataFrame) -> dict:
    """距離の変換の係数を学ぶ。{'figure': 表, 'l3f_figure': 表} を返す（表は JSON にできる dict のリスト）。"""
    p = pairs.copy()
    p['step'] = p['band_t'] - p['band_s']
    out = {}
    for col in ('figure', 'l3f_figure'):
        cell = _fit_cells(p, ['surface', 'band_s', 'band_t'], col)
        step = _fit_cells(p, ['surface', 'step'], col)
        out[col] = {'cell': cell.to_dict('records'), 'step': step.to_dict('records')}
    return out


def _coef_table(transfer: dict, col: str) -> pd.DataFrame:
    """芝ダ × 元の距離帯 × 今回の距離帯 → (a, b)。組数の足りない組は距離帯の差でまとめた係数。"""
    keys = ['surface', 'band_s', 'band_t', 'n', 'a', 'b', 'sd', 'r']
    cell = pd.DataFrame(transfer[col]['cell'], columns=keys)
    step = pd.DataFrame(transfer[col]['step'])
    full = pd.MultiIndex.from_product([sorted(set(step['surface']) | set(cell['surface'])), range(len(DIST_LABELS)),
                                       range(len(DIST_LABELS))], names=['surface', 'band_s', 'band_t']).to_frame(index=False)
    full['step'] = full['band_t'] - full['band_s']
    full = full.merge(cell, on=['surface', 'band_s', 'band_t'], how='left')
    full = full.merge(step.rename(columns={'n': 'n2', 'a': 'a2', 'b': 'b2', 'sd': 'sd2', 'r': 'r2'}),
                      on=['surface', 'step'], how='left')
    thin = full['n'].isna() | (full['n'] < TRANSFER_MIN_PAIRS)
    for k in ('a', 'b', 'sd'):
        full.loc[thin, k] = full.loc[thin, f'{k}2']
    return full[['surface', 'band_s', 'band_t', 'a', 'b', 'sd']]


def converted_summary(targets: pd.DataFrame, history: pd.DataFrame, transfer: dict) -> pd.DataFrame:
    """過去走の指数を今回の距離に直して集計する。

    targets: 1行が1頭の出走（key 列・horse_id・race_date・surface・distance）。
    history: add_figures の結果（horse_id・race_date・surface・distance・figure・l3f_figure）。
    同じ芝ダの過去走だけを使う（芝ダが違う・無い馬は空欄）。返す列は
    fig_r3（近3走の平均）・fig_best（最高）・l3fig_r3（上がりの近3走平均）。
    """
    if len(targets) > 50000:
        # 全出走を一度に結ぶとメモリが足りないので、馬ごとに分けて計算する
        hid = targets['horse_id'].astype(str)
        part = pd.util.hash_pandas_object(hid, index=False) % (len(targets) // 25000 + 1)
        return pd.concat([converted_summary(targets[part == i], history, transfer)
                          for i in sorted(part.unique())]).reindex(targets.index)
    t = targets.copy()
    t['_k'] = np.arange(len(t))
    t['horse_id'] = t['horse_id'].astype(str)
    t['band_t'] = distance_band(t['distance'])
    h = history[['horse_id', 'race_date', 'surface', 'distance', 'figure', 'l3f_figure']].copy()
    h['horse_id'] = h['horse_id'].astype(str)
    h['band_s'] = distance_band(h['distance'])
    j = t[['_k', 'horse_id', 'race_date', 'surface', 'band_t']].merge(
        h.rename(columns={'race_date': 'date_s'}), on=['horse_id', 'surface'])
    j = j[j['date_s'] < j['race_date']].sort_values(['_k', 'date_s'])
    for col, name in (('figure', 'fig'), ('l3f_figure', 'l3fig')):
        c = _coef_table(transfer, col)
        jj = j.merge(c, on=['surface', 'band_s', 'band_t'], how='left')
        jj[name] = jj['a'] + jj['b'] * jj[col]
        g = jj.dropna(subset=[name]).groupby('_k')[name]
        t[f'{name}_r3'] = t['_k'].map(g.apply(lambda x: x.tail(3).mean()))
        if name == 'fig':
            t['fig_best'] = t['_k'].map(g.max())
    return t.drop(columns=['_k', 'band_t'])


def load_transfer(path=None) -> dict:
    """models/figure_transfer.json（scripts/figure_transfer.py が作る）。無ければ None。"""
    import json
    from . import config
    p = path or (config.MODEL_DIR / 'figure_transfer.json')
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
