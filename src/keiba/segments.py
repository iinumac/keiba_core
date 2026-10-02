"""レースを区分して、荒れ具合と回収率を見る

「どのレースが荒れるか」と「どのレースが儲かるか」は別物。
荒れても市場がそれを織り込んでいれば配当に反映され、優位にはならない。
だから荒れ具合（記述）と回収率（収益）を必ず並べて見る。

区分の軸

  馬場状態  良 / 稍重 / 重 / 不良
  距離      ~1500 / 1600-1900 / 2000-2300 / 2400-2500 / 2600~
  芝ダ      芝 / ダート
  グレード  G1 / G2 / G3 / OP・3勝 / 1-2勝 / 未勝利 / 新馬
  人気帯    1番人気 / 2-3 / 4-5 / 6-8 / 9-12 / 13-
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

DIST_BINS = [0, 1500, 1900, 2300, 2500, 9999]
DIST_LABELS = ['~1500', '1600-1900', '2000-2300', '2400-2500', '2600~']

POP_BINS = [0, 1, 3, 5, 8, 12, 99]
POP_LABELS = ['1番人気', '2-3', '4-5', '6-8', '9-12', '13-']

# 賞金ベースの race_level を、名前から分からないレースの補完に使う
LEVEL_TO_GRADE = {'S': 'G1', 'A': 'G2', 'B': 'OP・3勝', 'C': 'OP・3勝',
                  'D': '1-2勝', 'E': '未勝利'}

GRADE_ORDER = ['G1', 'G2', 'G3', 'OP・3勝', '1-2勝', '未勝利', '新馬', '不明']


def race_grade(race_name: str, race_level: Optional[str] = None) -> str:
    """レース名からグレードを判定する。

    特別競走は名前にクラスが出ないため（「知床特別」など、実データで
    約10,900レース）、賞金ベースの `race_level` で補完する。
    """
    n = str(race_name)
    if re.search(r'\((?:G1|GI)\)', n):
        return 'G1'
    if re.search(r'\((?:G2|GII)\)', n):
        return 'G2'
    if re.search(r'\((?:G3|GIII)\)', n):
        return 'G3'
    if '新馬' in n:
        return '新馬'
    if '未勝利' in n:
        return '未勝利'
    return LEVEL_TO_GRADE.get(race_level, '不明')


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """grade / dist_band / pop_band を付けて返す。"""
    out = df.copy()
    if 'race_name' in out.columns:
        lv = out['race_level'] if 'race_level' in out.columns else None
        out['grade'] = [race_grade(n, l) for n, l in
                        zip(out['race_name'], lv if lv is not None else [None] * len(out))]
    if 'distance' in out.columns:
        out['dist_band'] = pd.cut(pd.to_numeric(out['distance'], errors='coerce'),
                                  DIST_BINS, labels=DIST_LABELS)
    if 'popularity' in out.columns:
        out['pop_band'] = pd.cut(pd.to_numeric(out['popularity'], errors='coerce'),
                                 POP_BINS, labels=POP_LABELS)
    return out


def upset_table(df: pd.DataFrame, by: str, min_races: int = 150) -> pd.DataFrame:
    """荒れ具合。**儲かるかどうかとは別物**なので roi_table と並べて見ること。"""
    d = df[df['finish_position'].notna()].copy()
    d['is_top3'] = (d['finish_position'] <= 3).astype(int)
    d['pop'] = pd.to_numeric(d['popularity'], errors='coerce')

    rows = []
    for k, g in d.groupby(by, observed=True):
        n = g['race_id'].nunique()
        if n < min_races:
            continue
        fav = g[g['pop'] == 1]
        rows.append({
            by: k,
            'レース数': n,
            '6番人気以下の3着内率%': round(100 * g[g['pop'] >= 6]['is_top3'].mean(), 2),
            '1番人気の3着内率%': round(100 * fav['is_top3'].mean(), 1),
            '1番人気の勝率%': round(100 * (fav['finish_position'] == 1).mean(), 1),
            '3着内の平均人気': round(g[g['is_top3'] == 1]['pop'].mean(), 2),
        })
    return pd.DataFrame(rows)


def roi_table(df: pd.DataFrame, by: str, payout_col: str = 'fuku',
              min_rows: int = 500) -> pd.DataFrame:
    """区分ごとの複勝回収率。信頼区間つき。"""
    rows = []
    for k, g in df.groupby(by, observed=True):
        if len(g) < min_rows:
            continue
        ret = (g[payout_col] - 100).to_numpy(float)
        se = ret.std(ddof=1) / np.sqrt(len(g))
        lo = 100 * (1 + (ret.mean() - 1.96 * se) / 100)
        hi = 100 * (1 + (ret.mean() + 1.96 * se) / 100)
        rows.append({
            by: k, '頭数': len(g),
            '的中率%': round(100 * (g[payout_col] > 0).mean(), 1),
            '回収率%': round(100 * g[payout_col].sum() / (len(g) * 100), 1),
            '95%区間': f'{lo:.0f}〜{hi:.0f}',
            '黒字': '◯' if lo > 100 else '×',
        })
    return pd.DataFrame(rows)


def cross_roi(df: pd.DataFrame, by: str, payout_col: str = 'fuku',
              order: Optional[Sequence] = None, min_rows: int = 500) -> pd.DataFrame:
    """区分 × 人気帯 の回収率。人気帯の勾配とセグメント差を同時に見る。"""
    keys = order if order is not None else sorted(df[by].dropna().unique().tolist())
    rows = []
    for k in keys:
        sub = df[df[by] == k]
        row = {by: k}
        for pb in POP_LABELS:
            g = sub[sub['pop_band'] == pb]
            row[pb] = (f'{100 * g[payout_col].sum() / (len(g) * 100):.0f}%'
                       if len(g) >= min_rows else '-')
        rows.append(row)
    return pd.DataFrame(rows)


def overvalued_after_last_race(df: pd.DataFrame) -> pd.Series:
    """市場が前走に過剰反応して買われすぎている馬を判定する。

    同じ人気帯の平均回収率と比べて有意に劣るグループ。実データでは
    この集団の複勝回収率は 54.8%（全体 74.5%）。

      前走13番人気以下 × 14着以下   超過 -10.9pt
      前走13番人気以下 × 2-3着      超過  -6.6pt （人気薄で激走した反動）
      前走1番人気 × 9-13着          超過  -5.9pt （1番人気で大敗した反動）

    除外すると全体の回収率は 74.5% → 75.8%。
    効果は人気薄に集中する（9番人気以下で 67.0% → 69.2%）。
    """
    p1_pop = pd.to_numeric(df.get('p1_pop'), errors='coerce')
    p1_fin = pd.to_numeric(df.get('p1_fin'), errors='coerce')
    return (((p1_pop >= 13) & (p1_fin >= 14))
            | ((p1_pop >= 13) & p1_fin.between(2, 3))
            | ((p1_pop == 1) & p1_fin.between(9, 13))).fillna(False)


def attach_place_payout(df: pd.DataFrame, payouts: pd.DataFrame) -> pd.DataFrame:
    """各馬に複勝の払戻を付ける（3着内でなければ0）。"""
    f = payouts[payouts['bet_type'] == '複勝'].copy()
    f['horse_number'] = f['horse_numbers'].apply(lambda x: int(x[0]))
    f = f[['race_id', 'horse_number', 'payout']].rename(columns={'payout': 'fuku'})
    out = df.copy()
    out['race_id'] = out['race_id'].astype(str)
    f['race_id'] = f['race_id'].astype(str)
    out = out.merge(f, on=['race_id', 'horse_number'], how='left')
    out['fuku'] = out['fuku'].fillna(0)
    # 複勝が発売されていないレース（4頭以下）は対象外
    return out[out['race_id'].isin(set(f['race_id']))]
