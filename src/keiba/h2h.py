"""対戦比較による序列づけ

同じレースを走った馬どうしの着差から、まだ対戦していない組み合わせも含めて
「AはBより何秒速いか」を推定し、出走馬を並べる。

    段階1 直接対決     A と B が同じレースを走った        m(A,B)
    段階2 共通の相手   A–C と B–C が走った                m(A,C) - m(B,C)
    段階3 2頭はさむ   A–C, C–D, D–B が走った             m(A,C) + m(C,D) + m(D,B)

m(X,Y) は「X が Y に何秒先着したか」の平均（負なら負け）。
段階1があればそれだけを使い、無いときだけ次の段階に進む。

着差は走破タイムの差で、±MARGIN_CAP 秒で打ち切る。大敗した馬は流して
入線していることが多く、2秒差と5秒差の違いに意味は無いため。

判定日より前のレースだけを使う（`as_of` 当日のレースは含めない）。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

MARGIN_CAP = 2.0        # 秒。着差をこの範囲に打ち切る
LOOKBACK_DAYS = 730     # 何日前までの対戦を見るか
MAX_PATHS = 200         # 段階3で平均を取る経路の上限（多すぎる馬の計算量を抑える）

# 推定着差 → 先着確率（ロジスティック: P = 1 / (1 + exp(-k * 着差))）。
# scripts/h2h_eval.py で 2025-01〜 の 3,000 レース・14万組から当てはめた値。
LEVEL_SLOPE = {1: 0.70, 2: 0.64, 3: 0.79}


@dataclass
class Estimate:
    margin: float       # A が B に何秒先着するか（推定）
    level: int          # 1 直接 / 2 共通の相手 / 3 2頭はさむ / 0 不明
    n: int              # 根拠になった対戦（段階1）または相手・経路（段階2,3）の数


class History:
    """過去の対戦を引けるようにした索引。一度作って何度でも問い合わせる。"""

    def __init__(self, races: pd.DataFrame, results: pd.DataFrame):
        r = results[['race_id', 'horse_id', 'finish_position', 'time_seconds']].copy()
        r['race_id'] = r['race_id'].astype(str)
        r = r.dropna(subset=['horse_id', 'finish_position', 'time_seconds'])
        meta = races[['race_id', 'date', 'race_name']].copy()
        meta['race_id'] = meta['race_id'].astype(str)
        # 障害は平地と強さの物差しが違うので使わない（surface は芝/ダートのままなので名前で判定）
        meta = meta[~meta['race_name'].astype(str).str.contains('障害')]
        meta['date'] = pd.to_datetime(meta['date'], errors='coerce')
        r = r.merge(meta[['race_id', 'date']], on='race_id').dropna(subset=['date'])

        self.race_date: Dict[str, np.datetime64] = dict(zip(meta['race_id'], meta['date'].values))
        self.race_rows: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
        for rid, hid, pos, t in r[['race_id', 'horse_id', 'finish_position',
                                   'time_seconds']].itertuples(index=False):
            self.race_rows[rid][str(hid)] = (float(t), float(pos))
        self.horse_races: Dict[str, List[Tuple[np.datetime64, str]]] = defaultdict(list)
        for hid, d, rid in r[['horse_id', 'date', 'race_id']].itertuples(index=False):
            self.horse_races[str(hid)].append((np.datetime64(d), rid))
        for v in self.horse_races.values():
            v.sort()
        self._cache: Dict[Tuple[str, np.datetime64], Dict[str, Tuple[float, int, int]]] = {}

    def opponents(self, horse: str, as_of, lookback_days: int = LOOKBACK_DAYS
                  ) -> Dict[str, Tuple[float, int, int]]:
        """horse が as_of より前に走った相手ごとの (平均着差, 対戦数, 先着数)。"""
        end = np.datetime64(pd.Timestamp(as_of), 'ns')
        key = (horse, end)
        if key in self._cache:
            return self._cache[key]
        start = end - np.timedelta64(lookback_days, 'D')
        acc: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0, 0])
        for d, rid in self.horse_races.get(horse, ()):
            if d < start:
                continue
            if d >= end:
                break
            rows = self.race_rows[rid]
            t0, p0 = rows[horse]
            for other, (t1, p1) in rows.items():
                if other == horse:
                    continue
                a = acc[other]
                a[0] += max(-MARGIN_CAP, min(MARGIN_CAP, t1 - t0))
                a[1] += 1
                a[2] += p0 < p1
        out = {k: (s / n, n, w) for k, (s, n, w) in acc.items()}
        self._cache[key] = out
        return out

    def meetings(self, a: str, b: str, as_of, lookback_days: int = LOOKBACK_DAYS
                 ) -> List[dict]:
        """a と b が同じレースを走った記録（確認用）。"""
        end = pd.Timestamp(as_of)
        start = end - pd.Timedelta(days=lookback_days)
        rows = []
        for d, rid in self.horse_races.get(a, ()):
            if not (start <= pd.Timestamp(d) < end):
                continue
            rr = self.race_rows[rid]
            if b in rr:
                (ta, pa), (tb, pb) = rr[a], rr[b]
                rows.append({'date': pd.Timestamp(d).date(), 'race_id': rid,
                             'pos_a': int(pa), 'pos_b': int(pb), 'margin': round(tb - ta, 1)})
        return rows

    def clear_cache(self):
        self._cache.clear()


def estimate(hist: History, a: str, b: str, as_of, max_level: int = 3) -> Estimate:
    """A が B に何秒先着するかを、段階1→2→3の順に探して推定する。"""
    oa = hist.opponents(a, as_of)
    if b in oa:
        m, n, _ = oa[b]
        return Estimate(m, 1, n)
    ob = hist.opponents(b, as_of)
    if max_level >= 2:
        common = oa.keys() & ob.keys()
        if common:
            diffs = [oa[c][0] - ob[c][0] for c in common]
            return Estimate(float(np.mean(diffs)), 2, len(common))
    if max_level >= 3 and oa and ob:
        # A–C–D–B。C は A の相手、D は B の相手で、C と D が対戦している
        total, n = 0.0, 0
        for c, (m_ac, _, _) in oa.items():
            oc = hist.opponents(c, as_of)
            for d in oc.keys() & ob.keys():
                total += m_ac + oc[d][0] - ob[d][0]
                n += 1
                if n >= MAX_PATHS:
                    break
            if n >= MAX_PATHS:
                break
        if n:
            return Estimate(total / n, 3, n)
    return Estimate(0.0, 0, 0)


def win_prob(e: Estimate) -> float:
    """推定着差を、A が B に先着する確率に直す。根拠が無ければ 0.5。"""
    if e.level == 0:
        return 0.5
    return float(1 / (1 + np.exp(-LEVEL_SLOPE[e.level] * e.margin)))


def rank(hist: History, horses: Iterable[str], as_of, names: Optional[Dict[str, str]] = None,
         max_level: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """出走馬を並べる。

    score は「他の各馬に先着する確率」の平均（0.5 が中位）。
    根拠の無い組み合わせは 0.5 として扱うので、比較できた相手の数 `known` も見ること。

    戻り値は (順位表, 組み合わせごとの推定)。
    """
    horses = [str(h) for h in horses]
    names = names or {}
    pairs = []
    for i, a in enumerate(horses):
        for b in horses[i + 1:]:
            e = estimate(hist, a, b, as_of, max_level)
            p = win_prob(e)
            pairs.append({'a': a, 'b': b, 'margin': e.margin, 'level': e.level,
                          'n': e.n, 'p_a': p})
    pw = pd.DataFrame(pairs, columns=['a', 'b', 'margin', 'level', 'n', 'p_a'])
    # 両方向にして集計する
    rev = pw.rename(columns={'a': 'b', 'b': 'a'}).assign(margin=lambda d: -d['margin'],
                                                         p_a=lambda d: 1 - d['p_a'])
    both = pd.concat([pw, rev], ignore_index=True)
    both['known'] = both['level'] > 0
    table = both.groupby('a').agg(
        score=('p_a', 'mean'),
        wins=('p_a', lambda s: int((s > 0.5).sum())),
        known=('known', 'sum'),
        lv1=('level', lambda s: int((s == 1).sum())),
        lv2=('level', lambda s: int((s == 2).sum())),
        lv3=('level', lambda s: int((s == 3).sum())),
    ).reset_index().rename(columns={'a': 'horse_id'})
    table['horse_name'] = table['horse_id'].map(names)
    table = table.sort_values(['score', 'known'], ascending=False).reset_index(drop=True)
    table.insert(0, 'rank', range(1, len(table) + 1))
    pw['a_name'] = pw['a'].map(names)
    pw['b_name'] = pw['b'].map(names)
    return table, pw
