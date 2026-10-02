"""三連複の買い方を決める

## 考え方

全レースを買うと控除率（三連複25%）を毎回払う。**買わないレースは
回収率100%**なので、勝ち筋は「買うレースを選ぶ」「点数を抑える」に尽きる。

## データから分かったこと

実データ（2025-2026、5,246レース）で、買い方ごとに全レース購入した場合:

    軸2頭→相手5頭(5点) 84.2%   軸2頭→相手3頭(3点) 82.6%
    軸1頭→相手4頭(6点) 81.9%   4頭BOX(4点)        81.6%
    軸1頭→相手5頭(10点)78.4%   5頭BOX(10点)       76.8%

**点数が多いほど回収率が下がる。** 期待値の低い買い目まで広げるため。

混戦のレースは「点数を増やせば当たる」のではなく、**どの買い方でも悪い**。
上位3頭の確率合計が最も低い層では、最良の買い方でも 76〜78% にとどまる。
したがって混戦では**点数を増やすのではなく買わない**。

## 推奨ルール

レースの「堅さ」を上位3頭の予測確率の合計で測る（`confidence`）。

    confidence < 1.4            買わない
    過大評価の馬が上位3頭にいる  買わない（segments.overvalued）
    1.4 <= confidence < 1.6     軸1頭→相手4頭（6点）
    confidence >= 1.6           軸2頭→相手3頭（3点）

2025年で閾値を決め、2026年で検証した結果:

    1.4 軸1頭→相手4頭＋除外   探索 89.0% → 検証 92.8%（参加率 21.6%）
    1.5 軸2頭→相手3頭＋除外   探索 93.8% → 検証 96.6%（参加率 17.1%）
    1.6 軸2頭→相手3頭＋除外   探索 88.9% → 検証 100.5%（参加率 11.8%）

**ただし 100.5% の95%信頼区間は 78〜123 で、黒字とは言い切れない。**
参加率を絞るほどレース数が減り、確認は難しくなる。

なお 3頭BOX（1点）は探索で高く出るが検証で崩れる（102.6% → 74.2%）。
的中率が15.8%と低く分散が大きいため。**少数レースの高回収率は信用しないこと。**
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

UNIT = 100
MAX_POINTS = 10
"""1レースの上限点数。これを超える買い方は期待値が薄くなる。"""


def box(order: Sequence[int], n: int) -> Set[FrozenSet[int]]:
    """上位 n 頭のBOX。"""
    return {frozenset(c) for c in itertools.combinations(order[:n], 3)}


def nagashi(order: Sequence[int], axis: int, partners: int) -> Set[FrozenSet[int]]:
    """軸 axis 頭 → 相手 partners 頭の流し。"""
    a, o = list(order[:axis]), list(order[axis:axis + partners])
    if len(a) < axis or len(o) < partners:
        return set()
    if axis == 1:
        return {frozenset([a[0], x, y]) for x, y in itertools.combinations(o, 2)}
    return {frozenset([a[0], a[1], x]) for x in o}


SHAPES = {
    '3頭BOX':       lambda o: box(o, 3),
    '軸2頭→相手3頭': lambda o: nagashi(o, 2, 3),
    '4頭BOX':       lambda o: box(o, 4),
    '軸2頭→相手5頭': lambda o: nagashi(o, 2, 5),
    '軸1頭→相手4頭': lambda o: nagashi(o, 1, 4),
    '軸1頭→相手5頭': lambda o: nagashi(o, 1, 5),
    '5頭BOX':       lambda o: box(o, 5),
}
"""点数が10以内の買い方。値は「スコア順の馬番リスト → 買い目集合」。"""


@dataclass
class Plan:
    """1レースの買い方。`shape` が None なら買わない。"""
    race_id: str
    confidence: float
    shape: Optional[str]
    combos: Set[FrozenSet[int]]
    reason: str = ''

    @property
    def points(self) -> int:
        return len(self.combos)

    @property
    def cost(self) -> int:
        return self.points * UNIT


MIN_CONFIDENCE = 1.4
"""これを下回るレースは買わない。上位3頭の予測確率の合計。

実データで、この値が低い層はどの買い方でも 76〜78% にとどまる。
点数を増やしても改善しない。
"""

SHAPE_BY_CONFIDENCE = ((1.6, '軸2頭→相手3頭'), (MIN_CONFIDENCE, '軸1頭→相手4頭'))
"""（下限, 買い方）。堅いほど点数を絞る。"""


def plan_race(race_id: str, horse_numbers: Sequence[int], scores: Sequence[float],
              skip: bool = False, min_confidence: float = MIN_CONFIDENCE,
              min_runners: int = 8) -> Plan:
    """1レースの買い方を決める。

    Args:
        horse_numbers / scores: 同じ並びの馬番と予測確率
        skip: 買わない理由が外部で判明している場合（過大評価馬が上位にいる等）
    """
    order = [h for _, h in sorted(zip(scores, horse_numbers), reverse=True)]
    s = sorted(scores, reverse=True)
    confidence = float(np.sum(s[:3]))

    if len(order) < min_runners:
        return Plan(race_id, confidence, None, set(), '少頭数')
    if skip:
        return Plan(race_id, confidence, None, set(), '過大評価の馬が上位にいる')
    if confidence < min_confidence:
        return Plan(race_id, confidence, None, set(), '混戦（確信度が低い）')

    for lo, name in SHAPE_BY_CONFIDENCE:
        if confidence >= lo:
            combos = SHAPES[name](order)
            if combos and len(combos) <= MAX_POINTS:
                return Plan(race_id, confidence, name, combos)
    return Plan(race_id, confidence, None, set(), '買い方を決められない')


def evaluate(plans: Sequence[Plan],
             payout_index: Dict[str, List[Tuple[FrozenSet[int], int]]]) -> Dict:
    """買い方の一覧を払戻と突き合わせる。

    同着があると1レースで複数の買い目が当たるので合計する。
    """
    bet = [p for p in plans if p.shape]
    cost = sum(p.cost for p in bet)
    returns, hits, payout = [], 0, 0
    for p in bet:
        wins = payout_index.get(p.race_id, [])
        got = sum(v for c, v in wins if c in p.combos)
        payout += got
        returns.append(got - p.cost)
        hits += got > 0

    out = {'対象レース': len(plans), '購入レース': len(bet),
           '参加率%': round(100 * len(bet) / len(plans), 1) if plans else 0.0,
           '平均点数': round(np.mean([p.points for p in bet]), 1) if bet else 0.0,
           '的中率%': round(100 * hits / len(bet), 1) if bet else 0.0,
           '回収率%': round(100 * payout / cost, 1) if cost else 0.0,
           '収支': payout - cost}
    if len(returns) > 1:
        a = np.asarray(returns, float)
        se = a.std(ddof=1) / np.sqrt(len(a))
        per = cost / len(bet)
        out['95%区間'] = (f'{100 * (1 + (a.mean() - 1.96 * se) / per):.0f}〜'
                        f'{100 * (1 + (a.mean() + 1.96 * se) / per):.0f}')
    return out
