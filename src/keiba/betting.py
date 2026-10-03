"""馬券の買い方を決める

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

## 券種で点数が変わる

同じ「上位3頭を買う」でも、三連複は3頭組、馬連・ワイドは2頭組なので
点数が減る。同じ1,109レースを券種別に通した結果:

    三連複 3頭BOX(1点)     的中10.0% 回収93.4%
    三連複 軸2頭→3(3点)    的中20.7% 回収82.0%  ← 旧ルールの主力。最下位
    馬連  軸1頭→3(3点)    的中36.4% 回収90.7%
    ワイド 軸1頭→3(3点)    的中58.3% 回収88.1%

**同じ3点なら2頭組の券種のほうが回収率も的中率も上。**

## 推奨ルール

レースの「堅さ」を上位3頭の予測確率の合計で測る（`confidence`）。

    confidence < 1.4            買わない
    過大評価の馬が上位3頭にいる  買わない（segments.overvalued）
    実績の薄い馬が上位3頭にいる  買わない（segments.thin_record）
    1.4 <= confidence < 1.55    三連複 3頭BOX（1点）
    confidence >= 1.55          馬連 軸1頭→相手3頭（3点）

2025年で閾値を決め、2026年で検証した（対象レースは新旧で同一）:

    旧（三連複のみ） 全期間 4.4点 的中25.2% 回収84.1%  2026〜 89.8%
    新（券種で切替） 全期間 2.3点 的中29.3% 回収94.1%  2026〜 94.7%

**点数が半分になり、的中率も回収率も上がった。** 損失は 77,340円 → 15,150円。

**ただし検証でも100%には届いていない。** 低い帯の三連複1点は
95%区間が 41〜163 と極端に広く、102% を「黒字のゾーン」と読んではいけない。
**少数レースの高回収率は信用しないこと。**

採点にはステージ4型の買い目用モデル（models/model_strategy.pkl）を使う。
単勝オッズだけ、あるいは13特徴量の model_with_odds では回収率が
79〜80% にとどまり、この数字は出ない。docs/BETTING.md「採点に使うモデル」。
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


LEG = {'三連複': 3, 'ワイド': 2, '馬連': 2}
"""券種ごとの組み合わせの頭数。ワイドと馬連は2頭なので点数を絞りやすい。"""


def box(order, n: int, legs: int = 3):
    """上位 n 頭のBOX。"""
    return {frozenset(c) for c in itertools.combinations(order[:n], legs)}


def nagashi(order, axis: int, partners: int, legs: int = 3):
    """軸 axis 頭 → 相手 partners 頭の流し。"""
    a, o = list(order[:axis]), list(order[axis:axis + partners])
    if len(a) < axis or len(o) < partners:
        return set()
    if legs == 2:
        if axis != 1:
            return set()
        return {frozenset([a[0], x]) for x in o}
    if axis == 1:
        return {frozenset([a[0], x, y]) for x, y in itertools.combinations(o, 2)}
    return {frozenset([a[0], a[1], x]) for x in o}


SHAPES = {
    '三連複': {
        '3頭BOX':       lambda o: box(o, 3, 3),
        '軸2頭→相手3頭': lambda o: nagashi(o, 2, 3, 3),
        '4頭BOX':       lambda o: box(o, 4, 3),
        '軸1頭→相手4頭': lambda o: nagashi(o, 1, 4, 3),
        '軸2頭→相手5頭': lambda o: nagashi(o, 2, 5, 3),
        '5頭BOX':       lambda o: box(o, 5, 3),
    },
    'ワイド': {
        '軸1頭→相手2頭': lambda o: nagashi(o, 1, 2, 2),
        '3頭BOX':       lambda o: box(o, 3, 2),
        '軸1頭→相手3頭': lambda o: nagashi(o, 1, 3, 2),
        '軸1頭→相手4頭': lambda o: nagashi(o, 1, 4, 2),
        '4頭BOX':       lambda o: box(o, 4, 2),
    },
    '馬連': {
        '軸1頭→相手2頭': lambda o: nagashi(o, 1, 2, 2),
        '3頭BOX':       lambda o: box(o, 3, 2),
        '軸1頭→相手3頭': lambda o: nagashi(o, 1, 3, 2),
        '軸1頭→相手4頭': lambda o: nagashi(o, 1, 4, 2),
    },
}
"""券種ごとの買い方。値は「スコア順の馬番リスト → 買い目集合」。

ワイドと馬連は2頭の組み合わせなので、三連複より少ない点数で買える。
軸1頭→相手2頭なら2点。
"""


@dataclass
class Plan:
    """1レースの買い方。`shape` が None なら買わない。"""
    race_id: str
    confidence: float
    shape: Optional[str]
    combos: Set[FrozenSet[int]]
    reason: str = ''
    bet_type: str = '三連複'

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

SHAPE_BY_CONFIDENCE = (
    (1.55, '馬連', '軸1頭→相手3頭'),
    (MIN_CONFIDENCE, '三連複', '3頭BOX'),
)
"""（確信度の下限, 券種, 買い方）。確信度が高いほうを先に書く。

実データ（1,109レース、過大評価を除外し確信度1.4以上）での回収率:

    確信度       三連複3頭BOX  馬連 軸1頭→3  ワイド 軸1頭→3
    1.4-1.55       102.1%       86.8%        83.7%
    1.55-1.7        88.9%       94.1%        88.3%
    1.7-1.9         89.4%       97.3%        93.5%

確信度が低いうちは三連複の1点（当たれば大きい）、上がったら馬連に切り替える。

的中率を優先したいなら SHAPE_HIT_FIRST を渡す（下記）。
"""


SHAPE_HIT_FIRST = (
    (1.55, 'ワイド', '軸1頭→相手3頭'),
    (MIN_CONFIDENCE, '三連複', '3頭BOX'),
)
"""的中率を優先する組み方。馬連をワイドに置き換えただけ。

2025年で閾値を決め、2026年で検証した（点数はどちらも平均2.3点）:

    組み方                探索 的中/回収      検証 的中/回収    検証95%区間
    SHAPE_BY_CONFIDENCE   29.4% / 93.6%    29.1% / 94.7%    79〜110
    SHAPE_HIT_FIRST       44.0% / 93.0%    45.0% / 91.1%    79〜103

回収率は馬連、的中率はワイドが上。**どちらも検証で100%には届かない。**
回収率が同じなら的中率が高いほうが収支は早く安定するが、ここでは
3.6pt の回収率と引き換えになる。既定は回収率が高い馬連。

    plan_race(..., rules=bt.SHAPE_HIT_FIRST)   # 的中率を優先する場合
"""


def plan_race(race_id: str, horse_numbers: Sequence[int], scores: Sequence[float],
              skip=False, min_confidence: float = MIN_CONFIDENCE,
              min_runners: int = 8, rules=SHAPE_BY_CONFIDENCE) -> Plan:
    """1レースの買い方を決める。

    Args:
        horse_numbers / scores: 同じ並びの馬番と予測確率
        skip: 買わない理由が外部で判明している場合（過大評価馬が上位にいる等）。
            文字列を渡すとそれを理由として記録する
        rules: （確信度の下限, 券種, 買い方）の並び。確信度が高い順に書く
    """
    order = [h for _, h in sorted(zip(scores, horse_numbers), reverse=True)]
    s = sorted(scores, reverse=True)
    confidence = float(np.sum(s[:3]))

    if len(order) < min_runners:
        return Plan(race_id, confidence, None, set(), '少頭数')
    if skip:
        reason = skip if isinstance(skip, str) else '過大評価の馬が上位にいる'
        return Plan(race_id, confidence, None, set(), reason)
    if confidence < min_confidence:
        return Plan(race_id, confidence, None, set(), '混戦（確信度が低い）')

    for lo, bet_type, name in rules:
        if confidence >= lo:
            combos = SHAPES[bet_type][name](order)
            if combos and len(combos) <= MAX_POINTS:
                return Plan(race_id, confidence, name, combos,
                            bet_type=bet_type)
    return Plan(race_id, confidence, None, set(), '買い方を決められない')


def payout_index(payouts: pd.DataFrame, bet_types=None) -> Dict[str, Dict]:
    """払戻テーブルを `evaluate` に渡す形にする。券種 -> race_id -> [(組, 払戻)]。

    券種を混ぜて買うので、必ず券種ごとに分けて持つ。1つの券種だけの
    平らな辞書を作ると、他の券種の買い目が黙って外れ扱いになる。
    """
    bet_types = bet_types or list(LEG)
    out: Dict[str, Dict] = {b: {} for b in bet_types}
    sub = payouts[payouts['bet_type'].isin(bet_types)]
    for b, rid, hn, p in zip(sub['bet_type'], sub['race_id'],
                             sub['horse_numbers'], sub['payout']):
        out[b].setdefault(str(rid), []).append(
            (frozenset(int(x) for x in hn), int(p)))
    return out


def evaluate(plans: Sequence[Plan], payout_index) -> Dict:
    """買い方の一覧を払戻と突き合わせる。

    同着があると1レースで複数の買い目が当たるので合計する。

    Args:
        payout_index: race_id -> [(組み合わせ, 払戻)] の辞書。
            券種を混ぜる場合は 券種 -> その辞書 の入れ子でも受け付ける。
    """
    nested = bool(payout_index) and isinstance(
        next(iter(payout_index.values())), dict)
    bet = [p for p in plans if p.shape]
    if not nested and any(p.bet_type != '三連複' for p in bet):
        # 平らな辞書は三連複のものとして扱う（以前の形式）。他の券種を
        # 混ぜて渡すと全部外れ扱いになり、回収率が桁違いに低く出る
        raise ValueError('三連複以外の買い目があります。払戻は '
                         'betting.payout_index() で券種ごとに渡してください')
    cost = sum(p.cost for p in bet)
    returns, hits, payout = [], 0, 0
    for p in bet:
        table = payout_index.get(p.bet_type, {}) if nested else payout_index
        wins = table.get(p.race_id, [])
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
