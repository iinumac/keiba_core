"""買い目戦略のバックテスト

**的中率と回収率は両方見ないと判断できない。**

- 的中率だけ見ると、点数を増やせば必ず上がるので「良い戦略」に見えてしまう
- 回収率だけ見ると、ごく稀な超高配当1本に支配される。実データの三連単には
  5,800万円の払戻がある。1本当たるかどうかで数字が別物になる

さらに「プラスに持っていくまでに何回必要か」も効く。期待値がプラスでも
的中率が低すぎると、収束する前に資金が尽きる。そこで分散と必要試行回数も出す。

払戻は `payouts` テーブル（レース×券種×組み合わせの縦持ち）から取る。
**同着のときは複数行が当たりになるので合計する**（どれか1つではない）。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

UNIT = 100
"""1点あたりの購入額（円）。払戻も100円あたりで記録されている。"""

# 券種ごとの「何頭を選ぶか」と「着順が関係するか」。
# 着順が関係する券種（三連単・馬単）を組み合わせ数で数えると点数を過少に
# 見積もり、回収率が跳ね上がる。5頭BOXの三連単は C(5,3)=10 ではなく
# P(5,3)=60 点。実際にこれで三連単の回収率が439%と出てしまった。
BET_SPEC = {
    '単勝':   (1, False),
    '複勝':   (1, False),
    '枠連':   (2, False),
    '馬連':   (2, False),
    'ワイド': (2, False),
    '馬単':   (2, True),
    '三連複': (3, False),
    '三連単': (3, True),
}


def points_for(bet_type: str, n_pick: int) -> int:
    """n_pick 頭のBOXを買ったときの点数。"""
    size, ordered = BET_SPEC[bet_type]
    return math.perm(n_pick, size) if ordered else math.comb(n_pick, size)


@dataclass
class Result:
    label: str
    bet_type: str
    races: int = 0
    points_per_race: float = 0.0
    cost: int = 0
    payout: int = 0
    hits: int = 0
    returns: List[float] = field(default_factory=list)
    """レースごとの収支（払戻 - 購入）。分散の計算に使う。"""
    hit_payouts: List[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return 100 * self.hits / self.races if self.races else 0.0

    @property
    def roi(self) -> float:
        return 100 * self.payout / self.cost if self.cost else 0.0

    @property
    def roi_excl_top(self) -> float:
        """最高配当の1本を除いた回収率。外れ値への依存度を見る。"""
        if not self.hit_payouts or not self.cost:
            return 0.0
        return 100 * (self.payout - max(self.hit_payouts)) / self.cost

    @property
    def top_share(self) -> float:
        """払戻全体に占める最高配当1本の割合。"""
        if not self.hit_payouts or not self.payout:
            return 0.0
        return 100 * max(self.hit_payouts) / self.payout

    @property
    def median_payout(self) -> float:
        return float(np.median(self.hit_payouts)) if self.hit_payouts else 0.0

    @property
    def sharpe(self) -> float:
        """1レースあたり収支の 平均 / 標準偏差。ブレに対する稼ぎの効率。"""
        if len(self.returns) < 2:
            return 0.0
        sd = float(np.std(self.returns, ddof=1))
        return float(np.mean(self.returns)) / sd if sd else 0.0

    @property
    def races_to_confirm(self) -> Optional[int]:
        """回収率が100%を超えていると95%の確からしさで言うのに必要なレース数。

        n = (1.96 * σ / μ)^2 。μ が0以下なら意味がないので None。
        的中率が低い戦略ほど σ が大きくなり、必要数が跳ね上がる。
        """
        if len(self.returns) < 2:
            return None
        mu = float(np.mean(self.returns))
        if mu <= 0:
            return None
        sd = float(np.std(self.returns, ddof=1))
        return int(math.ceil((1.96 * sd / mu) ** 2))

    def as_row(self) -> Dict:
        return {
            '戦略': self.label,
            '点数': round(self.points_per_race, 1),
            'レース': self.races,
            '的中': self.hits,
            '的中率%': round(self.hit_rate, 2),
            '回収率%': round(self.roi, 1),
            '最高配当除く%': round(self.roi_excl_top, 1),
            '最高配当の寄与%': round(self.top_share, 1),
            '配当中央値': int(self.median_payout),
            '収支/σ': round(self.sharpe, 4),
            '要レース数': self.races_to_confirm,
        }


def payout_index(payouts: pd.DataFrame, bet_type: str
                 ) -> Dict[str, List[Tuple[FrozenSet[int], int]]]:
    """race_id -> [(当たりの組み合わせ, 払戻), ...] を作る。

    同着があると1レースに複数の当たりが入る。
    """
    sub = payouts[payouts['bet_type'] == bet_type]
    idx: Dict[str, List[Tuple[FrozenSet[int], int]]] = {}
    for rid, combo, pay in zip(sub['race_id'], sub['horse_numbers'], sub['payout']):
        idx.setdefault(str(rid), []).append((frozenset(int(x) for x in combo), int(pay)))
    return idx


def run(test: pd.DataFrame, payouts: pd.DataFrame, *,
        label: str, column: str, n_pick: int,
        bet_type: str = '三連複', ascending: bool = False,
        min_runners: int = 8) -> Result:
    """上位 n_pick 頭の BOX を買ったときの成績。

    Args:
        test: 検証対象。race_id / horse_number / finish_position と並べ替え用の列
        payouts: payouts テーブル
        column: 並べ替えに使う列（AI確率、人気 など）
        n_pick: 何頭を買い目に含めるか
        bet_type: 券種。必要頭数と、着順が関係するか（点数の数え方）を決める
    """
    if bet_type not in BET_SPEC:
        raise ValueError(f'未対応の券種: {bet_type}')
    size, _ = BET_SPEC[bet_type]
    if n_pick < size:
        raise ValueError(f'{bet_type} は {size} 頭必要（n_pick={n_pick}）')
    idx = payout_index(payouts, bet_type)
    points = points_for(bet_type, n_pick)

    res = Result(label=label, bet_type=bet_type, points_per_race=points)
    for rid, g in test.groupby('race_id'):
        if len(g) < min_runners:
            continue
        rid = str(rid)
        wins = idx.get(rid)
        if not wins:          # その券種が発売されていないレースは対象外
            continue

        picks = set(g.sort_values(column, ascending=ascending)
                     .head(n_pick)['horse_number'].astype(int))
        cost = points * UNIT
        # 同着は複数行が当たる。合計する
        got = sum(pay for combo, pay in wins if combo <= picks)

        res.races += 1
        res.cost += cost
        res.payout += got
        res.returns.append(got - cost)
        if got:
            res.hits += 1
            res.hit_payouts.append(got)
    return res


def compare(results: Sequence[Result]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in results])


def explain() -> str:
    return """各指標の読み方

  的中率%        当たったレースの割合。点数を増やせば必ず上がるので単独では見ない
  回収率%        払戻 ÷ 購入。100を超えればプラス
  最高配当除く%  最も高い配当1本を除いた回収率。これが大きく下がるなら、
                 その戦略は「稀な大当たり頼み」で再現性が低い
  最高配当の寄与% 払戻全体に占める最高配当1本の割合
  配当中央値     当たったときの配当の中央値。平均は外れ値に引っ張られる
  収支/σ         1レースあたり収支の 平均÷標準偏差。ブレに対する稼ぎの効率
  要レース数     回収率が100%超だと95%の確からしさで言うのに必要なレース数。
                 的中率が低いほど跳ね上がる。現実的に回せる回数を超えるなら、
                 期待値がプラスでも実用にならない
"""
