"""買い目戦略のバックテスト

## 全レース買う前提をやめる

全レースを買うと毎回控除率を払うので、回収率は構造的に 100 - 控除率 に
収束する。モデルの上乗せが +4.3pt あっても +22.5pt には届かない。

**買わないレースは回収率100%** （資金が減らない）。したがって勝ち筋は

  1. 買うレースを選ぶ（参加率を下げる）
  2. 買う点数を減らす
  3. 残したレースで上乗せを稼ぐ

となる。評価も「全レースの回収率」ではなく
**「参加率」と「購入したレースの回収率」**の組で見る。

## 的中率と回収率は両方見ないと判断できない

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
    candidates: int = 0
    """評価対象になったレース数（買わなかったものも含む）。"""
    races: int = 0
    """実際に購入したレース数。"""
    points_per_race: float = 0.0
    cost: int = 0
    payout: int = 0
    hits: int = 0
    returns: List[float] = field(default_factory=list)
    """レースごとの収支（払戻 - 購入）。分散の計算に使う。"""
    hit_payouts: List[int] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """参加率。対象レースのうち何%を買ったか。"""
        return 100 * self.races / self.candidates if self.candidates else 0.0

    @property
    def hit_rate(self) -> float:
        """購入したレースのうち当たった割合。"""
        return 100 * self.hits / self.races if self.races else 0.0

    @property
    def profit(self) -> int:
        return self.payout - self.cost

    @property
    def overall_roi(self) -> float:
        """買わなかったレースも含めた資金ベースの回収率。

        買わなければ減らないので、参加率が低いほど100%に近づく。
        «全レースを1点ずつ買う» との比較には使えないが、
        資金がどれだけ目減りするかの実感に近い。
        """
        if not self.candidates:
            return 0.0
        avg_cost = self.points_per_race * UNIT
        notional = self.candidates * avg_cost
        skipped = (self.candidates - self.races) * avg_cost
        return 100 * (self.payout + skipped) / notional if notional else 0.0

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
    def roi_ci95(self) -> Optional[Tuple[float, float]]:
        """回収率の95%信頼区間。

        少数レースで出た「回収率100%超」は、ほぼ必ず区間が100%をまたぐ。
        参加率を絞るほどレース数が減り、区間は広がる。
        数字の大小ではなく区間で判断すること。
        """
        if len(self.returns) < 2 or not self.points_per_race:
            return None
        per_race_cost = self.points_per_race * UNIT
        arr = np.asarray(self.returns, dtype=float)
        se = float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
        mu = float(np.mean(arr))
        lo = 100 * (1 + (mu - 1.96 * se) / per_race_cost)
        hi = 100 * (1 + (mu + 1.96 * se) / per_race_cost)
        return (lo, hi)

    @property
    def significantly_profitable(self) -> bool:
        """95%信頼区間の下限が100%を超えているか。"""
        ci = self.roi_ci95
        return ci is not None and ci[0] > 100.0

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
            '参加率%': round(self.coverage, 1),
            '購入レース': self.races,
            '的中': self.hits,
            '的中率%': round(self.hit_rate, 2),
            '回収率%': round(self.roi, 1),
            '最高配当除く%': round(self.roi_excl_top, 1),
            '最高配当の寄与%': round(self.top_share, 1),
            '配当中央値': int(self.median_payout),
            '回収率95%区間': (f'{self.roi_ci95[0]:.0f}〜{self.roi_ci95[1]:.0f}'
                           if self.roi_ci95 else None),
            '黒字と言えるか': '◯' if self.significantly_profitable else '×',
            '収支': self.profit,
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
        min_runners: int = 8,
        bet_races: Optional[set] = None) -> Result:
    """上位 n_pick 頭の BOX を買ったときの成績。

    Args:
        test: 検証対象。race_id / horse_number / finish_position と並べ替え用の列
        payouts: payouts テーブル
        column: 並べ替えに使う列（AI確率、人気 など）
        n_pick: 何頭を買い目に含めるか
        bet_type: 券種。必要頭数と、着順が関係するか（点数の数え方）を決める
        bet_races: 購入する race_id の集合。None なら全レース購入。
            指定すると、それ以外は「買わない」＝資金が減らない扱いになる。
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

        res.candidates += 1
        if bet_races is not None and rid not in bet_races:
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


def race_signals(test: pd.DataFrame, prob_col: str = 'pred_top3',
                 takeout: float = 0.2) -> pd.DataFrame:
    """「このレースを買うべきか」の判断材料をレース単位で作る。

    買わないレースは回収率100%（資金が減らない）。したがって全レースを
    買うのではなく、優位がありそうなレースだけを選ぶ必要がある。

    Returns:
        race_id を index に、次の列を持つ DataFrame

        top1 / top2_sum / top3_sum
            AIが上位に置いた馬の確率。モデルの確信度
        mkt2 / mkt3
            同じ馬の市場暗示確率の合計（(1-控除率)/オッズ）
        edge2 / edge3
            AIの確率 − 市場の確率。**市場が見落としている度合い**
        runners / top1_odds / top1_pop
    """
    t = test.sort_values(prob_col, ascending=False)
    g = t.groupby('race_id')
    imp = t.assign(_imp=(1.0 - takeout) / t['odds']).groupby('race_id')['_imp']

    out = pd.DataFrame({
        # pandas 2系の nth(0) は行そのものを返しインデックスが揃わないため first を使う
        'top1': g[prob_col].first(),
        'top2_sum': g[prob_col].apply(lambda s: s.head(2).sum()),
        'top3_sum': g[prob_col].apply(lambda s: s.head(3).sum()),
        'mkt2': imp.apply(lambda s: s.head(2).sum()),
        'mkt3': imp.apply(lambda s: s.head(3).sum()),
        'runners': g.size(),
        'top1_odds': g['odds'].first(),
        'top1_pop': g['popularity'].first(),
    })
    out['edge2'] = out['top2_sum'] - out['mkt2']
    out['edge3'] = out['top3_sum'] - out['mkt3']
    return out


def coverage_curve(test: pd.DataFrame, payouts: pd.DataFrame, signals: pd.DataFrame,
                   signal: str, *, n_pick: int = 3, bet_type: str = 'ワイド',
                   prob_col: str = 'pred_top3',
                   quantiles: Sequence[float] = (0.0, 0.5, 0.75, 0.9, 0.95, 0.99)
                   ) -> pd.DataFrame:
    """シグナルで上位から絞ったときに、回収率がどう動くかを並べる。

    参加率を下げるほどレース数が減り、信頼区間が広がる。
    「回収率が上がった」だけでなく「区間が100%を超えたか」で判断すること。
    """
    rows = []
    for q in quantiles:
        th = signals[signal].quantile(q)
        ids = set(signals.index[signals[signal] >= th])
        label = '全レース' if q == 0.0 else f'上位{round((1 - q) * 100)}%'
        r = run(test, payouts, label=f'{signal} {label}', column=prob_col,
                n_pick=n_pick, bet_type=bet_type, bet_races=ids)
        rows.append(r.as_row())
    return pd.DataFrame(rows)


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
  回収率95%区間  回収率の信頼区間。少数レースで出た100%超はほぼ必ず
                 100%をまたぐ。数字の大小ではなく区間で判断する
  黒字と言えるか 区間の下限が100%を超えているか
  要レース数     回収率が100%超だと95%の確からしさで言うのに必要なレース数。
                 的中率が低いほど跳ね上がる。現実的に回せる回数を超えるなら、
                 期待値がプラスでも実用にならない
"""
