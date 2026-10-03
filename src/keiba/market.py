"""市場（オッズ）が見込んでいる確率を復元する

モデルが市場を上回れているかを見るには、まず市場の見立てを同じ土俵
（3着内確率）に揃える必要がある。単勝オッズは1着の配当なので、
そのままでは複勝・ワイド・三連複の評価に使えない。

手順

1. 単勝オッズ → 控除率を除いた勝率に正規化
2. 勝率 → 3着内確率（Harville モデル）

Harville は「1着が決まったら、残りの馬で2着が決まる」という逐次選択。
独立性を仮定するため、人気薄の複勝確率を過小に、本命を過大に見積もる
癖が知られている。実データでもその通りに出た。

  市場の見込み 0.04 → 実測 0.052（人気薄は過小評価）
  市場の見込み 0.82 → 実測 0.737（本命は過大評価）

このズレを「市場の誤り」と読むのは早計で、Harville 変換自体の癖が
混ざっている。優位の有無は実際の払戻で確かめること（docs/MARKET.md）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_TAKEOUT = 0.2


def implied_win_prob(odds: np.ndarray, normalize: bool = True) -> np.ndarray:
    """単勝オッズから勝率を復元する。

    Args:
        normalize: True なら合計が1になるよう正規化する（控除率が消える）。
            False なら (1 - 控除率) / オッズ をそのまま返す。
    """
    o = np.asarray(odds, dtype=float)
    ok = np.isfinite(o) & (o > 0)
    raw = np.zeros_like(o)
    raw[ok] = 1.0 / o[ok]
    total = raw.sum()
    if normalize and total > 0:
        return raw / total
    return raw * (1.0 - DEFAULT_TAKEOUT)


def harville_top3(win_prob: np.ndarray) -> np.ndarray:
    """勝率から3着内確率を出す（Harville）。

    P(i∈top3) = P(i=1着) + P(i=2着) + P(i=3着)
    """
    p = np.asarray(win_prob, dtype=float)
    n = len(p)
    out = p.copy()
    for i in range(n):
        if p[i] <= 0:
            continue
        s = 0.0
        for j in range(n):
            if j == i or p[j] <= 0:
                continue
            d1 = 1.0 - p[j]
            if d1 <= 1e-12:
                continue
            s += p[j] * p[i] / d1                       # j→i
            for k in range(n):
                if k == i or k == j or p[k] <= 0:
                    continue
                d2 = 1.0 - p[j] - p[k]
                if d2 <= 1e-12:
                    continue
                s += p[j] * (p[k] / d1) * (p[i] / d2)   # j→k→i
        out[i] += s
    return np.clip(out, 0.0, 1.0)


SUPPORT_TOP3_COEF = (4.335173265529065,          # 切片
                     2.2779574662938504,         # log(支持率)
                     0.3833137008563749,         # log(支持率)^2
                     0.03992235560882695,        # log(支持率)^3
                     -0.6340271100142684,        # log(頭数)
                     -0.061655305542466475)      # log(支持率) * log(頭数)
"""`top3_from_support` の係数。2024年までの全レースで学習（fit_support_top3）。"""


def _support_terms(support: np.ndarray, n_runners) -> np.ndarray:
    ls = np.log(np.clip(np.asarray(support, float), 1e-4, 1.0))
    ln = np.log(np.broadcast_to(np.asarray(n_runners, float), ls.shape))
    return np.column_stack([ls, ls ** 2, ls ** 3, ln, ls * ln])


def top3_from_support(odds: np.ndarray, coef=SUPPORT_TOP3_COEF) -> np.ndarray:
    """単勝オッズ（1レース分）から、市場の3着内確率を出す。

    単勝オッズの逆数をレース内で正規化したもの（**支持率**）は、実際の勝率と
    ほぼ一致する。ところが勝率から3着内率を出す Harville の式は偏る
    （人気馬で最大10pt過大、中穴で2〜3pt過小）。

    そこで、実際に3着内に入ったかどうかを log(支持率) の多項式と頭数で
    ロジスティック回帰した。支持率の帯ごとの誤差は概ね1pt以内
    （2024年までで学習、2025年以降で検証）。docs/MARKET.md「5.」。
    """
    o = np.asarray(odds, float)
    sup = implied_win_prob(o)
    z = coef[0] + _support_terms(sup, len(o)) @ np.asarray(coef[1:])
    p = 1.0 / (1.0 + np.exp(-z))
    p[~(np.isfinite(o) & (o > 0))] = np.nan
    return p


def fit_support_top3(df: pd.DataFrame, odds_col: str = 'odds',
                     race_col: str = 'race_id', target: str = 'is_top3'):
    """`SUPPORT_TOP3_COEF` を学習し直す。(切片, 係数...) を返す。"""
    from sklearn.linear_model import LogisticRegression
    d = df[[race_col, odds_col, target]].copy()
    d[odds_col] = pd.to_numeric(d[odds_col], errors='coerce')
    d = d[d[odds_col] > 0]
    inv = 1.0 / d[odds_col]
    sup = inv / inv.groupby(d[race_col]).transform('sum')
    n = d.groupby(race_col)[race_col].transform('size')
    X = _support_terms(sup.to_numpy(), n.to_numpy())
    lr = LogisticRegression(C=10, max_iter=2000).fit(X, d[target])
    return (float(lr.intercept_[0]),) + tuple(float(c) for c in lr.coef_[0])


def add_market_probs(df: pd.DataFrame, odds_col: str = 'odds',
                     race_col: str = 'race_id') -> pd.DataFrame:
    """`p_mkt_win` と `p_mkt_top3` を付けて返す。

    レース単位で計算するため、出走馬がそろった DataFrame を渡すこと。
    """
    out = df.copy()
    win = np.full(len(out), np.nan)
    top3 = np.full(len(out), np.nan)
    for _, idx in out.groupby(race_col, sort=False).indices.items():
        o = pd.to_numeric(out[odds_col].iloc[idx], errors='coerce').to_numpy(float)
        if np.isfinite(o).sum() < 3:
            continue
        p = implied_win_prob(o)
        win[idx] = p
        top3[idx] = harville_top3(p)
    out['p_mkt_win'] = win
    out['p_mkt_top3'] = top3
    return out


def value(df: pd.DataFrame, ai_col: str = 'p_ai',
          mkt_col: str = 'p_mkt_top3') -> pd.DataFrame:
    """AIの見立てと市場の見立ての差。

    `value` が正なら「AIは市場より高く見ている」＝市場が見落としている候補。
    `ratio` は比。人気薄では分母が小さく極端な値になるので、
    絞り込みに使うと穴ばかり拾うことになる（実測で回収率44.7%まで落ちた）。
    """
    out = df.copy()
    out['value'] = out[ai_col] - out[mkt_col]
    out['ratio'] = out[ai_col] / out[mkt_col].clip(lower=1e-4)
    return out
