"""ピックアップフラグと凡走フラグ、補正スコア。

目線の表（偏差値）に載らない「該当する／しない」の要素を数える。
境目と補正の重みは scripts/flags_fit.py で2024年まで（補正は2022〜2024年）のデータから決め、
2025年以降で確かめた値を固定している。

## 凡走フラグ（1〜3番人気の6着以下が増える要素。境目は1〜3番人気で決めた）

候補は10あった。市場の見込みとほかの候補を入れたうえで、指数(近3走)がトップから離れている
（指数(最高)と重なる）・昇級・前走 人気より着順が悪い は重みがほぼ残らなかったので外した
（2026-10-08）。残りの7つは重みがそれぞれ違うので、重み（FLOP_WEIGHTS）の合計を使う。
重みは1〜5番人気が6着以下になるかで決めた。

    1〜5番人気（2025年以降）  凡走点   0〜0.5  0.5〜1.5  1.5〜2.5  2.5〜3  3〜3.5  3.5〜
                              6着以下   31%      35%       42%      47%     49%    50%

## ピックアップフラグ（人気4番以下の3着内が増える要素。境目は人気4番以下で決めた）

候補は8つあったが、市場の見込みとほかの候補を入れたうえで重みが残ったのは指数の2つだけ
（2026-10-07）。上がり指数・前走の指数・前走の上がり・使い詰め・前走の勝ち方・人気より
着順が良い、は単独では効いて見えても、市場か指数のフラグで説明できていた。
2つは同じ重さではないので、数ではなく重み（PICK_WEIGHTS）の合計を使う。

    人気4番以下（2025年以降）   3着内率 / 市場の見込み
    どちらも無し                 9.4% / 10.1%
    指数(近3走)だけ             18.0% / 17.2%
    指数(最高)だけ              16.8% / 15.7%
    両方                        23.7% / 21.7%   ← ピックアップリストに載せる

## 補正スコア

市場の見込み（market.market_top3）を、ピックアップの重みと凡走の重みで補正した3着内確率。
フラグの多くは市場もオッズに織り込んでいるので、補正は小さい
（2025年以降で 対数損失 0.3996→0.3990、AUC 0.8198→0.8205）。

## 指数は距離を変換したもの

指数（近3走・最高・上がり）は、同じ芝ダの過去走を今回の距離に変換してから集計した値
（figure.converted_summary）。境目・重みもその値で決め直した（2026-10-10）。
フラグの役割は、数字の理由を人に説明するリストとして見せること。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

T = {
    'flop_fig_best_rank': 7.0, 'flop_l3fig_rank': 7.0, 'flop_fig_gap': -0.279, 'flop_style': 0.278,
    'flop_p1_l3f': -0.25, 'flop_rest': 76.0, 'flop_p1_gap': 2.0, 'flop_place_vs_win': -0.12,
    'pick_fig_best_rank': 5.0, 'pick_fig_gap': -0.217, 'pick_l3fig_rank': 5.0, 'pick_fig_p1': 0.464,
    'pick_p1_l3f': 0.3125, 'pick_rest': 21.0, 'pick_p1_gap': -3.0, 'pick_spread5': 1.0,
}
"""境目。指数は1000mあたり秒、順位はメンバー内、休養は日数。"""

PICK_WEIGHTS = {'指数(最高)がメンバー内で上位': 0.1113, '指数(近3走)がトップに近い': 0.0576}
"""ピックアップの重み（logit）。補正スコアの式の係数そのもの。"""

FLOP_WEIGHTS = {'上がり指数(近3走)がメンバー内で下位': 0.1764, '芝ダ替わり': 0.1584, '休み明け': 0.1340,
                '先行型': 0.1278, '複勝の支持が単勝の見込みより低い': 0.1117,
                '前走 位置の割に上がりが遅い': 0.1107, '指数(最高)がメンバー内で下位': 0.0556}
"""凡走の重み（1〜5番人気が6着以下になる logit への上乗せ）。"""

CALIBRATION = (-0.0301, 0.9703, -0.2846)
"""補正スコア = logistic(切片 + a·logit(市場の見込み) + ピックアップの重みの合計 + c·凡走の重みの合計)。"""

COLUMNS = ['fig_best_rank', 'l3fig_r3_rank', 'fig_r3_gap', 'style', 'p1_l3f_vs_pos',
           'days_since_last', 'surface_change', 'place_vs_win']
"""判定に使う列。どれも欠けていてよい（欠けていればフラグは立たない）。"""


def _c(t: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(t[name], errors='coerce') if name in t else pd.Series(np.nan, index=t.index)


def flop_flags(t: pd.DataFrame) -> pd.DataFrame:
    """凡走フラグ（True/False の表）。列は FLOP_WEIGHTS の順。"""
    f = {
        '上がり指数(近3走)がメンバー内で下位': _c(t, 'l3fig_r3_rank') >= T['flop_l3fig_rank'],
        '芝ダ替わり': _c(t, 'surface_change') == 1,
        '休み明け': _c(t, 'days_since_last') >= T['flop_rest'],
        '先行型': _c(t, 'style') <= T['flop_style'],
        '複勝の支持が単勝の見込みより低い': _c(t, 'place_vs_win') <= T['flop_place_vs_win'],
        '前走 位置の割に上がりが遅い': _c(t, 'p1_l3f_vs_pos') <= T['flop_p1_l3f'],
        '指数(最高)がメンバー内で下位': _c(t, 'fig_best_rank') >= T['flop_fig_best_rank'],
    }
    return pd.DataFrame({k: v.fillna(False).astype(bool) for k, v in f.items()}, index=t.index)


def flop_score(flags_table: pd.DataFrame) -> pd.Series:
    """凡走の重みの合計（logit）。"""
    return flags_table.astype(float) @ pd.Series(FLOP_WEIGHTS)[flags_table.columns]


def flop_points(score) -> np.ndarray:
    """表示用の凡走点。いちばん重いフラグを1点とする。"""
    return np.round(np.asarray(score, float) / max(FLOP_WEIGHTS.values()), 1)


CAUTION_POP, CAUTION_POINTS = 5, 2.5
"""要注意リストに載せる人気の範囲と凡走点（1〜5番人気で凡走点2.5以上なら6着以下が46〜53%）。"""


def pickup_flags(t: pd.DataFrame) -> pd.DataFrame:
    """ピックアップフラグ（True/False の表）。列は PICK_WEIGHTS の順。"""
    f = {
        '指数(最高)がメンバー内で上位': _c(t, 'fig_best_rank') <= T['pick_fig_best_rank'],
        '指数(近3走)がトップに近い': _c(t, 'fig_r3_gap') >= T['pick_fig_gap'],
    }
    return pd.DataFrame({k: v.fillna(False).astype(bool) for k, v in f.items()}, index=t.index)


def pickup_score(flags_table: pd.DataFrame) -> pd.Series:
    """ピックアップの重みの合計（logit）。"""
    return flags_table.astype(float) @ pd.Series(PICK_WEIGHTS)[flags_table.columns]


def pickup_points(score) -> np.ndarray:
    """表示用の点数。いちばん重いフラグを1点とする（指数(最高)1.0、指数(近3走)0.5）。"""
    return np.round(np.asarray(score, float) / max(PICK_WEIGHTS.values()), 1)


PICKUP_LIST_MIN = sum(PICK_WEIGHTS.values())
"""ピックアップリストに載せる重みの合計。両方のフラグが立った馬だけ。"""


def calibrated(market_top3, pick_score, flop_score_) -> np.ndarray:
    """補正スコア（3着内確率）。pick_score / flop_score_ は pickup_score / flop_score の値。"""
    p = np.clip(np.asarray(market_top3, float), 1e-4, 0.9999)
    z = (CALIBRATION[0] + CALIBRATION[1] * np.log(p / (1 - p))
         + np.asarray(pick_score, float) + CALIBRATION[2] * np.asarray(flop_score_, float))
    return 1 / (1 + np.exp(-z))


def describe(row: pd.Series, flags: pd.DataFrame) -> str:
    """その馬に立ったフラグの名前を並べる。"""
    return '／'.join(k for k, v in flags.loc[row.name].items() if v)
