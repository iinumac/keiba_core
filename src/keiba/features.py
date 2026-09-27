"""
データセット単位の特徴量生成パイプライン

C03（モデル学習）と C04（三連複の期待値）が各ノートブック内に個別に持っていた
特徴量計算を1箇所に集約したもの。

両者の定義には差異があったため、いきなり片方に寄せるのではなく
`FeatureConfig` で差異を明示し、`C03_CONFIG` / `C04_CONFIG` で
それぞれの従来挙動を完全に再現できるようにしている。
差異の一覧と、どちらに寄せるべきかの検討は docs/FEATURE_BACKLOG.md を参照。

行単位で1頭ずつ計算する `calculator.py` とは別物で、こちらは
DataFrame 全体をベクトル演算で処理する。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """C03 と C04 で食い違っていた箇所を明示的に持つ設定。

    ここに並んでいる項目はすべて「かつて両ノートブックで値が違っていた」もの。
    新しく特徴量を足すときは、差異が生まれないよう既定値を1つに保つこと。
    """

    # --- クレンジング ---
    exclude_jockey_names: Tuple[str, ...] = ()
    """学習から外す騎手名。C03 のみ設定されていた。"""

    drop_unfinished: bool = False
    """着順が数値でない行（競走中止など）を落とすか。C04 のみ True 相当だった。"""

    require_positive_odds: bool = False
    """オッズが正の行だけ残すか。C04 のみ True 相当だった。"""

    # --- 新馬（過去成績なし）の埋め値 ---
    debut_top3_fill: Optional[float] = None
    """新馬の3着内率。None なら全体平均 × `debut_discount` を使う（C03の挙動）。"""

    debut_discount: float = 0.5
    debut_win_fill: float = 0.08

    # --- 前走情報の埋め値。None は欠損のまま（LightGBM に欠損として扱わせる）---
    prev_finish_fill: Optional[float] = None
    prev_last_3f_fill: Optional[float] = None
    prev_odds_fill: Optional[float] = None
    prev_popularity_fill: Optional[float] = None
    days_since_last_fill: Optional[float] = None

    # --- コース ---
    surface_fill: float = -1.0
    """surface が芝でもダートでもない行の値。C03 は -1、C04 は 0（＝芝と同値）だった。"""

    debut_from_prev_finish: bool = False
    """新馬判定を prev_finish の欠損で行うか（C03の従来挙動）。

    True にすると、前走が競走中止などで着順が数値でない馬も「新馬」と判定される。
    実データで 3,678 行が該当し、C03 はこれらを学習から除外してしまっている。
    False（既定）は出走回数（cumcount）で判定するため、この取りこぼしが起きない。
    docs/FEATURE_BACKLOG.md §3 参照。
    """

    surface_map: dict = field(default_factory=lambda: {'芝': 0, 'ダート': 1})

    # --- 数値列 ---
    numeric_fills: dict = field(default_factory=dict)
    """impost / distance / horse_weight などの埋め値。"""

    # --- 市場（オッズ）系特徴量 ---
    market_features: bool = False
    takeout_rate: float = 0.2
    """控除率。market_implied_win_prob = (1 - takeout_rate) / odds"""


C03_CONFIG = FeatureConfig(
    exclude_jockey_names=('藤岡康太',),
    drop_unfinished=False,
    require_positive_odds=False,
    debut_top3_fill=None,          # 全体平均 × 0.5
    surface_fill=-1.0,
    debut_from_prev_finish=True,   # 既存挙動の再現。既知の不具合あり
    numeric_fills={'impost': 55.0},
    market_features=False,
)

C04_CONFIG = FeatureConfig(
    exclude_jockey_names=(),
    drop_unfinished=True,
    require_positive_odds=True,
    debut_top3_fill=0.24,
    debut_win_fill=0.08,
    prev_finish_fill=10.0,
    prev_last_3f_fill=36.5,
    prev_odds_fill=30.0,
    prev_popularity_fill=10.0,
    days_since_last_fill=90.0,
    surface_fill=0.0,
    numeric_fills={'impost': 0.0, 'distance': 0.0, 'horse_weight': 0.0, 'horse_number': 0.0},
    market_features=True,
    takeout_rate=0.2,
)


def clean(races_df: pd.DataFrame, results_df: pd.DataFrame,
          config: FeatureConfig) -> pd.DataFrame:
    """障害レース・除外騎手などを落とし、目的変数を付けた DataFrame を返す。

    `results_df` は surface / distance / race_date / level_score を既に持っているため、
    `races_df` は障害レースの特定にのみ使う。
    """
    obstacle_ids = races_df.loc[
        races_df['race_name'].str.contains('障害', na=False), 'race_id'
    ].unique()

    df = results_df[~results_df['race_id'].isin(obstacle_ids)].copy()
    df = df[~df['surface'].astype(str).str.contains('障害', na=False)]

    for name in config.exclude_jockey_names:
        df = df[df['jockey_name'] != name]

    if config.drop_unfinished:
        df = df[pd.to_numeric(df['finish_position'], errors='coerce').notna()]
        df['finish_position'] = df['finish_position'].astype(int)

    if config.require_positive_odds:
        odds = pd.to_numeric(df['odds'], errors='coerce')
        df = df[odds.notna() & (odds > 0)]

    df['race_date'] = pd.to_datetime(df['race_date'])
    df['year'] = df['race_date'].dt.year
    df['is_win'] = (df['finish_position'] == 1).astype(int)
    df['is_top3'] = (df['finish_position'] <= 3).astype(int)
    return df


def add_horse_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """馬の過去実績と前走情報。すべて shift() 済みでリークしない。"""
    df = df.sort_values(by=['horse_id', 'race_date'])

    if config.debut_top3_fill is None:
        top3_fill = df['is_top3'].mean() * config.debut_discount
    else:
        top3_fill = config.debut_top3_fill

    grp = df.groupby('horse_id')
    df['horse_expected_top3_rate'] = grp['is_top3'].transform(
        lambda x: x.shift().expanding().mean()).fillna(top3_fill)
    # C04 での呼び名。中身は同じ
    df['horse_prev_top3_rate'] = df['horse_expected_top3_rate']
    df['horse_prev_win_rate'] = grp['is_win'].transform(
        lambda x: x.shift().expanding().mean()).fillna(config.debut_win_fill)

    shifts = {
        'prev_finish': ('finish_position', config.prev_finish_fill),
        'prev_last_3f': ('last_3f', config.prev_last_3f_fill),
        'prev_odds': ('odds', config.prev_odds_fill),
        'prev_popularity': ('popularity', config.prev_popularity_fill),
    }
    for out, (src, fill) in shifts.items():
        s = df.groupby('horse_id')[src].shift(1)
        df[out] = s if fill is None else s.fillna(fill)

    df['prev_race_date'] = df.groupby('horse_id')['race_date'].shift(1)
    days = (df['race_date'] - df['prev_race_date']).dt.days
    df['days_since_last'] = (days if config.days_since_last_fill is None
                             else days.fillna(config.days_since_last_fill))

    df['prev_jockey_id'] = df.groupby('horse_id')['jockey_id'].shift(1)
    df['is_jockey_changed'] = (df['jockey_id'] != df['prev_jockey_id']).astype(int)
    df.loc[df['prev_jockey_id'].isna(), 'is_jockey_changed'] = 0

    if config.debut_from_prev_finish:
        df['is_debut'] = df['prev_finish'].isna().astype(int)
    else:
        df['is_debut'] = df.groupby('horse_id').cumcount().eq(0).astype(int)
    return df


def add_added_value(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の押し上げ力。

    「そのレースの結果」から「馬の期待値」を引いた差を、その騎手／調教師の
    過去（前走まで）の累積平均としたもの。shift() でリークを防ぐ。

    副作用として、マスタ用に最新行を取り出したとき直近1走が反映されない。
    この扱いは docs/FEATURE_BACKLOG.md §2 の論点。
    """
    df['added_value_in_race'] = df['is_top3'] - df['horse_expected_top3_rate']

    df = df.sort_values(by=['jockey_id', 'race_date'])
    df['jockey_added_value'] = df.groupby('jockey_id')['added_value_in_race'].transform(
        lambda x: x.shift().expanding().mean()).fillna(0.0)

    df = df.sort_values(by=['trainer_id', 'race_date'])
    df['trainer_added_value'] = df.groupby('trainer_id')['added_value_in_race'].transform(
        lambda x: x.shift().expanding().mean()).fillna(0.0)
    return df


def add_course_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    df['surface_encoded'] = df['surface'].map(config.surface_map).fillna(config.surface_fill)
    # C04 での呼び名。中身は同じ
    df['surface_code'] = df['surface_encoded']

    for col, fill in config.numeric_fills.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill)
        else:
            df[col] = fill
    return df


def add_market_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """オッズが示す市場の歪み。C04 の期待値戦略の中核。"""
    odds = pd.to_numeric(df['odds'], errors='coerce')
    pop = pd.to_numeric(df['popularity'], errors='coerce')

    df['market_implied_win_prob'] = (1.0 - config.takeout_rate) / odds
    df['min_odds_in_race'] = df.groupby('race_id')['odds'].transform('min')
    df['odds_ratio_to_fav'] = odds / (df['min_odds_in_race'] + 1e-5)
    df['pop_odds_mismatch'] = pop * 2.5 - np.log(odds + 1)
    return df


def build_features(races_df: pd.DataFrame, results_df: pd.DataFrame,
                   config: FeatureConfig = C03_CONFIG) -> pd.DataFrame:
    """特徴量付きの学習・予測用 DataFrame を作る。

    Args:
        races_df: races.parquet
        results_df: results.parquet
        config: C03_CONFIG / C04_CONFIG、または独自設定

    Returns:
        特徴量を追加した DataFrame
    """
    df = clean(races_df, results_df, config)
    df = add_horse_features(df, config)
    df = add_added_value(df)
    df = add_course_features(df, config)
    if config.market_features:
        df = add_market_features(df, config)

    # 行順を馬・日付順に固定して返す。
    # マスタ生成は sort_values('race_date').groupby(...).tail(1) で最新行を取るが、
    # 同一日に複数レースがある場合（騎手の 356/574 が該当）どの行が選ばれるかは
    # 呼び出し時点の行順に依存する。順序を固定しないと結果が変わってしまう。
    # 本質的な対処は docs/FEATURE_BACKLOG.md §2 を参照。
    return df.sort_values(by=['horse_id', 'race_date'])
