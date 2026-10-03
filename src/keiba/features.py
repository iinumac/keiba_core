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

import re

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

    speed_features: bool = True
    """走破タイム・上がり3F の集計を作るか。

    レース内で正規化するため集計に数十秒かかる。不要なら False。
    """

    recency_features: bool = True
    """前走・前々走の「人気と着順のズレ」を作るか。

    前走の人気を使うため市場情報を部分的に含む。
    """

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


def add_speed_features(df: pd.DataFrame) -> pd.DataFrame:
    """走破タイムと上がり3Fから、馬の地力を測る特徴量を作る。

    走破タイムは距離・馬場・馬場状態・ペースで水準がまるで違うため
    （ダ1000m 60.2秒 / ダ2000m 127.4秒）、そのままでは使えない。

    そこで**レース内で正規化**する。同じレースを走った馬との比較なので、
    条件の差が自動的に消える。同一レースの他馬の情報を使うが、
    レース結果は確定後に既知なのでリークではない。

        time_z   = -(タイム - レース平均) / レース標準偏差   速いほど大きい
        last3f_z = -(上がり - レース平均) / レース標準偏差

    そのうえで馬ごとに過去の実績を集計する。shift() で当該レースを含めない。

        r3_time_z     直近3走の平均。**実測で最も効く特徴量**
        avg_time_z    過去平均
        best_time_z   過去最高
        best_last_3f  過去最速の上がり（絶対値）

    これらを足すと市場フリーモデルの AUC は 0.7417 → 0.7527 に上がる。
    ただし市場込みモデルには効かない（0.8212 → 0.8209）。
    **市場は既にタイムと上がりを織り込んでいる。** docs/MARKET.md 参照。
    """
    df = df.copy()
    t = pd.to_numeric(df.get('time_seconds'), errors='coerce')
    l = pd.to_numeric(df.get('last_3f'), errors='coerce')
    df['_t_raw'], df['_l_raw'] = t, l

    for src, dst in (('_t_raw', 'time_z'), ('_l_raw', 'last3f_z')):
        g = df.groupby('race_id')[src]
        sd = g.transform('std').replace(0, np.nan)
        df[dst] = -(df[src] - g.transform('mean')) / sd

    df = df.sort_values(by=['horse_id', 'race_date'])
    g = df.groupby('horse_id')
    for col, base in (('time_z', 'time'), ('last3f_z', 'l3f')):
        s = g[col]
        df[f'best_{base}_z'] = s.transform(lambda x: x.shift().expanding().max())
        df[f'avg_{base}_z'] = s.transform(lambda x: x.shift().expanding().mean())
        df[f'r3_{base}_z'] = s.transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    df['best_last_3f'] = g['_l_raw'].transform(lambda x: x.shift().expanding().min())

    return df.drop(columns=['_t_raw', '_l_raw'])


MARGIN_WORDS = {'ハナ': 0.05, 'アタマ': 0.1, 'クビ': 0.2, '同着': 0.0, '大': 10.0}
_MARGIN_FRAC = re.compile(r'(\d+)?\.?(\d+)/(\d+)')


def margin_to_lengths(value) -> float:
    """着差の表記を馬身に直す。

    netkeiba は「クビ」「1/2」「1.1/4」「大」のように書く。
    "1.1/4" は 1 と 1/4 で 1.25 馬身。
    """
    s = str(value).strip()
    if not s or s == 'nan':
        return np.nan
    if s in MARGIN_WORDS:
        return MARGIN_WORDS[s]
    m = _MARGIN_FRAC.fullmatch(s.replace(' ', ''))
    if m:
        return float(m.group(1) or 0) + float(m.group(2)) / float(m.group(3))
    try:
        return float(s)
    except ValueError:
        return np.nan


def add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """前走・前々走の「人気と着順のズレ」から、市場の過剰反応を捉える。

    市場は直近の目立った結果に引きずられる。1番人気で大敗した馬も、
    人気薄で激走した馬も、次走では**過大評価される**。
    逆に地味に凡走した馬は過小評価される。

    人気より着順が悪ければ正、良ければ負:

        gap = 前走着順 - 前走人気

    実データで、同じ人気帯の平均回収率との差（超過pt）。いずれも有意:

        前走13番人気以下 × 2-3着（人気薄で激走）   -6.6pt  過大評価
        前走1番人気 × 9-13着（1番人気で大敗）      -5.9pt  過大評価
        前走4-6番人気 × 9-13着（地味に凡走）       +3.9pt  過小評価

    同じ「目立った実績への過剰反応」は他の軸でも出る:

        2走連続勝利                              -8.5pt  過大評価
        前走5馬身以上の大敗                       -10.5pt  過大評価
        1年以上の休み明け                        -20.3pt  過大評価
        前走クビ・ハナ差負け                       +2.0pt  過小評価
        6-8週の間隔                              +3.8pt  過小評価
        3走とも着外                               +1.1pt  過小評価

    **良くも悪くも目立った馬は過大評価され、地味な馬は過小評価される。**

    これらは前走の人気を使うので、市場情報を部分的に含む。
    `MARKET_FREE_FEATURES` には入れない。
    """
    df = df.sort_values(by=['horse_id', 'race_date']).copy()
    pop = pd.to_numeric(df['popularity'], errors='coerce')
    fin = pd.to_numeric(df['finish_position'], errors='coerce')
    df['_pop'], df['_fin'] = pop, fin
    g = df.groupby('horse_id')

    for k in (1, 2):
        df[f'p{k}_pop'] = g['_pop'].shift(k)
        df[f'p{k}_fin'] = g['_fin'].shift(k)
        df[f'p{k}_gap'] = df[f'p{k}_fin'] - df[f'p{k}_pop']
        # 良くも悪くも「目立った」度合い。市場の記憶に残りやすさ
        df[f'p{k}_surprise'] = df[f'p{k}_gap'].abs()

    df['gap_mean2'] = df[['p1_gap', 'p2_gap']].mean(axis=1)

    # --- 前走の着差。僅差負けは過小評価、大差負けは過大評価される ---
    #   クビ・ハナ差負け  +2.0pt
    #   5馬身以上負け    -10.5pt
    if 'margin' in df.columns:
        uniq = pd.Series(df['margin'].astype(str).unique())
        conv = dict(zip(uniq, uniq.map(margin_to_lengths)))
        df['margin_len'] = df['margin'].astype(str).map(conv)
        df['p1_margin'] = g['margin_len'].shift(1)
    else:
        df['margin_len'] = np.nan
        df['p1_margin'] = np.nan

    # --- 連続好走・連続凡走。連勝中の馬は買われすぎる ---
    #   2走連続勝利   -8.5pt
    #   3走とも着外   +1.1pt
    df['p3_fin'] = g['_fin'].shift(3)
    t = [df[c] <= 3 for c in ('p1_fin', 'p2_fin', 'p3_fin')]
    df['streak_top3'] = (t[0].fillna(False).astype(int)
                         + (t[0] & t[1]).fillna(False).astype(int)
                         + (t[0] & t[1] & t[2]).fillna(False).astype(int))
    df['streak_win'] = ((df['p1_fin'] == 1).fillna(False).astype(int)
                        + ((df['p1_fin'] == 1) & (df['p2_fin'] == 1)).fillna(False).astype(int))

    # --- 騎手の格。乗り替わりで上がったか下がったか ---
    # 過去の騎乗数を格の代理とする（cumcount なので当該レースを含めない）
    if 'jockey_id' in df.columns:
        df = df.sort_values(by=['jockey_id', 'race_date'])
        df['jockey_rides'] = df.groupby('jockey_id').cumcount()
        df = df.sort_values(by=['horse_id', 'race_date'])
        df['jockey_rides_delta'] = (df['jockey_rides']
                                    - df.groupby('horse_id')['jockey_rides'].shift(1))
    else:
        df['jockey_rides'] = np.nan
        df['jockey_rides_delta'] = np.nan

    return df.drop(columns=['_pop', '_fin'])


def add_added_value(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の押し上げ力。

    「そのレースの結果」から「馬の期待値」を引いた差を、その騎手／調教師の
    過去（前走まで）の累積平均としたもの。shift() でリークを防ぐ。

    副作用として、マスタ用に最新行を取り出したとき直近1走が反映されない。
    この扱いは docs/FEATURE_BACKLOG.md §2 の論点。
    """
    df['added_value_in_race'] = df['is_top3'] - df['horse_expected_top3_rate']

    # まだ走っていない行（出馬表から足したもの）は結果が無いので、
    # is_top3 が 0 のまま騎手・調教師の平均に混ざってしまう。同じ騎手が
    # 同じ週末に何鞍も乗ると、先の行が後の行の平均を押し下げる。除外する。
    if 'is_unrun' in df.columns:
        df.loc[df['is_unrun'].fillna(False).astype(bool), 'added_value_in_race'] = np.nan

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
    if config.speed_features:
        df = add_speed_features(df)
    if config.recency_features:
        df = add_recency_features(df)
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
