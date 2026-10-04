"""レースレベル（出走メンバーの強さ）

同じ馬が別々のレースを走っていれば、その馬を物差しにしてレース同士を比べられる。
宝塚記念で勝ち馬から0.5秒差、天皇賞(春)で0.2秒差だった馬がいれば、
（その馬の調子が同じなら）宝塚記念の勝ち馬の方が0.3秒ぶん強いレースをしたことになる。

これを全レース・全馬について同時に解く。

    勝ち馬との差[馬, レース] = レベル[レース] − 能力[馬] + 誤差

- レベル    そのレースの勝ち馬がどれだけ強いレースをしたか。大きいほど強いレース
- 能力      その馬の地力。全レースを通して1つの値（期間内は一定とみなす）
- 単位はすべて **1000mあたりの秒**（距離で着差の重みが違うため。figure と同じ）

レベル − 勝ち馬との差 が「その馬がそのレースで見せた強さ」（パフォーマンス）になる。
宝塚12着と天皇賞(春)5着のどちらが上かは、これで比べる。

勝ち馬との差は MARGIN_CAP で打ち切る。大きく負けた馬は流していることが多いため。
判定日より前のレースだけで解く。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

MARGIN_CAP = 1.0        # 1000mあたり秒（2000mで2秒差）
WINDOW_DAYS = 730       # 何日前までのレースで解くか
DAMP = 0.3              # 正則化。つながりの少ないレース・馬を 0（平均）に寄せる


def prepare_runs(races: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """1走1行。勝ち馬との差（1000mあたり秒、打ち切り後）を付ける。障害は除く。"""
    meta = races[['race_id', 'date', 'race_name', 'distance', 'venue_name', 'surface',
                  'level_score']].copy()
    meta['race_id'] = meta['race_id'].astype(str)
    meta = meta[~meta['race_name'].astype(str).str.contains('障害')]
    meta['date'] = pd.to_datetime(meta['date'], errors='coerce')
    r = results[['race_id', 'horse_id', 'horse_name', 'finish_position', 'time_seconds',
                 'popularity', 'odds']].copy()
    r['race_id'] = r['race_id'].astype(str)
    r['horse_id'] = r['horse_id'].astype(str)
    r = r.merge(meta, on='race_id').dropna(subset=['date', 'finish_position', 'time_seconds'])
    km = r['distance'] / 1000.0
    win_t = r.groupby('race_id')['time_seconds'].transform('min')
    r['behind'] = ((r['time_seconds'] - win_t) / km).clip(upper=MARGIN_CAP)
    return r.sort_values(['date', 'race_id', 'finish_position']).reset_index(drop=True)


def fit(runs: pd.DataFrame, as_of, window_days: int = WINDOW_DAYS,
        damp: float = DAMP) -> Tuple[pd.Series, pd.Series]:
    """as_of より前のレースで (レベル[race_id], 能力[horse_id]) を解く。"""
    end = pd.Timestamp(as_of)
    w = runs[(runs['date'] < end) & (runs['date'] >= end - pd.Timedelta(days=window_days))]
    race_ix, race_ids = pd.factorize(w['race_id'])
    horse_ix, horse_ids = pd.factorize(w['horse_id'])
    n, nr = len(w), len(race_ids)
    rows = np.repeat(np.arange(n), 2)
    cols = np.column_stack([race_ix, nr + horse_ix]).ravel()
    vals = np.tile([1.0, -1.0], n)
    X = csr_matrix((vals, (rows, cols)), shape=(n, nr + len(horse_ids)))
    sol = lsqr(X, w['behind'].to_numpy(), damp=damp, atol=1e-6, btol=1e-6)[0]
    level = pd.Series(sol[:nr], index=race_ids, name='level')
    ability = pd.Series(sol[nr:], index=horse_ids, name='ability')
    # 全体をずらしても式は変わらないので、能力の中央値を 0 にそろえる
    shift = float(np.median(ability))
    return level - shift, ability - shift


def next_run_stats(runs: pd.DataFrame, race_ids, as_of=None) -> pd.DataFrame:
    """出走馬のその後。各レースの出走馬が次に走ったレースでの成績。

    as_of を渡すと、それより前の次走だけを数える（判定日以降の結果を使わない）。
    """
    r = runs.sort_values(['horse_id', 'date'])
    r = r.assign(next_fin=r.groupby('horse_id')['finish_position'].shift(-1),
                 next_date=r.groupby('horse_id')['date'].shift(-1))
    if as_of is not None:
        late = r['next_date'] >= pd.Timestamp(as_of)
        r.loc[late, 'next_fin'] = np.nan
    r = r[r['race_id'].isin(set(map(str, race_ids)))]
    ran = r['next_fin'].notna()
    return r.assign(ran=ran, win=ran & (r['next_fin'] == 1), top3=ran & (r['next_fin'] <= 3)
                    ).groupby('race_id').agg(
        出走=('horse_id', 'size'), 次走済=('ran', 'sum'), 次走勝利=('win', 'sum'),
        次走3着内=('top3', 'sum'))
