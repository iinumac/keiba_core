"""データウェアハウス

正本は年パーティションの Parquet。DuckDB はその上に乗る SQL 層で、
実体を持たずビューとして Parquet を参照する。

    data/warehouse/races/year=2025/part.parquet
    data/warehouse/results/year=2025/part.parquet
    data/keiba.duckdb        ← races / results ビューの定義だけ

年で分けている理由は2つ。

- 更新が入った年の Parquet だけ書き換わるので Git の差分が小さい
- DuckDB / pandas / polars のいずれからも部分読み込みできる

パーティションキーの year は「race_id の先頭4桁（＝HTMLの格納年）」ではなく
**実際のレース開催年** を使う。両者は年跨ぎの開催でずれることがある。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from . import config

RACES = 'races'
RESULTS = 'results'
TABLES = (RACES, RESULTS)


def table_dir(table: str, warehouse: Optional[Path] = None) -> Path:
    return (warehouse or config.WAREHOUSE_DIR) / table


def partition_path(table: str, year: int, warehouse: Optional[Path] = None) -> Path:
    return table_dir(table, warehouse) / f'year={year}' / 'part.parquet'


def read_table(table: str, years: Optional[Iterable[int]] = None,
               warehouse: Optional[Path] = None,
               columns: Optional[List[str]] = None) -> pd.DataFrame:
    """テーブル全体、または指定年だけを読む。"""
    base = table_dir(table, warehouse)
    if not base.exists():
        return pd.DataFrame()
    if years is None:
        parts = sorted(base.glob('year=*/part.parquet'))
    else:
        parts = [partition_path(table, y, warehouse) for y in sorted(years)]
        parts = [p for p in parts if p.exists()]
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p, columns=columns) for p in parts],
                     ignore_index=True)


# features.build_features() が results 側に期待する、races 由来の列。
# ウェアハウスは results を正規化して持つ（同じ値をレースごとに繰り返さない）が、
# 特徴量計算はレース属性を1行に並べた形を前提にしているため、読み出し時に結合する。
RACE_COLUMNS_FOR_FEATURES = ['date', 'venue_code', 'venue_name', 'surface',
                             'distance', 'track_condition', 'race_level',
                             'level_score', 'race_name']


def read_results_enriched(years: Optional[Iterable[int]] = None,
                          warehouse: Optional[Path] = None) -> pd.DataFrame:
    """results にレース属性を結合して返す。

    `features.build_features()` にはこちらを渡すこと。
    `date` は `race_date` という名前で入る（特徴量計算がその名前を使うため）。
    """
    results = read_table(RESULTS, years=years, warehouse=warehouse)
    if results.empty:
        return results
    races = read_table(RACES, years=years, warehouse=warehouse)
    cols = ['race_id'] + [c for c in RACE_COLUMNS_FOR_FEATURES if c in races.columns]
    merged = results.merge(races[cols], on='race_id', how='left')
    if 'date' in merged.columns:
        merged = merged.rename(columns={'date': 'race_date'})
    return merged


def load_for_features(years: Optional[Iterable[int]] = None,
                      warehouse: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """特徴量計算に渡す (races, results) を返す。"""
    return (read_table(RACES, years=years, warehouse=warehouse),
            read_results_enriched(years=years, warehouse=warehouse))


def existing_race_ids(warehouse: Optional[Path] = None) -> set:
    """ウェアハウスに入っている race_id の集合。"""
    df = read_table(RACES, warehouse=warehouse, columns=['race_id'])
    return set() if df.empty else set(df['race_id'].astype(str))


def _resolve_year(df: pd.DataFrame) -> pd.Series:
    """パーティションキーにする開催年を決める。

    date 列（レース開催日）を最優先し、無い行だけ race_id 先頭4桁で補う。
    """
    year = pd.Series(pd.NA, index=df.index, dtype='Int64')
    if 'date' in df.columns:
        year = pd.to_datetime(df['date'], errors='coerce').dt.year.astype('Int64')
    if 'race_date' in df.columns:
        fallback = pd.to_datetime(df['race_date'], errors='coerce').dt.year.astype('Int64')
        year = year.fillna(fallback)
    from_id = pd.to_numeric(df['race_id'].astype(str).str[:4], errors='coerce').astype('Int64')
    return year.fillna(from_id)


def upsert(races: pd.DataFrame, results: pd.DataFrame,
           warehouse: Optional[Path] = None) -> dict:
    """新しいレコードを取り込む。同じ race_id は新しい方で置き換える。

    書き換えるのは該当する年のパーティションだけ。
    """
    warehouse = warehouse or config.WAREHOUSE_DIR
    stats = {}

    if races.empty:
        return {'races': 0, 'results': 0, 'partitions': []}

    races = races.copy()
    races['_year'] = _resolve_year(races)
    year_of = dict(zip(races['race_id'].astype(str), races['_year']))

    results = results.copy()
    results['_year'] = results['race_id'].astype(str).map(year_of)
    results['_year'] = results['_year'].fillna(_resolve_year(results))

    touched = sorted({int(y) for y in races['_year'].dropna().unique()})

    for table, df in ((RACES, races), (RESULTS, results)):
        n = 0
        for year in touched:
            incoming = df[df['_year'] == year].drop(columns=['_year'])
            if incoming.empty:
                continue
            path = partition_path(table, year, warehouse)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                old = pd.read_parquet(path)
                old = old[~old['race_id'].astype(str).isin(incoming['race_id'].astype(str))]
                merged = pd.concat([old, incoming], ignore_index=True)
            else:
                merged = incoming
            sort_cols = ['race_id'] + (['horse_number'] if 'horse_number' in merged.columns else [])
            merged.sort_values(sort_cols).to_parquet(path, index=False)
            n += len(incoming)
        stats[table] = n

    stats['partitions'] = touched
    return stats


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------

VIEW_SQL = """
CREATE OR REPLACE VIEW {table} AS
SELECT * FROM read_parquet('{glob}', hive_partitioning = true);
"""


def build_duckdb(db_path: Optional[Path] = None,
                 warehouse: Optional[Path] = None) -> Path:
    """Parquet を参照するビューを張った DuckDB を作る。

    データは複製せず、Parquet を直接読むビューだけを定義する。
    したがってウェアハウスを更新すれば、DB を作り直さなくても内容は最新になる。
    """
    import duckdb

    db_path = db_path or config.DUCKDB_PATH
    warehouse = warehouse or config.WAREHOUSE_DIR
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        for table in TABLES:
            glob = str(table_dir(table, warehouse) / 'year=*' / '*.parquet')
            con.execute(VIEW_SQL.format(table=table, glob=glob))
        # レース属性をすべて結合したビュー。分析は基本これを使う。
        con.execute("""
            CREATE OR REPLACE VIEW race_results AS
            SELECT r.*, c.* EXCLUDE (race_id)
            FROM results r
            LEFT JOIN races c USING (race_id);
        """)
    finally:
        con.close()
    return db_path


def connect(db_path: Optional[Path] = None,
            warehouse: Optional[Path] = None):
    """分析用の接続を開く。

    ビューは毎回張り直す。Parquet の列の型が変わる（distance が欠損を
    含む DOUBLE から BIGINT になる等）と、保存済みのビュー定義と食い違って
    "Contents of view were altered" で読めなくなるため。
    ビュー定義のみなので作り直しのコストは無視できる。
    """
    import duckdb

    db_path = db_path or config.DUCKDB_PATH
    build_duckdb(db_path, warehouse)
    return duckdb.connect(str(db_path))
