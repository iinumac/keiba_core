# keiba_core

競馬データの収集・蓄積・分析の基盤。

netkeiba のレース結果を集め、分析しやすい形に整え、期待値ベースの予想まで行う。
Google Colab 上で動かすことを前提にしているが、ローカルでも同じコードが動く。

---

> **このリポジトリは Private です。** Colab から使うには、画面左の鍵マーク
> （シークレット）に `GITHUB_TOKEN`（GitHub の Personal Access Token / Classic、
> `repo` スコープ）を登録してください。各ノートブックの先頭セルが
> それを使って clone し、末尾セルが push します。

## クイックスタート

```python
import sys; sys.path.insert(0, 'src')
from keiba import store

con = store.connect()
con.sql("""
    SELECT venue_name, track_condition, count(*) AS n
    FROM race_results
    WHERE surface = '芝'
    GROUP BY 1, 2 ORDER BY n DESC LIMIT 10
""").df()
```

DuckDB を使わず Parquet を直接読むこともできる。

```python
import pandas as pd
df = pd.read_parquet('data/warehouse/results/year=2025/part.parquet')
```

---

## ノートブック

| | 役割 | 出力 |
|---|---|---|
| [`01_collect`](notebooks/01_collect.ipynb) | 新しいレースを検知して HTML を取得 | `data/html/YYYY/*.html` |
| [`02_build`](notebooks/02_build.ipynb) | 増えた分だけパースしてウェアハウスを更新 | `data/warehouse/` |
| [`03_train`](notebooks/03_train.ipynb) | 3着内確率モデルの学習 | `models/`, `data/master/` |
| [`04_strategy`](notebooks/04_strategy.ipynb) | 三連複の期待値・買い目 | 画面出力 |
| [`90_analysis`](notebooks/90_analysis.ipynb) | SQL による自由分析 | — |

各ノートブックは成果物を GitHub に保存し、次のノートブックはそれを取りに行く。
**別タブ・別ランタイムで実行してよい。**

---

## 設計

### データの流れ

```
netkeiba
   │  01: 開催日一覧から race_id を列挙 → 未取得のものだけ取得
   ▼
data/html/YYYY/*.html                    生HTML（Git管理）
   │  02: マニフェストで差分判定 → 変更分だけパース
   ▼
data/warehouse/races/year=YYYY/part.parquet     ← 正本
data/warehouse/results/year=YYYY/part.parquet
data/warehouse/manifest.parquet                 ← 何をどう解析したかの台帳
   │
   ├─ data/keiba.duckdb   Parquet を参照するビュー（SQL層・環境ごとに自動生成）
   ├─ 03: モデル学習
   └─ 04: 期待値・買い目
```

### 1. 新しいHTMLの検知

中央競馬は毎日開催していないので、2段階で絞る。

```
1. https://race.netkeiba.com/top/calendar.html?year=YYYY&month=M
     → その月の開催日だけが返る（月1リクエスト）
2. https://db.netkeiba.com/race/list/YYYYMMDD/
     → その日の全レースの race_id（1日1リクエスト）
```

2026年1月なら **カレンダー1回 + 開催日10回 = 11リクエスト**。
31日を順に叩くのに比べて3分の1で済み、しかも
**1/5・1/12 の月曜祝日開催を取りこぼさない**（土日固定のフィルタでは漏れる）。

レース一覧ページには**地方競馬も並んでいる**ため、中央（JRA）の10場に絞る。
2026-01-04 は全69レース中、中央は24件（中山12・京都12）だけだった。

### 2. 毎回パースしない

`data/warehouse/manifest.parquet` が
`race_id / HTMLのSHA256 / パーサ版 / 解析日時 / 頭数` を持つ。
再パースの対象になるのは次の3つだけ。

| 理由 | 判定 |
|---|---|
| `new` | マニフェストに無い |
| `changed` | HTML の SHA256 が記録と違う |
| `parser_upgraded` | `config.PARSER_VERSION` が記録より新しい |

パーサを直したら `PARSER_VERSION` を上げる。それだけで全件が再パース対象になる。

パースは CPU 律速なので並列実行する（実測 約180件/秒、全55,000件で約5分）。

> ローカルのスクリプトから `build.build()` を呼ぶときは
> `if __name__ == "__main__":` で囲むこと。macOS は spawn 方式のため、
> ガードが無いと子プロセスがスクリプトを再実行してプロセスが増え続ける。

### 3. 多角的に分析できる形

正本は **年パーティションの Parquet**。更新が入った年のファイルだけが書き換わるので
Git の差分が小さく、pandas / polars / DuckDB のどれからでも読める。

その上に DuckDB のビューを張る。データは複製せず Parquet を直接参照するため、
ウェアハウスを更新すれば DB を作り直さなくても内容は最新になる。

| ビュー | 内容 |
|---|---|
| `races` | レース単位（コース・距離・馬場・賞金など） |
| `results` | 出走馬 × レース（着順・オッズ・人気・タイムなど） |
| `race_results` | 上記を結合したもの（52列）。通常はこれを使う |

`data/keiba.duckdb` はビュー定義に絶対パスを持つため Git 管理しない。
`store.connect()` が無ければ自動生成する。

---

## モジュール

| | 役割 |
|---|---|
| `config` | パス・定数・`PARSER_VERSION` |
| `discovery` | 開催日から race_id を検知 |
| `fetch` | 取得（レート制御・403判定・リトライ） |
| `parse` | HTML → レコード |
| `manifest` | 何を解析済みかの台帳 |
| `build` | パース → ウェアハウス取り込み（並列・増分） |
| `store` | Parquet の読み書きと DuckDB |
| `features` | 特徴量計算（03 と 04 で共用） |
| `strategy` | 三連複の期待値予想エンジン |
| `gitpush` | 成果物の GitHub 保存 |

---

## 現状のデータ

| | |
|---|---|
| レース | 57,760（2010-01-05 〜 2026-09-22） |
| 出走馬レコード | 815,479 |
| 生HTML | 57,829 ファイル |
| ウェアハウス | 38MB |

パーサの出力仕様を変えたときは `config.PARSER_VERSION` を上げる。
現在は 3（コース表記の取りこぼし修正）。

HTML はあるがウェアハウスに入っていないものが 69 件ある。
これは netkeiba が結果の無いレースに返す「枠だけのページ」で、
取り込むと日付が 1970-01-01 の空行になるため `build.validate()` で弾いている。

一方、**ウェアハウスにあって HTML が無いレースは 0 件**。
旧プロジェクトではここが 432 件ずれていた。

---

## 旧プロジェクト（keiba_prediction）から変えたところ

| | 旧 | 本プロジェクト |
|---|---|---|
| race_id の探索 | 12桁IDを総当たり（年24,000通り） | 開催日一覧から列挙（1日1リクエスト） |
| 取得したHTML | Colab上に置いたまま消えていた | GitHub に保存 |
| ノートブック間の受け渡し | 同一ランタイム前提（暗黙） | 各段が GitHub 経由。別ランタイム可 |
| パース | 毎回 race_id 突合のみ | ハッシュ＋パーサ版のマニフェスト |
| パース速度 | 逐次 | 並列（約180件/秒） |
| データストア | 単一 parquet 2本 | 年パーティション + DuckDB |
| 特徴量 | 学習用と戦略用で別実装・定義が不一致 | `features.py` に集約 |
| 予想エンジン | ノートブックに11,000文字のセル | `strategy.py` |
| 空ページ | そのまま取り込み（1970年の行が69件） | `validate()` で除外 |
| マスタの最新行 | 同一日複数レースで不定 | `race_id` まで含めて決定的に |

詳細は [`docs/MIGRATION.md`](docs/MIGRATION.md)。

---

## 関連ドキュメント

- [`docs/MIGRATION.md`](docs/MIGRATION.md) — 旧プロジェクトからの移行と設計判断
- [`docs/FEATURES.md`](docs/FEATURES.md) — 特徴量の定義と未決の論点
- [`docs/odds_deepresearch.md`](docs/odds_deepresearch.md) — オッズ断層の理論（04の裏付け）
