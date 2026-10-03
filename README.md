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
| [`00_run`](notebooks/00_run.ipynb) | **1〜5 を範囲指定してまとめて実行。通常はこれだけ使う** | — |
| [`01_collect`](notebooks/01_collect.ipynb) | 1. 取り込み — 新しいレースを検知して HTML を取得 | `data/html/YYYY/*.html` |
| [`02_build`](notebooks/02_build.ipynb) | 2. パース — 増えた分だけウェアハウスへ | `data/warehouse/` |
| [`03_train`](notebooks/03_train.ipynb) | 3. 学習 — 3着内確率モデル・**買い目用モデル**（`model_strategy.pkl`）とマスタ | `models/`, `data/master/` |
| [`04_strategy`](notebooks/04_strategy.ipynb) | 4. 戦略評価 — 買い方の検証 | 画面出力 |
| [`05_predict`](notebooks/05_predict.ipynb) | 5. 予想 — **発走直前に何度でも回す用**。生HTML不要。買い目用モデルで採点（2回目以降は一瞬） | `data/shutuba/` |
| [`90_analysis`](notebooks/90_analysis.ipynb) | 分析 — SQL による自由分析（独立） | — |

01〜04 が一本のパイプライン、90 はそれを横から覗く道具。

**処理の実体は `src/keiba/stages.py` にあり、00_run も 01〜04 も同じ関数を呼ぶ。**
ノートブック側にロジックを書かないのは、片方だけ直して食い違う事故を避けるため。

```python
from keiba import pipeline

pipeline.run(1, 5)          # 取り込み → パース → 学習 → 戦略評価 → 今週末の買い目
pipeline.run(2, 3)          # パースと学習だけ
pipeline.run('train', 5)    # 名前でも指定できる
pipeline.run(5, 5)          # 今週末の買い目だけ
pipeline.run(5, 5, within_minutes=90)   # 発走90分以内のレースだけ
```

途中の段が失敗したらそこで止まる。前の段の出力が次の段の入力になるため、
失敗を無視して進めると壊れたデータで学習してしまう。

各段は成果物を GitHub に保存し、次の段はそれを取りに行く。
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

開始日を自動で決めるときは、手持ちの最新日から **14日さかのぼって**確認する。
db.netkeiba は結果の反映が遅れることがあり（2026-09-27 時点で 9/26 の中央は
未反映だった）、「最新日の翌日から」にすると反映前に通過した日が
二度と見られなくなるため。取得済みの race_id は除外されるので、
増えるのは開催日ぶんのリクエストだけ。

### 抜けの検出

取りこぼしは静かに起きる。netkeiba は結果の反映が遅れることがあり、
**取得できなかった日と、そもそも開催が無かった日は見分けがつかない**。

race_id は `YYYY` + 場コード2桁 + 開催回2桁 + 開催日2桁 + レース番号2桁。
先頭10桁が同じものが「ある競馬場のある開催日」で、通常はレース番号1〜12が揃う
（実データで 4,826 開催日のうち 4,798 が12R）。ここから抜けを割り出す。

```python
from keiba import audit, stages

report = audit.local()      # 通信なし
report.print_report()

if report.gaps:
    stages.collect(race_ids=report.gaps)   # 欠番だけ取得
```

| 検査 | 意味 |
|---|---|
| **内部の欠番** | レース番号が飛んでいる。**確実な抜け** |
| 末尾の不足 | 12Rに満たない。開催中止の可能性があり要確認 |
| HTMLが無いレース | ウェアハウスにあるのに HTML が無い。再構築すると失われる |
| 取り込まれていないHTML | 結果の無い空ページ |

日付でまとめてはいけない。開催が順延されると同じ開催日が2日にまたがり、
存在するレースを欠番と誤検出する（実際に3件そうなった）。

`audit.online(start, end)` は netkeiba のカレンダーと直接突き合わせる。
確実だがリクエストを使うので、期間を絞って使う。

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
| `payouts` | レース × 券種 × 組み合わせ（払戻・人気） |
| `race_results` | races と results を結合（52列）。通常はこれを使う |
| `race_payouts` | payouts にレース属性を結合したもの |

払戻を別テーブルにしているのは、同着で組数が増えるため横持ちでは破綻するから。
詳細とパターン一覧は [`docs/PAYOUTS.md`](docs/PAYOUTS.md)。

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
| `strategy` | 期待値予想エンジン |
| `shutuba` | 週末の出馬表とオッズの取得 |
| `betting` | 買い方の決定（三連複・馬連・ワイド） |
| `market` | 単勝オッズ → 3着内確率（Harville） |
| `segments` | レースの区分と市場バイアスの判定 |
| `backtest` | 的中率と回収率による戦略評価 |
| `gitpush` | 成果物の GitHub 保存 |

---

## 現状のデータ

| | |
|---|---|
| レース | 57,810（2010-01-05 〜 2026-09-27） |
| 出走馬レコード | 816,131 |
| 払戻 | 692,004 |
| 生HTML | 57,879 ファイル |

パーサの出力仕様を変えたときは `config.PARSER_VERSION` を上げる。
現在は 4（払戻の抽出を追加）。

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
- [`docs/PAYOUTS.md`](docs/PAYOUTS.md) — 払戻データのパターンと、戦略評価の考え方
- [`docs/MARKET.md`](docs/MARKET.md) — 市場を上回れているかの検証と結論
- [`docs/SEGMENTS.md`](docs/SEGMENTS.md) — どのレースが荒れ、どのレースが儲かるか
- [`docs/BETTING.md`](docs/BETTING.md) — 券種と買い方の選び方、買わない判断の閾値
- [`docs/SHUTUBA.md`](docs/SHUTUBA.md) — 週末の出馬表の取得（JRA公式が直リンク不可な件を含む）
- [`docs/odds_deepresearch.md`](docs/odds_deepresearch.md) — オッズ断層の理論
