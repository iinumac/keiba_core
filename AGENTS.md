# AGENTS.md

AI エージェント向けの入口。人向けの説明は README.md。

## このリポジトリ

中央競馬の3着内予想。netkeiba から集めたレース結果を Parquet のウェアハウス（`data/warehouse/`）に
ため、モデルと「目線」（物差しの違う評価）・ピックアップ・要注意のリストを出す。
コードは `src/keiba/`、検証用のスクリプトは `scripts/`、文書は `docs/`、評価の結果は `reports/`。

## 頼まれることが多い作業

- **過去の開催日で予想ロジックを評価する** → `docs/EVALUATION.md` の手順どおりに行う
  （`scripts/replay_day.py <日付>` → `scripts/summarize_replays.py`）

## 必ず守ること

- **netkeiba に手元の端末から直接アクセスしない**（IPを制限されたことがある）。取得は
  `keiba.colab_remote`（Colab の VM）を通す。`KEIBA_ALLOW_LOCAL_FETCH=1` で制限を外さない
- Colab でも大量に取らない（1回に数日分まで）。使い終わったセッションは止める
  （`colab_remote.stop()`、確認は `colab sessions`）
- 秘密情報（トークン・パスワード・認証コード）をコマンドや `colab --env` に書かない。
  Colab へのログインが切れていたら、人に頼む
- 予想ロジックの数字（`src/keiba/flags.py` の境目・重み、`src/keiba/views.py` のリストの条件）は
  頼まれない限り変えない。変えたほうがよい根拠があれば提案にとどめる
- 評価で未来の情報を使わない（評価する日以降の結果・オッズを入れない）
- コードを変えたら `python3 tests/test_pipeline.py` を回し、「✅ すべて成功」を確かめる
- コミット・push は頼まれたときだけ。`data/shutuba/*_weekend.json` と `flop_explore.csv` は
  コミットしない。push 先は個人アカウント（このリポジトリの git 設定で済んでいる）

## 書き方

- 説明・コメント・レポートは日本語。数字には頭数・レース数を添える
- 文書の言葉は、競馬の用語以外はやさしく（「3着内率」「人気」「市場の見込み」など既存の言い方に合わせる）
