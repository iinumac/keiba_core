"""プロジェクト共通の設定とパス。"""

from pathlib import Path

# src/keiba/config.py から2階層上がプロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / 'data'
HTML_DIR = DATA_DIR / 'html'
WAREHOUSE_DIR = DATA_DIR / 'warehouse'
MANIFEST_PATH = WAREHOUSE_DIR / 'manifest.parquet'
DUCKDB_PATH = DATA_DIR / 'keiba.duckdb'
MODEL_DIR = PROJECT_ROOT / 'models'

PARSER_VERSION = 1
"""パーサの出力仕様のバージョン。

parse.py の出力（列の追加・削除・値の意味の変更）を変えたら必ず上げること。
マニフェストがこの値を記録しており、上げるだけで全HTMLが再パース対象になる。
値を変えずにパーサを直すと、古い解析結果が残り続ける。
"""

GITHUB_OWNER = 'iinumac'
GITHUB_REPO = 'keiba_core'
RAW_BASE_URL = f'https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main'

# netkeiba
NETKEIBA_DB = 'https://db.netkeiba.com'
REQUEST_INTERVAL_SEC = 1.0
"""リクエスト間隔。相手サイトへの負荷を抑えるため、短くしないこと。"""
