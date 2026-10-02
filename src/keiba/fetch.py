"""netkeiba からの取得

やっていること。

- ブラウザ相当のヘッダと Cookie を保つセッション（403 対策）
- リクエスト間隔の固定（相手サイトへの配慮。詰めないこと）
- 429 / 503 / 403 の区別と指数バックオフ
- 取得結果の分類を呼び出し側に返し、「取れなかった理由」を潰せるようにする

旧実装は失敗が握りつぶされ、403 で全件失敗しても「未発見」に見えていた。
ここでは理由を Outcome として明示的に返す。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import requests

from . import config

def encoding_for(url: str) -> str:
    """URL からページの文字コードを決める。

    netkeiba はホストで文字コードが違う。
      db.netkeiba.com    （結果データベース）  EUC-JP
      race.netkeiba.com  （出馬表・オッズ）     UTF-8

    一律に EUC-JP を指定すると出馬表が文字化けする。
    """
    return 'EUC-JP' if '//db.netkeiba.com' in url else 'UTF-8'


BROWSER_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}


class Outcome(str, Enum):
    SAVED = 'saved'
    EXISTS = 'exists'
    NOT_FOUND = 'not_found'
    BLOCKED = 'blocked'          # 403
    RATE_LIMITED = 'rate_limited'  # 429 / 503
    ERROR = 'error'


@dataclass
class Fetcher:
    """セッションとレート制御を持つ取得器。"""

    interval_sec: float = config.REQUEST_INTERVAL_SEC
    timeout: int = 20
    max_retries: int = 3
    session: requests.Session = field(default_factory=requests.Session)
    _last_request_at: float = 0.0
    warmed_up: bool = False
    stats: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.session.headers.update(BROWSER_HEADERS)

    # -- 内部 ---------------------------------------------------------------
    def _throttle(self):
        wait = self.interval_sec - (time.time() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.time()

    def _count(self, key: str):
        self.stats[key] = self.stats.get(key, 0) + 1

    def warm_up(self) -> bool:
        """トップページを1回踏んで Cookie を得る。403 対策。"""
        try:
            self._throttle()
            r = self.session.get(config.NETKEIBA_DB + '/', timeout=self.timeout)
            self.warmed_up = r.status_code == 200
            return self.warmed_up
        except requests.RequestException:
            return False

    # -- 公開 ---------------------------------------------------------------
    def get(self, url: str, referer: Optional[str] = None):
        """GET。リトライ込み。失敗時は None。"""
        if not self.warmed_up:
            self.warm_up()

        headers = {'Referer': referer} if referer else {}
        backoff = 2.0
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                r = self.session.get(url, headers=headers, timeout=self.timeout)
                r.encoding = encoding_for(url)
                if r.status_code in (429, 503):
                    self._count('rate_limited')
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return r
            except requests.RequestException:
                self._count('error')
                time.sleep(backoff)
                backoff *= 2
        return None

    def diagnose(self) -> str:
        """1リクエストだけで、この環境から取得できるかを判定する。

        Colab の IP がブロックされている場合に、本処理を走らせる前に気づくため。
        """
        url = f'{config.NETKEIBA_DB}/race/list/20260125/'
        r = self.get(url, referer=config.NETKEIBA_DB + '/')
        if r is None:
            return 'error: リクエスト自体が失敗（ネットワークまたはタイムアウト）'
        if r.status_code == 403:
            return '403: この環境のIPがブロックされています。ローカルから取得してください'
        if r.status_code != 200:
            return f'{r.status_code}: 予期しない応答'
        n = len(set(__import__('re').findall(r'/race/(\d{12})/?', r.text)))
        if n == 0:
            return '200 だが race_id を抽出できず。ページ構造が変わった可能性'
        return f'ok: 正常に取得できます（テスト日のレース数 {n}）'

    def download_race(self, race_id: str, html_dir: Path,
                      overwrite: bool = False) -> Outcome:
        """1レースのHTMLを保存する。

        保存先は `html_dir/<race_idの先頭4桁>/<race_id>.html`。
        既存のレイアウトを踏襲している。
        """
        out = html_dir / race_id[:4] / f'{race_id}.html'
        if out.exists() and not overwrite:
            self._count(Outcome.EXISTS)
            return Outcome.EXISTS

        r = self.get(f'{config.NETKEIBA_DB}/race/{race_id}',
                     referer=f'{config.NETKEIBA_DB}/')
        if r is None:
            self._count(Outcome.ERROR)
            return Outcome.ERROR
        if r.status_code == 403:
            self._count(Outcome.BLOCKED)
            return Outcome.BLOCKED
        if r.status_code in (429, 503):
            self._count(Outcome.RATE_LIMITED)
            return Outcome.RATE_LIMITED
        if r.status_code != 200 or len(r.text) < 5000 or r.text.count('/horse/') < 3:
            # 200 でも中身が無いページを「取得成功」にしない
            self._count(Outcome.NOT_FOUND)
            return Outcome.NOT_FOUND

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r.text, encoding='utf-8')
        self._count(Outcome.SAVED)
        return Outcome.SAVED
