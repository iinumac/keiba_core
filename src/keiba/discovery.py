"""開催日から実在する race_id を検知する

旧実装（keiba_prediction の C01）は race_id を機械的に組み立てて
1件ずつ存在確認していた。年あたり
    10競馬場 × 10開催回 × 20日 × 12レース = 24,000通り
を叩くことになり、大半が空振りになる。netkeiba から 403 を返される
主因はこれだった。

ここでは開催日一覧ページを1回叩いて、その日の race_id をまとめて得る。

    https://db.netkeiba.com/race/list/YYYYMMDD/

1リクエストでその日の全レース（複数場×12R、実測で71件）が取れるので、
週次更新なら7リクエストで検知が終わる。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, List, Optional, Set

from . import config
from .fetch import Fetcher

RACE_ID_RE = re.compile(r'/race/(\d{12})/?')


@dataclass
class DayResult:
    day: date
    race_ids: List[str]
    status: int


def race_ids_on(day: date, fetcher: Fetcher) -> DayResult:
    """その日に開催された全レースの race_id。開催が無ければ空リスト。"""
    url = f'{config.NETKEIBA_DB}/race/list/{day:%Y%m%d}/'
    resp = fetcher.get(url, referer=f'{config.NETKEIBA_DB}/')
    if resp is None:
        return DayResult(day, [], 0)
    if resp.status_code != 200:
        return DayResult(day, [], resp.status_code)
    ids = sorted(set(RACE_ID_RE.findall(resp.text)))
    return DayResult(day, ids, resp.status_code)


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def discover(start: date, end: date, fetcher: Fetcher,
             known: Optional[Set[str]] = None,
             weekends_and_holidays_only: bool = False,
             on_day=None) -> List[str]:
    """期間内で未取得の race_id を返す。

    Args:
        known: すでに手元にある race_id。差し引いた分だけ返す。
        weekends_and_holidays_only: 土日だけを見る。JRA は原則土日開催だが
            年末年始や振替開催で平日もあるため、既定では全日を確認する。
            リクエストを減らしたいときだけ True にする。
        on_day: 1日処理するごとに呼ばれるコールバック (DayResult) -> None
    """
    known = known or set()
    found: List[str] = []
    for day in daterange(start, end):
        if weekends_and_holidays_only and day.weekday() < 5:
            continue
        result = race_ids_on(day, fetcher)
        if on_day is not None:
            on_day(result)
        found.extend(rid for rid in result.race_ids if rid not in known)
    # 重複を除きつつ順序を保つ
    seen: Set[str] = set()
    return [r for r in found if not (r in seen or seen.add(r))]
