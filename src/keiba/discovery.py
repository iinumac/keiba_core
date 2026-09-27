"""開催日と race_id の検知

中央競馬（JRA）は毎日開催しているわけではない。土日が中心だが、
月曜の祝日開催、年末年始、振替開催もある。全日付を順に叩くのは
無駄が多く、相手サイトへの負荷にもなる。

そこで2段階で絞る。

1. **月次カレンダー** を1回叩いて、その月の開催日だけを得る
   `https://race.netkeiba.com/top/calendar.html?year=YYYY&month=M`
   → `kaisai_date=YYYYMMDD` が開催日ぶんだけ並んでいる
2. 開催日ごとに **レース一覧** を1回叩いて race_id を得る
   `https://db.netkeiba.com/race/list/YYYYMMDD/`
   → 1リクエストでその日の全レース（複数場×12R、実測で約70件）

2026年1月の例では、31日を順に叩く代わりに
「カレンダー1回 + 開催日10回 = 11リクエスト」で済む。
しかも 1/5（月）1/12（月）の祝日開催を取りこぼさない。

旧実装（keiba_prediction の C01）は race_id を機械的に組み立てて
1件ずつ存在確認していた。年あたり
    10競馬場 × 10開催回 × 20日 × 12レース = 24,000通り
を叩くことになり、netkeiba から 403 を返される主因になっていた。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Set

from . import config
from .fetch import Fetcher

RACE_ID_RE = re.compile(r'/race/(\d{12})/?')
KAISAI_DATE_RE = re.compile(r'kaisai_date=(\d{8})')

CALENDAR_URL = 'https://race.netkeiba.com/top/calendar.html'

# 中央競馬（JRA）の競馬場コード。race_id の 5〜6 桁目。
# レース一覧ページには地方競馬（NAR）も混ざっている。2026-01-04 は
# 全69レースのうち中央は24件（中山12・京都12）だけで、
# 残る45件は地方（場コード 45 / 48 / 55 / 65）だった。
# 本プロジェクトは中央のみを対象とし、パーサも中央のページ構造を前提にしている。
JRA_VENUES: Dict[str, str] = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉',
}


def is_jra(race_id: str) -> bool:
    """中央競馬の race_id か。"""
    return len(race_id) == 12 and race_id[4:6] in JRA_VENUES


def venue_name(race_id: str) -> Optional[str]:
    return JRA_VENUES.get(race_id[4:6]) if len(race_id) == 12 else None


@dataclass
class DayResult:
    day: date
    race_ids: List[str]
    """中央競馬のみ（jra_only=True のとき）。"""
    status: int
    excluded: int = 0
    """中央以外として除外した件数。"""


def race_days_in_month(year: int, month: int, fetcher: Fetcher) -> List[date]:
    """その月の開催日。開催が無ければ空リスト。

    カレンダーが取得できなかった場合も空リストを返す。呼び出し側で
    取りこぼしに気づけるよう、例外にはしない（`discover` が警告する）。
    """
    resp = fetcher.get(f'{CALENDAR_URL}?year={year}&month={month}',
                       referer='https://race.netkeiba.com/top/')
    if resp is None or resp.status_code != 200:
        return []
    days = []
    for s in sorted(set(KAISAI_DATE_RE.findall(resp.text))):
        try:
            d = date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        except ValueError:
            continue
        if d.year == year and d.month == month:
            days.append(d)
    return days


def race_days(start: date, end: date, fetcher: Fetcher,
              on_month=None) -> List[date]:
    """期間内の開催日を、月次カレンダーから集める。"""
    days: List[date] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        found = race_days_in_month(y, m, fetcher)
        if on_month is not None:
            on_month(y, m, found)
        days.extend(d for d in found if start <= d <= end)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return sorted(days)


def race_ids_on(day: date, fetcher: Fetcher,
                jra_only: bool = True) -> DayResult:
    """その日に開催されたレースの race_id。

    Args:
        jra_only: 中央競馬のみに絞る（既定）。レース一覧ページには
            地方競馬も並んでおり、そのまま取ると中央以外が混入する。
    """
    url = f'{config.NETKEIBA_DB}/race/list/{day:%Y%m%d}/'
    resp = fetcher.get(url, referer=f'{config.NETKEIBA_DB}/')
    if resp is None:
        return DayResult(day, [], 0, 0)
    if resp.status_code != 200:
        return DayResult(day, [], resp.status_code, 0)

    all_ids = sorted(set(RACE_ID_RE.findall(resp.text)))
    ids = [r for r in all_ids if is_jra(r)] if jra_only else all_ids
    return DayResult(day, ids, 200, len(all_ids) - len(ids))


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def discover(start: date, end: date, fetcher: Fetcher,
             known: Optional[Set[str]] = None,
             on_month=None, on_day=None,
             scan_all_days: bool = False,
             jra_only: bool = True) -> List[str]:
    """期間内で未取得の race_id を返す。

    Args:
        known: すでに手元にある race_id。差し引いた分だけ返す。
        on_month: (year, month, days) を受け取るコールバック
        on_day: DayResult を受け取るコールバック
        jra_only: 中央競馬のみに絞る（既定）。レース一覧ページには地方競馬も
            並んでいるため、切ると中央以外が混入する。
        scan_all_days: カレンダーを使わず全日付を確認する。
            カレンダーのページ構造が変わったときの退避用。通常は不要。
    """
    known = known or set()

    if scan_all_days:
        days = list(daterange(start, end))
    else:
        days = race_days(start, end, fetcher, on_month=on_month)
        if not days:
            # 期間内に開催が無いのか、カレンダーが読めなかったのか区別できない
            print('⚠️ カレンダーから開催日を取得できませんでした。'
                  'ページ構造が変わった可能性があります。'
                  'scan_all_days=True で全日付を確認できます。')
            return []

    found: List[str] = []
    for day in days:
        result = race_ids_on(day, fetcher, jra_only=jra_only)
        if on_day is not None:
            on_day(result)
        found.extend(rid for rid in result.race_ids if rid not in known)

    seen: Set[str] = set()
    return [r for r in found if not (r in seen or seen.add(r))]
