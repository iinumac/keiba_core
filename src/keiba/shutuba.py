"""週末の出馬表を取ってくる

## JRA公式サイトは使えない

jra.go.jp は POST ベースのナビゲーションで、リンクに
`doAction('/JRADB/accessS.html', 'pw01sde...')` のようなトークンを渡す。
URLを直接叩けないうえ、トークンの生成規則はセッションに依存する。

同じ出馬表は netkeiba から race_id で直接取れるので、そちらを使う。
race_id は既に `discovery` で検知できる。

## 取得の流れ

    1. カレンダー            開催日を得る（未来の日付も返る）
       race.netkeiba.com/top/calendar.html?year=YYYY&month=M

    2. 当日のレース一覧       race_id を得る
       race.netkeiba.com/top/race_list_sub.html?kaisai_date=YYYYMMDD
       ※ race_list.html は JavaScript で後から読むので中身が空。
          実データは race_list_sub.html にある

    3. 出馬表                出走馬・枠番・斤量・騎手を得る
       race.netkeiba.com/race/shutuba.html?race_id=XXXXXXXXXXXX

    4. オッズAPI             単勝オッズと人気を得る
       race.netkeiba.com/api/api_get_jra_odds.html?race_id=...&type=1
       返り値は {"data": {"odds": {"1": {"01": ["9.1", "0", "5"], ...}}}}
       馬番 -> [オッズ, 不明, 人気]

## 文字コード

race.netkeiba.com は UTF-8、db.netkeiba.com は EUC-JP。
`fetch.encoding_for` が振り分ける。

## オッズの発表時刻

前日はまだ `---.-` のことがある。確定オッズが要るなら発走直前に取ること。
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from . import discovery
from .fetch import Fetcher

RACE_HOST = 'https://race.netkeiba.com'
RACE_ID_RE = re.compile(r'race_id=(\d{12})')


def race_ids_on(day: dt.date, fetcher: Fetcher, jra_only: bool = True) -> List[str]:
    """その日に開催されるレースの race_id。**未来の日付でも取れる。**

    結果データベース（db.netkeiba）は終わったレースしか持たないため、
    出馬表側の一覧を使う。
    """
    resp = fetcher.get(f'{RACE_HOST}/top/race_list_sub.html?kaisai_date={day:%Y%m%d}',
                       referer=f'{RACE_HOST}/top/race_list.html')
    if resp is None or resp.status_code != 200:
        return []
    ids = sorted(set(RACE_ID_RE.findall(resp.text)))
    return [r for r in ids if discovery.is_jra(r)] if jra_only else ids


def upcoming_race_days(fetcher: Fetcher, days_ahead: int = 10,
                       start: Optional[dt.date] = None) -> List[dt.date]:
    """これから開催される日を、カレンダーから拾う。"""
    start = start or dt.date.today()
    end = start + dt.timedelta(days=days_ahead)
    return [d for d in discovery.race_days(start, end, fetcher) if d >= start]


ODDS_STATUS = {'before': '発売前', 'middle': '発売中', 'final': '確定', 'result': '確定'}
"""オッズAPI の status。middle は前日・当日を問わず「発売中の途中経過」で、
発走後は result が返る。"""


def fetch_odds(race_id: str, fetcher: Fetcher) -> Dict:
    """単勝オッズと人気、そして**いつ時点のオッズか**。

    オッズは発走直前まで動く。直前に取るほど市場の評価を正しく反映するので、
    いつ更新されたものかが分かるようにしておく。

    Returns:
        {'odds': {馬番: {'odds': x, 'popularity': n}},
         'updated_at': '2026-10-02 23:42:16', 'status': 'middle'}
        未発表なら odds は空。
    """
    # action=init は1時間近く前のスナップショットを返すことがある
    # （13:24 に取得して 12:55 時点）。update は最新を返すので先に試し、
    # 空なら init に戻す。前日など update がまだ空を返す場面がある
    empty = {'odds': {}, 'updated_at': None, 'status': None}
    for action in ODDS_ACTIONS:
        got = _fetch_odds_once(race_id, fetcher, action)
        if got['odds']:
            return got
        empty = got if got['status'] else empty
    return empty


ODDS_ACTIONS = ('update', 'init')
"""オッズAPI の action。update が最新、init は古いことがある。"""


def _fetch_odds_once(race_id: str, fetcher: Fetcher, action: str) -> Dict:
    empty = {'odds': {}, 'updated_at': None, 'status': None}
    resp = fetcher.get(
        f'{RACE_HOST}/api/api_get_jra_odds.html?race_id={race_id}&type=1&action={action}',
        referer=f'{RACE_HOST}/odds/index.html?race_id={race_id}')
    if resp is None or resp.status_code != 200:
        return empty
    try:
        payload = json.loads(resp.text)
        data = payload.get('data') or {}
        table = (data.get('odds') or {}).get('1') or {}
    except (ValueError, AttributeError):
        return empty

    out: Dict[int, Dict[str, float]] = {}
    for umaban, values in table.items():
        try:
            out[int(umaban)] = {'odds': float(values[0]), 'popularity': int(values[2])}
        except (TypeError, ValueError, IndexError):
            continue
    return {'odds': out, 'updated_at': data.get('official_datetime'),
            'status': payload.get('status')}


def _text(node, default: str = '') -> str:
    return node.get_text(' ', strip=True) if node else default


def _weight_in(node) -> Optional[float]:
    """行から馬体重を拾う。当日まで空欄なので、無ければ None。"""
    m = re.search(r'(\d{3})\s*\(\s*[+-]?\d+\s*\)', _text(node))
    return float(m.group(1)) if m else None


def _id_in(node, kind: str) -> Optional[str]:
    """行の中の /horse/xxxx/ や /jockey/result/recent/xxxx/ から ID を拾う。"""
    a = node.find('a', href=re.compile(rf'/{kind}/'))
    if a is None:
        return None
    m = re.search(rf'/{kind}/(?:result/recent/)?(\w+)', a['href'])
    return m.group(1) if m else None


def fetch_race_card(race_id: str, fetcher: Fetcher,
                    with_odds: bool = True) -> Optional[Dict]:
    """1レースの出馬表を取る。"""
    resp = fetcher.get(f'{RACE_HOST}/race/shutuba.html?race_id={race_id}',
                       referer=f'{RACE_HOST}/top/')
    if resp is None or resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, 'html.parser')

    table = soup.find('table', class_=re.compile('Shutuba'))
    if table is None:
        return None

    # ページ上部にレース一覧のナビがあり、そこにも同名のクラスが出る。
    # 当該レースの情報は RaceList_NameBox の中にあるので、そこに絞る。
    box = soup.find('div', class_='RaceList_NameBox') or soup
    info = _text(box.find('div', class_='RaceData01'))
    cond = _text(box.find('div', class_='RaceData02'))
    m_dist = re.search(r'([芝ダ障])[^\d]*(\d+)m', info)
    horses = []
    for tr in table.find_all('tr', class_=re.compile('HorseList')):
        tds = tr.find_all('td')
        if len(tds) < 7:
            continue
        def num(i):
            try:
                return int(_text(tds[i]))
            except ValueError:
                return None
        name = tr.find('span', class_='HorseName') or tr.find('a')
        horses.append({
            'bracket': num(0),
            'horse_number': num(1),
            'horse_name': _text(name),
            # 馬名で過去走を引くと同名馬・表記ゆれで取り違える。
            # 出馬表には /horse/{id}/ のリンクがあるので、それを使う。
            'horse_id': _id_in(tr, 'horse'),
            'jockey_id': _id_in(tr, 'jockey'),
            'trainer_id': _id_in(tr, 'trainer'),
            'sex_age': _text(tds[4]),
            'impost': _try_float(_text(tds[5])),
            'jockey_name': _text(tds[6]),
            # 馬体重は当日にならないと出ない。出ていれば使う
            'horse_weight': _weight_in(tr),
        })
    horses = [h for h in horses if h['horse_number'] and h['horse_name']]

    card = {
        'race_id': race_id,
        # RaceName は div ではなく h1。タグを決め打ちしない
        'race_name': _text(box.find(class_='RaceName')),
        'race_num': _try_int(_text(box.find(class_='RaceNum'))),
        'venue_name': discovery.venue_name(race_id),
        'info': info,
        'condition': cond,
        'grade': _grade_from(box, soup, cond),
        'horse_count': _try_int(re.search(r'(\d+)頭', cond).group(1)) if re.search(r'(\d+)頭', cond) else None,
        'track_condition': (re.search(r'馬場:(\S+)', info) or [None, None])[1]
                           if re.search(r'馬場:(\S+)', info) else None,
        'weather': (re.search(r'天候:(\S+)', info) or [None, None])[1]
                   if re.search(r'天候:(\S+)', info) else None,
        'surface': m_dist.group(1) if m_dist else None,
        'distance': int(m_dist.group(2)) if m_dist else None,
        'start_time': (re.search(r'(\d{1,2}:\d{2})発走', info) or [None, None])[1]
                      if re.search(r'(\d{1,2}:\d{2})発走', info) else None,
        'horses': horses,
    }

    if with_odds:
        info_odds = fetch_odds(race_id, fetcher)
        table = info_odds['odds']
        for h in horses:
            o = table.get(h['horse_number'], {})
            h['odds'] = o.get('odds')
            h['popularity'] = o.get('popularity')
        card['odds_available'] = bool(table)
        card['odds_updated_at'] = info_odds['updated_at']
        card['odds_status'] = ODDS_STATUS.get(info_odds['status'], info_odds['status'])
    return card


GRADE_ICON = {'1': 'G1', '2': 'G2', '3': 'G3', '4': 'OP', '5': 'OP',
              '15': 'L', '16': 'L', '17': 'L'}


def _grade_from(box, soup, cond: str) -> Optional[str]:
    """グレード。アイコンのクラス名か、タイトル・条件から拾う。

    クラス名は `Icon_GradeType15`（リステッド）のように2桁のことがある。
    `\d` 1文字で取ると 15 を 1（G1）と誤認するので、数字全体を取る。
    """
    icon = box.find('span', class_=re.compile(r'Icon_GradeType\d'))
    if icon is not None:
        m = re.search(r'Icon_GradeType(\d+)', ' '.join(icon.get('class', [])))
        if m and m.group(1) in GRADE_ICON:
            return GRADE_ICON[m.group(1)]
    title = _text(soup.find('title'))
    for pat, g in ((r'\(G1\)|\(GI\)', 'G1'), (r'\(G2\)|\(GII\)', 'G2'),
                   (r'\(G3\)|\(GIII\)', 'G3'), (r'\(L\)', 'L')):
        if re.search(pat, title):
            return g
    for word in ('オープン', '3勝', '2勝', '1勝', '未勝利', '新馬'):
        if word in cond:
            return word
    return None


def _try_float(s: str) -> Optional[float]:
    try:
        return float(re.sub(r'[^\d.]', '', s))
    except (ValueError, TypeError):
        return None


def _try_int(s: str) -> Optional[int]:
    m = re.search(r'\d+', s or '')
    return int(m.group()) if m else None


def minutes_to_post(card: Dict, now: Optional[dt.datetime] = None) -> Optional[int]:
    """発走まであと何分か。過ぎていれば負の値。"""
    if not card.get('start_time') or not card.get('date'):
        return None
    try:
        post = dt.datetime.fromisoformat(f"{card['date']}T{card['start_time']}:00")
    except ValueError:
        return None
    return int((post - (now or dt.datetime.now())).total_seconds() // 60)


def fetch_weekend(fetcher: Optional[Fetcher] = None, days_ahead: int = 10,
                  with_odds: bool = True, on_race=None,
                  race_ids: Optional[List[str]] = None) -> List[Dict]:
    """これから開催される全レースの出馬表をまとめて取る。

    リクエスト数は「月1 + 開催日数 + レース数×2（出馬表とオッズ）」。
    1日36レースなら約150。1秒間隔なので3分ほどかかる。
    """
    fetcher = fetcher or Fetcher()
    cards: List[Dict] = []

    if race_ids is not None:
        # race_id を指定しての取り直し。オッズだけ更新したいときに使う
        for rid in race_ids:
            card = fetch_race_card(rid, fetcher, with_odds=with_odds)
            if card:
                cards.append(card)
            if on_race is not None:
                on_race(None, rid, card)
        return cards

    for day in upcoming_race_days(fetcher, days_ahead):
        ids = race_ids_on(day, fetcher)
        for rid in ids:
            card = fetch_race_card(rid, fetcher, with_odds=with_odds)
            if card:
                card['date'] = day.isoformat()
                cards.append(card)
            if on_race is not None:
                on_race(day, rid, card)
    return cards


UNRUN = 99
"""まだ走っていない行に入れる着順。

`features.clean` は着順のある行だけを残すので、NaN のままだと落とされる。
かといって 1 や 0 にすると is_win / is_top3 が立ってしまう。99 なら
どちらも 0 になり、過去の集計を汚さない。出馬表の行はその馬にとって
最後の行なので、ここから未来へ shift される先も無い。
"""


def _sex_age(text) -> Tuple[Optional[str], Optional[int]]:
    """「牡2」→ ('牡', 2)。"""
    m = re.match(r'([牡牝セ])\s*(\d+)', str(text or ''))
    return (m.group(1), int(m.group(2))) if m else (None, None)


def _first_prize(condition: str) -> Optional[float]:
    """条件欄の「本賞金:590,240,150,89,59万円」から1着賞金（万円）を取る。

    level_score は本来レース後の賞金から決まるが、出馬表にも載っている。
    クラス名から推定する必要はない。
    """
    m = re.search(r'本賞金[:：]\s*([\d,]+)', str(condition or ''))
    if not m:
        return None
    return float(m.group(1).split(',')[0])


SURFACE_TO_WAREHOUSE = {'芝': '芝', 'ダ': 'ダート', '障': '障害'}
"""出馬表の芝ダ表記 → ウェアハウスの表記。

出馬表は「ダ」、ウェアハウスは「ダート」。そのままだと `surface_map` に
当たらず、ダート戦の芝ダ特徴量がすべて欠損扱いになる。
障害は「障害」にしておけば `features.clean` が落とす（モデルは障害戦を
学習していない）。
"""


def to_result_rows(cards: List[Dict]) -> 'pandas.DataFrame':
    """出馬表を results と同じ形の行に直す。

    学習時の特徴量は shift()/expanding() で**過去走だけ**から作られる。
    だからこの行を results の末尾に足して `features.build_features` を
    通せば、そのまま予測用の特徴量になる。馬の紐付けは horse_id なので
    同名馬で取り違えることもない。

    走ってみないと分からない列（タイム・上がり・着差）は None のまま。
    馬体重は当日になれば出馬表に出るので、出ていれば使う。
    """
    import pandas as pd
    from .parse import classify_race_level

    rows = []
    for card in cards:
        prize = _first_prize(card.get('condition'))
        level = classify_race_level(prize)[1] if prize is not None else None
        for h in card.get('horses', []):
            sex, age = _sex_age(h.get('sex_age'))
            rows.append({
                'race_id': card['race_id'],
                'race_date': card.get('date'),
                'race_name': card.get('race_name'),
                'horse_id': h.get('horse_id'),
                'horse_name': h.get('horse_name'),
                'horse_number': h.get('horse_number'),
                'gate_number': h.get('bracket'),
                'jockey_id': h.get('jockey_id'),
                'jockey_name': h.get('jockey_name'),
                'trainer_id': h.get('trainer_id'),
                'surface': SURFACE_TO_WAREHOUSE.get(card.get('surface'),
                                                    card.get('surface')),
                'distance': card.get('distance'),
                'level_score': level,
                'impost': h.get('impost'),
                'horse_weight': h.get('horse_weight'),
                'odds': h.get('odds'),
                'popularity': h.get('popularity'),
                'sex': sex,
                'age': age,
                'finish_position': UNRUN,
                'is_finished': True,
                'is_unrun': True,
                'time_seconds': None,
                'last_3f': None,
                'margin': None,
                'weight_change': None,
                'prize_money': 0.0,
            })
    return pd.DataFrame(rows)


def to_json(cards: List[Dict], indent: int = 1) -> str:
    return json.dumps(cards, ensure_ascii=False, indent=indent)
