"""HTMLパーサ

netkeiba のレース結果HTMLを1ファイル単位でパースする。

keiba_prediction/src/scraper/parser.py から単体パース部分を引き継いだもの。
一括処理・増分判定は manifest.py / store.py が担うため、ここには持たない。

パーサの出力を変えたときは config.PARSER_VERSION を上げること。
マニフェストがそれを見て自動的に再パースする。
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple


COURSE_RE = re.compile(
    r'(障)?(芝|ダ(?:ート)?)\s*(右|左)?\s*((?:[外内直線\-\s]|\d+周)*)(\d+)m')
"""コース表記から 障害/馬場/回り/コース区分/距離 を取り出す。

全HTML（54,889件）を走査した結果、表記は14種類あり、いずれもこの式で解析できる。

    ダ右1400m / 芝左2000m / 芝右 外1600m / 芝左 内2000m
    芝直線1000m      新潟の直線コース。「直線」で2文字
    障芝 外-内2890m  障害戦。コース区分がハイフンで連結される
    芝右 内2周3600m  中山のステイヤーズS。距離の前に「2周」が入る
    障芝 ダート3000m 障害戦の芝・ダート混合

旧実装は方向の後を1文字ぶんしか見ておらず、776レースで surface と distance が
None になっていた。テストから参照するため、ここに定数として置く（二重管理を避ける）。
"""


PAY_TABLE_RE = re.compile(
    r'<table[^>]*class="[^"]*pay_table_01[^"]*"[^>]*>(.*?)</table>', re.S)
PAY_TR_RE = re.compile(r'<tr[^>]*>(.*?)</tr>', re.S)
PAY_CELL_RE = re.compile(r'<(th|td)[^>]*>(.*?)</\1>', re.S)
BR_RE = re.compile(r'<br\s*/?>', re.I)
STRIP_TAG_RE = re.compile(r'<[^>]+>')

BET_TYPES = ('単勝', '複勝', '枠連', '馬連', 'ワイド', '馬単', '三連複', '三連単')
"""払戻表に現れる券種。出走頭数によって発売されないものがある。

  8頭以下 → 枠連なし（実データで2,242レース）
  4頭以下 → 複勝なし（実データで1レース）
"""


def _pay_cells(raw: str) -> List[str]:
    """<br> 区切りのセルを値のリストにする。

    同着や複勝の着順ぶんだけ値が並ぶ。全レースを走査した結果、
    組み合わせ・払戻・人気の個数は必ず一致していた（不一致0件）。
    """
    out = []
    for part in BR_RE.split(raw):
        v = STRIP_TAG_RE.sub('', part).replace('\xa0', ' ').strip()
        if v:
            out.append(v)
    return out


def _to_int(text: str) -> Optional[int]:
    t = re.sub(r'[,円\s]', '', text)
    return int(t) if t.isdigit() else None


def parse_payouts(html: str, race_id: str) -> List[Dict]:
    """払戻表を「レース×券種×組み合わせ」の行に展開する。

    横持ち（券種ごとに列を作る）にすると同着で破綻する。実データには
    3着が3頭同着でワイドが7組、三連複が3組になったレースが2件あった。
    縦持ちなら行が増えるだけで済む。

    Returns:
        [{race_id, bet_type, seq, combination, horse_numbers, payout, popularity}, ...]
    """
    rows: List[Dict] = []
    for block in PAY_TABLE_RE.findall(html):
        for tr in PAY_TR_RE.findall(block):
            cells = PAY_CELL_RE.findall(tr)
            if len(cells) < 4:
                continue
            bet = STRIP_TAG_RE.sub('', cells[0][1]).strip()
            if bet not in BET_TYPES:
                continue
            combos = _pay_cells(cells[1][1])
            pays = _pay_cells(cells[2][1])
            pops = _pay_cells(cells[3][1])
            if not (len(combos) == len(pays) == len(pops)):
                # 走査では0件だったが、崩れた行は取り込まない
                continue
            for i, (combo, pay, pop) in enumerate(zip(combos, pays, pops)):
                nums = [int(x) for x in re.findall(r'\d+', combo)]
                rows.append({
                    'race_id': race_id,
                    'bet_type': bet,
                    'seq': i,
                    'combination': '-'.join(str(n) for n in nums),
                    'horse_numbers': nums,
                    'payout': _to_int(pay),
                    'popularity': _to_int(pop),
                })
    return rows


def classify_race_level(prize_money: float) -> Tuple[str, int]:
    """
    1着賞金からレースレベルを分類
    
    Args:
        prize_money: 1着賞金（万円）
    
    Returns:
        (level, score): レベル（S/A/B/C/D/E）とスコア（6-1）
    """
    if prize_money >= 10000:
        return 'S', 6  # G1
    elif prize_money >= 5000:
        return 'A', 5  # G2
    elif prize_money >= 3000:
        return 'B', 4  # G3, リステッド
    elif prize_money >= 1500:
        return 'C', 3  # オープン, 3勝クラス
    elif prize_money >= 750:
        return 'D', 2  # 2勝, 1勝クラス
    else:
        return 'E', 1  # 新馬, 未勝利


def parse_time_to_seconds(time_str: str) -> Optional[float]:
    """
    タイム文字列を秒数に変換
    
    Args:
        time_str: "1:08.8" 形式のタイム
    
    Returns:
        秒数（float）または None
    """
    if not time_str:
        return None
    
    try:
        match = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            decimals = int(match.group(3))
            return minutes * 60 + seconds + decimals / 10
    except:
        pass
    
    return None


def parse_horse_weight(weight_str: str) -> Tuple[Optional[int], Optional[int]]:
    """
    馬体重文字列をパース
    
    Args:
        weight_str: "462(-2)" 形式
    
    Returns:
        (体重, 増減) のタプル
    """
    if not weight_str:
        return None, None
    
    try:
        match = re.match(r'(\d+)\(([\+\-]?\d+)\)', weight_str.strip())
        if match:
            weight = int(match.group(1))
            change = int(match.group(2))
            return weight, change
    except:
        pass
    
    return None, None


def parse_passing_order(passing_str: str) -> Optional[List[int]]:
    """
    通過順をパース
    
    Args:
        passing_str: "1-1" や "3-3-2-1" 形式
    
    Returns:
        通過順リスト
    """
    if not passing_str:
        return None
    
    try:
        parts = passing_str.strip().split('-')
        return [int(p) for p in parts if p.isdigit()]
    except:
        return None


def parse_race_html_full(html_path: Path) -> Dict:
    """
    レースHTMLをパースして全データを抽出
    
    Args:
        html_path: HTMLファイルパス
    
    Returns:
        dict: {
            'race_info': レース情報,
            'horses': 出走馬リスト,
            'payouts': 払戻リスト（レース×券種×組み合わせ）
        }
    """
    raw_html = Path(html_path).read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(raw_html, 'html.parser')

    race_id = Path(html_path).stem
    
    # ============ レース情報 ============
    race_info = {
        'race_id': race_id,
        'venue_code': race_id[4:6] if len(race_id) >= 6 else None,  # 競馬場コード
        'kaisai': race_id[6:8] if len(race_id) >= 8 else None,     # 開催回
        'day': race_id[8:10] if len(race_id) >= 10 else None,       # 開催日
        'race_num': race_id[10:12] if len(race_id) >= 12 else None, # レース番号
    }
    
    # 競馬場名マッピング
    venue_map = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }
    race_info['venue_name'] = venue_map.get(race_info['venue_code'], '')
    
    # レース名・詳細
    race_data = soup.find('dl', class_='racedata')
    if race_data:
        h1 = race_data.find('h1')
        race_info['race_name'] = h1.text.strip() if h1 else ''
        
        span = race_data.find('span')
        if span:
            info_text = span.text.strip()
            
            # 距離・コース
            # 実際に現れる表記:
            #   "芝右1600m" "ダ右1200m" "芝右 外1600m" "芝左 内2000m"
            #   "芝直線1000m"      新潟の直線コース。「直線」で2文字
            #   "障芝 外-内2890m"  障害戦。コース区分がハイフンで連結される
            #   "芝右 内2周3600m"  中山のステイヤーズS。距離の前に「2周」が入る
            # 旧実装は方向の後を (外|内|直) 1文字ぶんしか見ておらず、
            # distance も surface も None になっていた。
            # 全HTMLを走査した結果、コース表記は14種類あり、下の式で全て解析できる。
            match = COURSE_RE.search(info_text)
            if match:
                is_obstacle = match.group(1) is not None  # 障害レース
                surface = match.group(2)
                race_info['surface'] = 'ダート' if surface.startswith('ダ') else surface
                race_info['direction'] = match.group(3) or ''
                race_info['course_type'] = match.group(4).strip()
                race_info['distance'] = int(match.group(5))
            
            # 天候
            weather_match = re.search(r'天候\s*[:：]\s*(\S+)', info_text)
            race_info['weather'] = weather_match.group(1) if weather_match else ''
            
            # 馬場状態
            condition_match = re.search(r'(芝|ダート)\s*[:：]\s*(\S+)', info_text)
            race_info['track_condition'] = condition_match.group(2) if condition_match else ''
            
            # 発走時刻
            time_match = re.search(r'発走\s*[:：]\s*(\d+:\d+)', info_text)
            race_info['start_time'] = time_match.group(1) if time_match else ''
    
    # 日付・詳細情報
    date_elem = soup.find('p', class_='smalltxt')
    if date_elem:
        date_text = date_elem.text
        
        date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_text)
        if date_match:
            race_info['date'] = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
            race_info['year'] = int(date_match.group(1))
            race_info['month'] = int(date_match.group(2))
        
        # クラス情報を抽出
        race_info['class_info'] = date_text.strip()
    
    # ============ 出走馬情報 ============
    horses = []
    result_table = soup.find('table', class_='race_table_01')
    
    if result_table:
        rows = result_table.find_all('tr')[1:]  # ヘッダースキップ
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 10:
                continue
            
            horse = {'race_id': race_id}
            
            # 基本情報のセルインデックス
            # 0:着順, 1:枠番, 2:馬番, 3:馬名, 4:性齢, 5:斤量, 6:騎手, 7:タイム, 8:着差
            
            # 着順
            finish = cells[0].text.strip()
            horse['finish_position'] = int(finish) if finish.isdigit() else None
            horse['is_finished'] = finish.isdigit()
            horse['dnf_reason'] = finish if not finish.isdigit() else None  # 中止, 除外 等
            
            # 枠番・馬番
            gate = cells[1].text.strip()
            horse['gate_number'] = int(gate) if gate.isdigit() else None
            
            num = cells[2].text.strip()
            horse['horse_number'] = int(num) if num.isdigit() else None
            
            # 馬名・馬ID
            horse_link = cells[3].find('a')
            if horse_link:
                horse['horse_name'] = horse_link.text.strip()
                href = horse_link.get('href', '')
                match = re.search(r'/horse/(\d+)/', href)
                horse['horse_id'] = match.group(1) if match else None
            
            # 性齢
            sex_age = cells[4].text.strip()
            if sex_age:
                horse['sex'] = sex_age[0] if sex_age else ''
                horse['age'] = int(sex_age[1:]) if len(sex_age) > 1 and sex_age[1:].isdigit() else None
            
            # 斤量
            try:
                horse['impost'] = float(cells[5].text.strip())
            except ValueError:
                horse['impost'] = None
            
            # 騎手
            jockey_link = cells[6].find('a')
            if jockey_link:
                horse['jockey_name'] = jockey_link.text.strip()
                href = jockey_link.get('href', '')
                match = re.search(r'/jockey/result/recent/(\d+)/', href)
                horse['jockey_id'] = match.group(1) if match else None
            
            # タイム
            time_str = cells[7].text.strip()
            horse['time_str'] = time_str
            horse['time_seconds'] = parse_time_to_seconds(time_str)
            
            # 着差
            horse['margin'] = cells[8].text.strip()
            
            # diary_snap_cut 内の追加データを探す
            # 通過順、上がり3F はセル内の diary_snap_cut タグの中にある
            snap_cut = row.find('diary_snap_cut')
            if snap_cut:
                snap_cells = snap_cut.find_all('td')
                for sc in snap_cells:
                    text = sc.text.strip()
                    # 通過順（例: "1-1", "3-3-2"）
                    if '-' in text and all(p.isdigit() for p in text.split('-')):
                        horse['passing_order'] = text
                        passing_list = parse_passing_order(text)
                        if passing_list:
                            horse['first_corner'] = passing_list[0] if len(passing_list) > 0 else None
                            horse['last_corner'] = passing_list[-1] if len(passing_list) > 0 else None
                    # 上がり3F（例: "33.9"）
                    elif re.match(r'^\d+\.\d+$', text):
                        try:
                            horse['last_3f'] = float(text)
                        except:
                            pass
            
            # オッズ・人気（diary_snap_cut外のセルから）
            # セルの順序が異なる可能性があるため、クラス名で判定
            for cell in cells:
                text = cell.text.strip()
                classes = cell.get('class', [])
                
                # 単勝オッズ
                if 'txt_r' in classes and re.match(r'^\d+\.\d+$', text):
                    try:
                        if 'odds' not in horse:
                            horse['odds'] = float(text)
                    except:
                        pass
                
                # 人気
                if any('ml' in c for c in classes):
                    span = cell.find('span')
                    if span and span.text.strip().isdigit():
                        pop = int(span.text.strip())
                        if 1 <= pop <= 18:
                            horse['popularity'] = pop
            
            # 馬体重（例: "462(-2)"）
            for cell in cells:
                text = cell.text.strip()
                if re.match(r'^\d+\([\+\-]?\d+\)$', text):
                    weight, change = parse_horse_weight(text)
                    horse['horse_weight'] = weight
                    horse['weight_change'] = change
                    break
            
            # 調教師
            trainer_cell = None
            for cell in cells:
                if '[東]' in cell.text or '[西]' in cell.text:
                    trainer_cell = cell
                    break
            
            if trainer_cell:
                trainer_link = trainer_cell.find('a')
                if trainer_link:
                    horse['trainer_name'] = trainer_link.text.strip()
                    href = trainer_link.get('href', '')
                    match = re.search(r'/trainer/result/recent/(\d+)/', href)
                    horse['trainer_id'] = match.group(1) if match else None
                
                horse['trainer_region'] = '東' if '[東]' in trainer_cell.text else '西'
            
            # 馬主
            for cell in cells:
                owner_link = cell.find('a', href=re.compile(r'/owner/'))
                if owner_link:
                    horse['owner_name'] = owner_link.text.strip()
                    href = owner_link.get('href', '')
                    match = re.search(r'/owner/result/recent/(\d+)/', href)
                    horse['owner_id'] = match.group(1) if match else None
                    break
            
            # 賞金（最後のセル）- カンマ区切りに対応
            prize_text = cells[-1].text.strip().replace(',', '')
            try:
                horse['prize_money'] = float(prize_text)
            except ValueError:
                horse['prize_money'] = 0.0
            
            horses.append(horse)
    
    # レースレベルを1着賞金から判定
    if horses:
        first_place = [h for h in horses if h.get('finish_position') == 1]
        if first_place:
            first_prize = first_place[0].get('prize_money', 0)
            race_info['first_prize'] = first_prize
            level, score = classify_race_level(first_prize)
            race_info['race_level'] = level
            race_info['level_score'] = score
        
        # 出走頭数
        race_info['horse_count'] = len(horses)
    
    payouts = parse_payouts(raw_html, race_id)
    return {'race_info': race_info, 'horses': horses, 'payouts': payouts}
