"""対戦比較（keiba.h2h）でレースの出走馬を並べ、根拠を表示する。

    python scripts/h2h_rank.py <race_id>                 # 過去のレース（結果も並べる）
    python scripts/h2h_rank.py <race_id> <馬A> <馬B>     # 2頭の比較の根拠を詳しく
    python scripts/h2h_rank.py shutuba [ファイル] [race_id]  # 出馬表（data/shutuba/*.json）

過去のレースは、そのレース当日より前のデータだけで判定する。
馬は horse_id でも馬名でも指定できる。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from keiba import config, h2h, store

LEVEL = {0: '-', 1: '直接', 2: '共通', 3: '2頭'}
pd.set_option('display.width', 200)
pd.set_option('display.unicode.east_asian_width', True)


def entries_from_shutuba(path, race_id=None):
    data = json.load(open(path))
    out = []
    for r in data:
        if race_id and r['race_id'] != race_id:
            continue
        out.append((r['race_id'], f"{r['date']} {r['venue_name']}{r['race_num']}R {r['race_name']}",
                    pd.Timestamp(r['date']),
                    {h['horse_id']: h['horse_name'] for h in r['horses']},
                    {h['horse_id']: h.get('popularity') for h in r['horses']}, None))
    return out


def entries_from_warehouse(race_id, races, results):
    race = races[races['race_id'].astype(str) == race_id]
    if race.empty:
        sys.exit(f'race_id {race_id} がウェアハウスにありません')
    race = race.iloc[0]
    rr = results[results['race_id'].astype(str) == race_id]
    rr = rr.assign(horse_id=rr['horse_id'].astype(str))
    label = f"{race['date']} {race['venue_name']}{int(race['race_num'])}R {race['race_name']}"
    return [(race_id, label, pd.Timestamp(race['date']),
             dict(zip(rr['horse_id'], rr['horse_name'])),
             dict(zip(rr['horse_id'], rr['popularity'])),
             dict(zip(rr['horse_id'], rr['finish_position'])))]


def show_race(hist, race_id, label, as_of, names, pop, fin):
    table, pw = h2h.rank(hist, names.keys(), as_of, names)
    table['人気'] = table['horse_id'].map(pop)
    if fin:
        table['着順'] = table['horse_id'].map(fin)
    table['score'] = table['score'].round(3)
    print(f'\n=== {race_id}  {label}  （{as_of.date()} より前のデータで判定）')
    print(table.drop(columns='horse_id').rename(columns={
        'rank': '序列', 'horse_name': '馬名', 'wins': '上と判定', 'known': '比較可',
        'lv1': '直接', 'lv2': '共通', 'lv3': '2頭'}).to_string(index=False))
    print('  score = 他の各馬に先着する確率の平均（比較できない相手は0.5）')


def show_pair(hist, names, as_of, a, b, all_names):
    e = h2h.estimate(hist, a, b, as_of)
    na, nb = names.get(a, a), names.get(b, b)
    print(f'\n--- {na} vs {nb}')
    print(f'  推定: {na} が {e.margin:+.2f} 秒  段階 {LEVEL[e.level]}  根拠 {e.n}  '
          f'先着確率 {h2h.win_prob(e):.1%}')
    if e.level == 1:
        print(pd.DataFrame(hist.meetings(a, b, as_of)).to_string(index=False))
    elif e.level == 2:
        oa, ob = hist.opponents(a, as_of), hist.opponents(b, as_of)
        rows = [{'共通の相手': all_names.get(hid, hid), f'{na}との差': round(oa[hid][0], 2),
                 f'{nb}との差': round(ob[hid][0], 2), '差の差': round(oa[hid][0] - ob[hid][0], 2)}
                for hid in oa.keys() & ob.keys()]
        print(pd.DataFrame(rows).sort_values('差の差').to_string(index=False))
        print('  （差 = その馬に何秒先着したか）')


def resolve(names, key):
    if key in names:
        return key
    hit = [h for h, n in names.items() if n == key]
    if not hit:
        sys.exit(f'{key} が出走馬にいません')
    return hit[0]


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    races, results = store.read_table('races'), store.read_table('results')
    hist = h2h.History(races, results)
    if args[0] == 'shutuba':
        path = args[1] if len(args) > 1 else sorted((config.DATA_DIR / 'shutuba').glob('*.json'))[-1]
        entries = entries_from_shutuba(path, args[2] if len(args) > 2 else None)
    else:
        entries = entries_from_warehouse(args[0], races, results)
    for e in entries:
        show_race(hist, *e)
    if len(args) == 3 and args[0] != 'shutuba':
        _, _, as_of, names, _, _ = entries[0]
        all_names = dict(zip(results['horse_id'].astype(str), results['horse_name']))
        show_pair(hist, names, as_of, resolve(names, args[1]), resolve(names, args[2]), all_names)
