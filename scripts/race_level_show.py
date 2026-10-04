"""出走馬の前走のレースレベルを並べる（keiba.race_level）。

    python scripts/race_level_show.py <race_id>                  # ウェアハウスにある過去のレース
    python scripts/race_level_show.py <race_id> <YYYY-MM-DD>     # 出馬表（netkeiba から取得）

前走がどのレースで、そのレースのレベルがどうで、そこで何秒負けたか。
「格上のレースで負けた馬」と「格下のレースで好走した馬」のどちらが上かを、
前走パフォーマンス（= 前走レベル − 勝ち馬との差）で比べる。

秒は今回の距離に換算して表示する（内部は1000mあたり秒）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from keiba import race_level as rl, shutuba, store
from keiba.fetch import Fetcher

pd.set_option('display.width', 220)
pd.set_option('display.unicode.east_asian_width', True)


def entries(race_id, date, races, results):
    """(レース名, 判定日, 距離, 出走馬の表)。"""
    race = races[races['race_id'].astype(str) == race_id]
    if not race.empty:
        race = race.iloc[0]
        rr = results[results['race_id'].astype(str) == race_id]
        horses = pd.DataFrame({'馬番': rr['horse_number'], 'horse_id': rr['horse_id'].astype(str),
                               '馬名': rr['horse_name'], '人気': rr['popularity'],
                               '着順': rr['finish_position']})
        return (f"{race['date']} {race['venue_name']} {race['race_name']}",
                pd.Timestamp(race['date']), int(race['distance']), horses)
    if date is None:
        sys.exit('ウェアハウスに無いレースは日付も指定してください（出馬表を取りに行きます）')
    card = shutuba.fetch_race_card(race_id, Fetcher())
    if card is None:
        sys.exit(f'{race_id} の出馬表が取れませんでした')
    horses = pd.DataFrame([{'馬番': h['horse_number'], 'horse_id': h['horse_id'],
                            '馬名': h['horse_name'], '人気': h.get('popularity')}
                           for h in card['horses']])
    return (f"{date} {card['venue_name']} {card['race_name']}", pd.Timestamp(date),
            int(card['distance']), horses)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    race_id = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) > 2 else None
    races, results = store.read_table('races'), store.read_table('results')
    label, as_of, dist, horses = entries(race_id, date, races, results)
    km = dist / 1000.0

    runs = rl.prepare_runs(races, results)
    level, ability = rl.fit(runs, as_of)
    past = runs[(runs['date'] < as_of) & runs['horse_id'].isin(horses['horse_id'])]
    prev = past.sort_values('date').groupby('horse_id').tail(1).set_index('horse_id')

    t = horses.set_index('horse_id')
    t['前走'] = prev['race_name'].str.replace(r'^第\d+回', '', regex=True)
    t['前走日'] = prev['date'].dt.strftime('%y-%m-%d')
    t['前走着'] = prev['finish_position']
    t['前走レベル'] = prev['race_id'].map(level) * km
    t['勝ち馬との差'] = prev['behind'] * km
    t['前走パフォ'] = t['前走レベル'] - t['勝ち馬との差']
    t['能力'] = ability.reindex(t.index) * km
    t = t.sort_values('前走パフォ', ascending=False)
    t.insert(0, '順', range(1, len(t) + 1))

    print(f'\n=== {race_id}  {label}  {dist}m  （{as_of.date()} より前のデータ・秒は{dist}m換算）\n')
    print(t.reset_index(drop=True).to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print('\n  前走レベル  前走の勝ち馬がどれだけ強いレースをしたか（大きいほど強いレース）')
    print('  前走パフォ  前走レベル − 勝ち馬との差。その馬が前走で見せた強さ。この順に並べている')
    print('  能力        過去2年の全レースから推定した地力（前走パフォより安定、ただし調子の変化は追えない）')

    # 前走ごとにまとめる（同じレースから来た馬が複数いるところだけ）
    g = t.dropna(subset=['前走']).groupby('前走').agg(
        頭数=('馬名', 'size'), レベル=('前走レベル', 'first'),
        着順=('前走着', lambda s: '・'.join(str(int(x)) for x in sorted(s))),
        パフォ平均=('前走パフォ', 'mean'), 最上位=('前走パフォ', 'max'))
    g = g.sort_values('レベル', ascending=False)
    stats = rl.next_run_stats(runs, prev['race_id'].unique(), as_of=as_of)
    rid = prev.drop_duplicates('race_id').assign(前走=lambda d: d['race_name'].str.replace(
        r'^第\d+回', '', regex=True)).set_index('前走')['race_id']
    g = g.join(stats.rename(index={v: k for k, v in rid.items()}), how='left')
    print('\n■ 前走のレースごと（レベルの高い順）')
    print(g.to_string(float_format=lambda v: f'{v:.2f}'))
    print('  次走済・次走勝利・次走3着内 = そのレースの出走馬のうち、今日より前に次走を走った馬の成績')
