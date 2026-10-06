"""目線ごとの評価・フラグ・リストを出す（keiba.views）。

    python scripts/views.py <race_id> [<race_id> ...] [--date YYYY-MM-DD]   # レースを指定
    python scripts/views.py today [東京|京都 ...]                            # 今日の全レース

netkeiba の出馬表とオッズは、手元の端末からは Colab 経由で取る（keiba.colab_remote）。
Colab 上で動かしたときは直接取る。1回目は全履歴の計算に数分かかる。
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

from keiba import colab_remote, fetch, shutuba, views


def get_cards(day, race_ids, venues):
    if fetch.on_colab():
        f = fetch.Fetcher()
        out = []
        for rid in race_ids or shutuba.race_ids_on(day, f):
            c = shutuba.fetch_race_card(rid, f)
            if c and (not venues or c['venue_name'] in venues):
                c['date'] = c.get('date') or day.isoformat(); out.append(c)
        return out
    try:
        return colab_remote.fetch_cards(day, race_ids=race_ids, venues=venues)
    finally:
        colab_remote.stop()


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    day = dt.date.today()
    if '--date' in args:
        i = args.index('--date'); day = dt.date.fromisoformat(args[i + 1]); del args[i:i + 2]
    if args[0] == 'today':
        cards = get_cards(day, None, set(args[1:]))
    else:
        cards = get_cards(day, args, set())
    for card in sorted(cards, key=lambda c: (c.get('start_time') or '', c['venue_name'])):
        t = views.race_views(card)
        if t.empty:
            print(f"{card['race_id']}: オッズ未発表"); continue
        print(views.to_markdown(card, t) + '\n', flush=True)
