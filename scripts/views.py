"""目線ごとの上位3頭を出す（keiba.views）。

    python scripts/views.py <race_id> [<race_id> ...]      # 出馬表を取ってきて表示
    python scripts/views.py today [東京|京都 ...]          # 今日の全レース（場を指定できる）

1回目は全履歴の計算に数分かかる。同じプロセス内の2レース目以降はすぐ出る。
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

from keiba import shutuba, views
from keiba.fetch import Fetcher

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    f = Fetcher()
    today = dt.date.today()
    if sys.argv[1] == 'today':
        venues = set(sys.argv[2:])
        ids = [r for r in shutuba.race_ids_on(today, f)]
    else:
        venues, ids = set(), sys.argv[1:]
    for rid in ids:
        card = shutuba.fetch_race_card(rid, f)
        if not card or (venues and card['venue_name'] not in venues):
            continue
        card['date'] = card.get('date') or today.isoformat()
        t = views.race_views(card)
        if t.empty:
            print(f'{rid}: オッズ未発表'); continue
        print(views.to_markdown(card, t) + '\n', flush=True)
