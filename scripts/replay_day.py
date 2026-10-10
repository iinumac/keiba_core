"""過去の1日を「その日の朝に予想した」として目線の評価を出し、実際の着順と並べる。

    python scripts/replay_day.py 2026-10-04 [--cards 出馬表.json]

その日より前のデータだけを使う（ウェアハウスを日付で切る）。オッズは確定オッズ。

- その日の結果がウェアハウスに無ければ、Colab 経由で取って取り込む（stages.collect / build）
- 出馬表は reports/data/<日付>_cards.json に控える。控えがあれば netkeiba には行かない
  （--cards で別のファイルも渡せる）。無ければ Colab 経由で取る（keiba.colab_remote）
- 出力は reports/<日付>_views.md（人が読む表）と reports/data/<日付>_views.csv（集計用）。
  何日分かまとめた成績は scripts/summarize_replays.py で出す

手順と注意は docs/EVALUATION.md。
"""

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from keiba import flags, store, views

args = sys.argv[1:]
if not args:
    sys.exit(__doc__)
day = dt.date.fromisoformat(args[0])
CUT = pd.Timestamp(day)
DATA = ROOT / 'reports' / 'data'
DATA.mkdir(parents=True, exist_ok=True)
_rt, _lf = store.read_table, store.load_for_features


def races_on_day():
    r = _rt('races')
    return sorted(r.loc[pd.to_datetime(r['date']).dt.date == day, 'race_id'].astype(str))


race_ids = races_on_day()
if not race_ids:
    # 結果がまだ無い。Colab で HTML を取り、手元でパースしてウェアハウスに入れる
    from keiba import colab_remote, stages
    print(f'{day} の結果がウェアハウスに無いので、Colab 経由で取り込みます', flush=True)
    try:
        colab_remote.collect(day, day)
    finally:
        colab_remote.stop()
    stages.build(push=False)
    race_ids = races_on_day()
    if not race_ids:
        sys.exit(f'{day} は中央競馬の開催が無いか、結果がまだ出ていません')
results = _rt('results')
results = results[results['race_id'].astype(str).isin(race_ids)]
fin = {(str(r.race_id), int(r.horse_number)): pd.to_numeric(r.finish_position, errors='coerce')
       for r in results.itertuples()}

cards_path = Path(args[args.index('--cards') + 1]) if '--cards' in args else DATA / f'{day.isoformat()}_cards.json'
if cards_path.exists():
    cards = json.load(open(cards_path))
else:
    from keiba import colab_remote
    try:
        cards = colab_remote.fetch_cards(day, race_ids=race_ids)
    finally:
        colab_remote.stop()
    cards_path.write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding='utf-8')


# ここから先は、その日より前のデータだけが見えるようにする
def read_table(table, *a, **k):
    df = _rt(table, *a, **k)
    if table in ('races', 'results', 'payouts', 'odds'):
        races = _rt('races', *a, **k)
        keep = set(races.loc[pd.to_datetime(races['date']) < CUT, 'race_id'].astype(str))
        df = df[df['race_id'].astype(str).isin(keep)]
    return df


def load_for_features(*a, **k):
    races, res = _lf(*a, **k)
    races = races[pd.to_datetime(races['date']) < CUT]
    return races, res[res['race_id'].astype(str).isin(set(races['race_id'].astype(str)))]


store.read_table, store.load_for_features = read_table, load_for_features

stats = {v: [0, 0] for v in views.VIEWS + ['1番人気', '補正スコア']}
body, rows = [], []
for c in sorted(cards, key=lambda c: c['race_id']):
    c['date'] = day.isoformat()
    t = views.race_views(c)
    rid = str(c['race_id'])
    if t.empty:
        continue
    print(rid, len(t), flush=True)
    t['着'] = [fin.get((rid, n)) for n in t.index]
    t['_ピックアップ'] = t.index.isin(views.picked(t).index)
    t['_要注意'] = t.index.isin(views.cautioned(t).index)
    t['race_id'] = rid
    t['レース'] = f"{c['venue_name']}{c['race_num']}R {c.get('race_name') or ''}"
    top3 = t['着'] <= 3
    for v in views.VIEWS:
        if f'{v}_順' in t and t[v].notna().any():
            n1 = t[f'{v}_順'].idxmin(); stats[v][0] += 1; stats[v][1] += bool(top3[n1])
    for k, n1 in (('1番人気', t['人気'].idxmin()), ('補正スコア', t['補正スコア'].idxmax())):
        stats[k][0] += 1; stats[k][1] += bool(top3[n1])
    rows.append(t)

    lines = views.to_markdown(c, t).split('\n')
    for i, s in enumerate(lines):
        cells = s.split('|')
        if s.startswith('| 馬番 |'):
            lines[i] = s + ' 着 |'; lines[i + 1] += '---|'
        elif s.startswith('| ') and len(cells) >= 13 and cells[1].strip().isdigit():
            f = t.loc[int(cells[1]), '着']
            lines[i] = s + (f' **{int(f)}** |' if f <= 3 else f' {int(f)} |' if pd.notna(f) else ' 取消 |')
        elif s.startswith('- ') and s[2:].split()[0].isdigit():
            f = t.loc[int(s[2:].split()[0]), '着']
            if pd.notna(f):
                lines[i] = s + (f'　→ **{int(f)}着**' if f <= 3 else f'　→ {int(f)}着')
    r3 = t[top3].sort_values('着')
    lines.insert(1, '\n結果: ' + ' / '.join(f"{int(r['着'])}着 {n} {r['馬名']}（{int(r['人気'])}人気）"
                                            for n, r in r3.iterrows()))
    body.append('\n'.join(lines))

a = pd.concat(rows)
a = a[a['着'].notna()]
pk = a['_ピックアップ']
wn = a['_要注意']
lo, p5 = a['人気'] >= 4, a['人気'] <= flags.CAUTION_POP


def rate(mask, col):
    return f"{(a.loc[mask, '着'] <= 3 if col == 'top3' else a.loc[mask, '着'] >= 6).mean() * 100:.1f}%（{mask.sum()}頭）"


S = [f'## まとめ（{len(rows)}レース）', '', '| 目線 | 1番手が3着内 |', '|---|---|']
S += [f'| {v} | {h}/{n}（{h / n * 100:.0f}%） |' for v, (n, h) in stats.items() if n]
S += ['', '| | 該当 | 該当しない |', '|---|---|---|',
      f"| ピックアップ（人気4番以下）の3着内率 | {rate(lo & pk, 'top3')} | {rate(lo & ~pk, 'top3')} |",
      f"| ピックアップ（1〜3番人気）の3着内率 | {rate(~lo & pk, 'top3')} | {rate(~lo & ~pk, 'top3')} |",
      f"| 要注意（{flags.CAUTION_POP}番人気以内）の6着以下率 | {rate(wn, 'flop')} | {rate(p5 & ~wn, 'flop')} |"]
CSV_COLS = (['race_id', 'レース', '馬番', '馬名', '人気', '単勝', '脚質', '着', '市場あり', '市場なし', '市場の見込み',
             '補正スコア', 'ピックアップ点', '凡走点', 'ピックアップ', '要注意']
            + [f'{v}_順' for v in views.VIEWS])
a.reset_index().rename(columns={'index': '馬番', '_ピックアップ': 'ピックアップ', '_要注意': '要注意'}) \
    .reindex(columns=CSV_COLS).to_csv(DATA / f'{day.isoformat()}_views.csv', index=False)
out = ROOT / 'reports' / f'{day.isoformat()}_views.md'
out.parent.mkdir(exist_ok=True)
out.write_text(f'# {day.isoformat()} 全{len(rows)}レース 目線の評価\n\n'
               f'{day.month}/{day.day}より前のデータだけで計算（オッズは確定オッズ）。各馬の「着」は実際の着順。\n\n'
               + '\n'.join(S) + '\n\n' + '\n\n---\n\n'.join(body) + '\n', encoding='utf-8')
print('\n'.join(S))
print('保存:', out)
