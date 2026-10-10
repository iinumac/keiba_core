"""replay_day.py で作った日ごとの評価をまとめ、目線・ピックアップ・要注意の成績を出す。

    python scripts/summarize_replays.py

reports/data/*_views.csv をすべて読み、日ごとと合計の成績を reports/SUMMARY.md に書く。
手順と注意は docs/EVALUATION.md。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd

from keiba import flags, views

files = sorted((ROOT / 'reports' / 'data').glob('*_views.csv'))
if not files:
    sys.exit('reports/data/*_views.csv がありません。先に scripts/replay_day.py を回してください')
a = pd.concat([pd.read_csv(f, dtype={'race_id': str}).assign(日付=f.name[:10]) for f in files])
a['3着内'] = a['着'] <= 3
a['6着以下'] = a['着'] >= 6


def top1(g: pd.DataFrame, col: str, by_rank: bool) -> str:
    hit = n = 0
    for _, r in g.groupby('race_id'):
        if by_rank:
            r = r[r[col].notna()]
            if r.empty:
                continue
            pick = r.loc[r[col].idxmin()]
        else:
            pick = r.loc[r[col].idxmax()] if col != '人気' else r.loc[r['人気'].idxmin()]
        n += 1; hit += bool(pick['3着内'])
    return f'{hit}/{n}（{hit / n * 100:.0f}%）' if n else '—'


def rate(g: pd.DataFrame, mask, col: str) -> str:
    x = g.loc[mask, col]
    return f'{x.mean() * 100:.1f}%（{len(x)}頭）' if len(x) else '—'


def block(g: pd.DataFrame, title: str) -> list:
    lo, fav = g['人気'] >= 4, g['人気'] <= 3
    p5 = g['人気'] <= flags.CAUTION_POP
    pk, wn = g['ピックアップ'].astype(bool), g['要注意'].astype(bool)
    L = [f'## {title}（{g["race_id"].nunique()}レース）', '',
         '| 目線 | 1番手が3着内 |', '|---|---|']
    for v in views.VIEWS:
        L.append(f'| {v} | {top1(g, f"{v}_順", True)} |')
    L.append(f'| 1番人気 | {top1(g, "人気", False)} |')
    L.append(f'| 補正スコア | {top1(g, "補正スコア", False)} |')
    L += ['', '| | 該当 | 該当しない |', '|---|---|---|',
          f"| ピックアップ（人気4番以下）の3着内率 | {rate(g, lo & pk, '3着内')} | {rate(g, lo & ~pk, '3着内')} |",
          f"| ピックアップ（1〜3番人気）の3着内率 | {rate(g, fav & pk, '3着内')} | {rate(g, fav & ~pk, '3着内')} |",
          f"| 要注意（{flags.CAUTION_POP}番人気以内）の6着以下率 | {rate(g, wn, '6着以下')} | {rate(g, p5 & ~wn, '6着以下')} |",
          '']
    return L


L = ['# 再現評価のまとめ', '',
     f'scripts/replay_day.py で作った {len(files)} 日分（{files[0].name[:10]}〜{files[-1].name[:10]}）。'
     'どの日も、その日より前のデータだけで計算し、オッズは確定オッズ。', '',
     '凡走点ごとの6着以下率（5番人気以内）と、ピックアップ点ごとの3着内率（人気4番以下）も最後に付ける。', '']
L += block(a, '合計')
for d, g in a.groupby('日付'):
    L += block(g, d)

L += ['## 点数ごとの成績（合計）', '', '| 凡走点（5番人気以内） | 頭数 | 6着以下率 |', '|---|---|---|']
p5 = a[a['人気'] <= flags.CAUTION_POP]
for lo_, hi_ in ((0, .5), (.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 99)):
    g = p5[(p5['凡走点'] >= lo_ - 1e-9) & (p5['凡走点'] < hi_ - 1e-9)]
    L.append(f"| {lo_}〜{'' if hi_ == 99 else hi_} | {len(g)} | {g['6着以下'].mean() * 100:.1f}% |" if len(g) else
             f"| {lo_}〜{'' if hi_ == 99 else hi_} | 0 | — |")
L += ['', '| ピックアップ点（人気4番以下） | 頭数 | 3着内率 |', '|---|---|---|']
lo = a[a['人気'] >= 4]
for v, g in lo.groupby(lo['ピックアップ点'].round(1)):
    L.append(f"| {v:.1f} | {len(g)} | {g['3着内'].mean() * 100:.1f}% |")

out = ROOT / 'reports' / 'SUMMARY.md'
out.write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L))
print('保存:', out)
