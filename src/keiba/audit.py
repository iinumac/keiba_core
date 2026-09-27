"""データの抜けを検出する

取りこぼしは静かに起きる。netkeiba は結果の反映が遅れることがあり、
取得に失敗しても「その日は開催が無かった」のと区別がつかない。
気づかないまま先へ進むと、欠けたまま学習してしまう。

2通りの検査を用意する。

- `local()`  : 通信なし。race_id の構造から抜けを推定する
- `online()` : netkeiba のカレンダーと突き合わせる。確実だがリクエストを使う

## local() の考え方

race_id は `YYYY` + 場コード2桁 + 開催回2桁 + 開催日2桁 + レース番号2桁。
先頭10桁が同じものが「ある競馬場のある開催日」にあたり、通常はレース番号
1〜12 が揃う（実データで 4,826 開催日のうち 4,798 が12R）。

そこで次の2つを見る。

- **内部の欠番**: 1〜最大レース番号の間が飛んでいる。**確実な抜け**
- **末尾の不足**: 最大が12未満。開催中止の可能性もあるので「要確認」扱い

日付でまとめてはいけない。開催が順延されると同じ開催日が2日にまたがり、
存在するレースを欠番と誤検出する（2011-02-14 小倉、2013-01-21 中山、
2020-03-31 中山の3件がこれに当たった）。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

RACES_PER_DAY = 12


@dataclass
class AuditReport:
    """検査結果。`gaps` が空なら抜けは見つかっていない。"""

    gaps: List[str] = field(default_factory=list)
    """内部の欠番。確実な抜けで、再取得すべき race_id。"""

    short_days: Dict[str, List[str]] = field(default_factory=dict)
    """末尾が12Rに満たない開催日 -> 不足している race_id。中止の可能性あり。"""

    warehouse_without_html: List[str] = field(default_factory=list)
    """ウェアハウスにあるのに HTML が無いレース。再構築すると失われる。"""

    html_without_warehouse: List[str] = field(default_factory=list)
    """HTML はあるが取り込まれていないレース。多くは結果の無い空ページ。"""

    n_race_days: int = 0
    n_races: int = 0

    @property
    def ok(self) -> bool:
        return not self.gaps and not self.warehouse_without_html

    def print_report(self, max_list: int = 20) -> None:
        print(f'開催日 {self.n_race_days:,} / レース {self.n_races:,}')
        print()
        mark = '✅' if not self.gaps else '⚠️'
        print(f'{mark} 内部の欠番（確実な抜け）: {len(self.gaps)} 件')
        for r in self.gaps[:max_list]:
            print(f'     {r}  ({_describe(r)})')
        if len(self.gaps) > max_list:
            print(f'     … 他 {len(self.gaps) - max_list} 件')

        n_short = sum(len(v) for v in self.short_days.values())
        print(f'ℹ️ 末尾の不足（中止の可能性）: {n_short} 件 / {len(self.short_days)} 開催日')

        mark = '✅' if not self.warehouse_without_html else '⚠️'
        print(f'{mark} HTMLが無いレース: {len(self.warehouse_without_html)} 件')
        print(f'ℹ️ 取り込まれていないHTML: {len(self.html_without_warehouse)} 件'
              f'（結果の無い空ページ）')

        print()
        if self.ok:
            print('✅ 抜けは見つかりませんでした')
        else:
            print('⚠️ 抜けがあります。stages.collect(race_ids=report.gaps) で取得できます')


VENUES = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
          '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'}


def _describe(race_id: str) -> str:
    v = VENUES.get(race_id[4:6], '?')
    return f'{race_id[:4]}年 {v} {int(race_id[6:8])}回{int(race_id[8:10])}日 {int(race_id[10:12])}R'


def local(warehouse: Optional[Path] = None) -> AuditReport:
    """通信なしで抜けを検出する。"""
    import pandas as pd
    from . import config, manifest, store

    man = manifest.load()
    attempted: Set[str] = set(man['race_id'].astype(str))
    warehoused = store.existing_race_ids(warehouse)
    known = attempted | warehoused

    rep = AuditReport(n_races=len(warehoused))
    if not known:
        return rep

    df = pd.DataFrame({'race_id': sorted(known)})
    df['prefix'] = df['race_id'].str[:10]
    df['rn'] = pd.to_numeric(df['race_id'].str[10:12], errors='coerce')
    rep.n_race_days = df['prefix'].nunique()

    for prefix, g in df.groupby('prefix'):
        nums = set(g['rn'].dropna().astype(int))
        if not nums:
            continue
        top = max(nums)
        rep.gaps += [f'{prefix}{n:02d}' for n in range(1, top) if n not in nums]
        missing_tail = [f'{prefix}{n:02d}' for n in range(top + 1, RACES_PER_DAY + 1)]
        if missing_tail:
            rep.short_days[prefix] = missing_tail

    # 「手元で使えるHTML」は Git 追跡分とディスク上の実体の和集合。
    #   - 追跡されているがディスクに無い : 部分クローン（sparse-checkout）
    #   - ディスクにあるが未追跡         : 取得したてでまだコミットしていない
    # どちらか片方だけを見ると、その分を「欠落」と誤検出する。
    available = {p.stem for p in Path(config.HTML_DIR).rglob('*.html')}
    tracked = _tracked_html()
    if tracked:
        available |= tracked

    rep.warehouse_without_html = sorted(warehoused - available)
    rep.html_without_warehouse = sorted(available - warehoused)
    return rep


def _tracked_html() -> Optional[Set[str]]:
    import subprocess
    from . import config
    try:
        out = subprocess.run(['git', 'ls-files', 'data/html'],
                             cwd=config.PROJECT_ROOT,
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    ids = {Path(p).stem for p in out.split() if p.endswith('.html')}
    return ids or None


def online(start: dt.date, end: dt.date, fetcher=None,
           on_day=None) -> Dict:
    """netkeiba のカレンダーと突き合わせて、期間内の抜けを確定させる。

    `local()` は race_id の構造からの推定なので、開催そのものを丸ごと
    取りこぼしている場合（その日のHTMLが1件も無い）は検出できない。
    こちらは「開催日として存在するのに手元に1件も無い」も拾える。

    リクエストは「月数 + 開催日数」ぶん。期間を絞って使うこと。
    """
    from . import discovery, fetch, manifest, store

    fetcher = fetcher or fetch.Fetcher()
    known = set(manifest.load()['race_id'].astype(str)) | store.existing_race_ids()

    missing: List[str] = []
    days_checked = 0
    for day in discovery.race_days(start, end, fetcher):
        result = discovery.race_ids_on(day, fetcher)
        days_checked += 1
        gap = [r for r in result.race_ids if r not in known]
        if on_day is not None:
            on_day(day, len(result.race_ids), len(gap))
        missing += gap

    return {'start': start, 'end': end, 'days_checked': days_checked,
            'missing': missing}
