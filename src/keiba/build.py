"""HTML → ウェアハウス の構築

manifest が「再パースが必要なHTML」を決め、parse が解析し、store が
年パーティションの Parquet に取り込む、という流れをまとめたもの。

同じHTMLを二度パースしないのがここの主眼。初回だけ全件を処理し、
以降は新しく増えたHTML（と、パーサを直したときの全件）だけが対象になる。

【重要】並列実行するスクリプトから build() を呼ぶときは、必ず
`if __name__ == "__main__":` で囲むこと。

macOS の multiprocessing は spawn 方式で、子プロセスが親スクリプトを
再 import する。ガードが無いと子プロセスが build() を再実行し、
プロセスが再帰的に増え続ける。Colab（Linux）は fork なので起きないが、
ローカル実行で必ず踏む。逐次実行したいときは workers=1 を渡す。

    if __name__ == '__main__':
        build.build()
"""

from __future__ import annotations

import datetime as _dt
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from . import config, manifest, store
from .parse import parse_race_html_full


def _parse_one(path_str: str) -> Tuple[str, Optional[dict], Optional[str]]:
    """ワーカープロセス側の処理。

    プロセス間で受け渡すのは辞書だけ。BeautifulSoup のオブジェクトは持ち出さない。
    例外はここで握って呼び出し側に文字列で返す。1件の失敗で全体を止めないため。
    """
    try:
        parsed = parse_race_html_full(Path(path_str))
        return path_str, parsed, None
    except Exception as e:  # noqa: BLE001
        return path_str, None, f'{type(e).__name__}: {e}'


def default_workers() -> int:
    """並列数。CPUを使い切らず2コア残す。"""
    return max(1, (os.cpu_count() or 2) - 2)


def validate(parsed: dict) -> Optional[str]:
    """パース結果が使えるものかを判定する。問題があれば理由を返す。

    netkeiba は、開催中止などで結果が存在しないレースにも 200 で
    「枠だけのページ」を返すことがある。これをそのまま取り込むと、
    日付が 1970-01-01（エポック）、レース名も距離も空の行ができる。

    実際、旧プロジェクトの races.parquet には同じ行が 69 件混入していた。
    ここで弾き、ウェアハウスには入れない。
    """
    info = parsed.get('race_info') or {}
    horses = parsed.get('horses') or []
    if not horses:
        return 'no_horses'
    if not info.get('race_name'):
        return 'no_race_name'
    if info.get('distance') in (None, 0):
        return 'no_distance'
    return None


def _flush(tasks_done: List[dict], races: List[dict], horses: List[dict],
           warehouse: Optional[Path]) -> dict:
    if not races:
        return {'races': 0, 'results': 0, 'partitions': []}
    return store.upsert(pd.DataFrame(races), pd.DataFrame(horses), warehouse)


def build(years: Optional[Iterable[int]] = None,
          html_dir: Optional[Path] = None,
          warehouse: Optional[Path] = None,
          hash_all: bool = True,
          batch_size: int = 5000,
          workers: Optional[int] = None,
          progress=None) -> dict:
    """再パースが必要なHTMLだけを処理してウェアハウスを更新する。

    Args:
        years: 対象年。None で全年。
        hash_all: manifest.plan を参照。日常の増分更新では False で足りる。
        batch_size: 何件ごとに Parquet へ書き出すか。メモリと書き込み回数の兼ね合い。
        workers: パースの並列数。None で自動（CPU数-2）。1 で逐次実行。
        progress: (done, total, phase) を受け取るコールバック。

    Returns:
        処理結果のサマリ。
    """
    html_dir = html_dir or config.HTML_DIR
    warehouse = warehouse or config.WAREHOUSE_DIR

    if progress:
        progress(0, 0, 'plan')
    man = manifest.load()
    tasks = manifest.plan(html_dir, years, man, hash_all=hash_all)

    reasons = {}
    for t in tasks:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1

    if not tasks:
        return {'parsed': 0, 'races': 0, 'results': 0, 'reasons': reasons,
                'partitions': [], 'failed': 0, 'invalid': 0, 'invalid_samples': []}

    now = _dt.datetime.now().isoformat(timespec='seconds')
    total = len(tasks)
    races: List[dict] = []
    horses: List[dict] = []
    man_rows: List[dict] = []
    touched = set()
    n_races = n_results = failed = 0
    invalid: List[tuple] = []

    by_path = {str(t.path): t for t in tasks}
    workers = default_workers() if workers is None else workers

    def absorb(path_str, parsed, err, i):
        nonlocal failed
        if err is not None or parsed is None:
            failed += 1
            return
        task = by_path[path_str]

        reason = validate(parsed)
        if reason is not None:
            # ウェアハウスには入れない。ただしマニフェストには記録し、
            # 毎回パースし直さないようにする（n_horses=0 が目印）。
            invalid.append((task.race_id, reason))
            man_rows.append({
                'race_id': task.race_id, 'year_dir': task.year_dir,
                'html_sha256': task.html_sha256, 'html_bytes': task.html_bytes,
                'parser_version': config.PARSER_VERSION, 'parsed_at': now,
                'n_horses': 0,
            })
            return

        races.append(parsed['race_info'])
        horses.extend(parsed['horses'])
        man_rows.append({
            'race_id': task.race_id,
            'year_dir': task.year_dir,
            'html_sha256': task.html_sha256,
            'html_bytes': task.html_bytes,
            'parser_version': config.PARSER_VERSION,
            'parsed_at': now,
            'n_horses': len(parsed['horses']),
        })

    if workers <= 1:
        results_iter = (_parse_one(str(t.path)) for t in tasks)
        for i, (path_str, parsed, err) in enumerate(results_iter, 1):
            absorb(path_str, parsed, err, i)
            if len(races) >= batch_size:
                s = _flush(man_rows, races, horses, warehouse)
                n_races += s['races']; n_results += s['results']; touched |= set(s['partitions'])
                races, horses = [], []
            if progress and i % 500 == 0:
                progress(i, total, 'parse')
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            paths = [str(t.path) for t in tasks]
            for i, (path_str, parsed, err) in enumerate(
                    ex.map(_parse_one, paths, chunksize=64), 1):
                absorb(path_str, parsed, err, i)
                if len(races) >= batch_size:
                    s = _flush(man_rows, races, horses, warehouse)
                    n_races += s['races']; n_results += s['results']; touched |= set(s['partitions'])
                    races, horses = [], []
                if progress and i % 500 == 0:
                    progress(i, total, 'parse')

    if races:
        s = _flush(man_rows, races, horses, warehouse)
        n_races += s['races']; n_results += s['results']; touched |= set(s['partitions'])

    manifest.save(manifest.upsert(man, man_rows))
    if progress:
        progress(total, total, 'done')

    return {'parsed': len(man_rows), 'races': n_races, 'results': n_results,
            'reasons': reasons, 'partitions': sorted(touched), 'failed': failed,
            'invalid': len(invalid), 'invalid_samples': invalid[:10]}
