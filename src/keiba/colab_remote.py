"""netkeiba からの取得を Colab に頼む（手元の端末から使う）。

手元の端末から netkeiba に直接アクセスすると、量によっては手元のIPアドレスが
制限される（2026-10-06 に起きた）。そこで取得だけを Colab CLI の VM で行い、
結果をダウンロードして手元で計算する。ブラウザのノートブックは最初から
Colab 上で動くので、これを使う必要はない。

    from keiba import colab_remote as cr
    cards = cr.fetch_cards(dt.date(2026, 10, 10), venues={'東京'})   # 出馬表とオッズ
    cr.collect()                                                     # 新しいレースのHTML
    cr.stop()                                                        # 使い終わったら止める

Colab でも、量が多ければ同じように制限される（約2時間連続で取ったら拒否された）。
間隔は手元と同じ1秒に1回。まとめて取るときは数日に分けること。

前提: `colab`（google-colab-cli）がインストール済みで、ログイン済みであること。
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

from . import config

SESSION = 'keiba-net'
REMOTE_ROOT = '/content/keiba_core'
_UPLOADED: set = set()


def _colab(*args: str, timeout: int = 1800, stdin: Optional[str] = None) -> str:
    exe = shutil.which('colab') or str(Path.home() / '.local' / 'bin' / 'colab')
    p = subprocess.run([exe, *args], input=stdin, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f'colab {" ".join(args)} が失敗: {p.stderr.strip()[-500:]}')
    return p.stdout


def _session_alive() -> bool:
    try:
        return SESSION in _colab('sessions', timeout=120)
    except RuntimeError:
        return False


def _upload_tree(paths: Iterable[Path], name: str) -> None:
    """プロジェクトの一部を tar にまとめて送り、REMOTE_ROOT に展開する。"""
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / f'{name}.tgz'
        with tarfile.open(tgz, 'w:gz') as tar:
            for p in paths:
                if p.exists():
                    tar.add(p, arcname=str(p.relative_to(config.PROJECT_ROOT)))
        _colab('upload', '-s', SESSION, str(tgz), f'/content/{name}.tgz', timeout=600)
    run_python(f"import subprocess; subprocess.run('mkdir -p {REMOTE_ROOT} && "
               f"tar xzf /content/{name}.tgz -C {REMOTE_ROOT}', shell=True, check=True)")


def ensure_session() -> None:
    """セッションを用意し、コード（src）を送る。"""
    global _UPLOADED
    if not _session_alive():
        _colab('new', '-s', SESSION, timeout=600)
        _UPLOADED = set()
    if 'src' not in _UPLOADED:
        _upload_tree([config.PROJECT_ROOT / 'src'], 'src')
        _UPLOADED.add('src')


def run_python(code: str, timeout: int = 7200) -> str:
    """VM の上で Python を実行し、標準出力を返す。作業ディレクトリは REMOTE_ROOT。"""
    prelude = (f"import os, sys; os.makedirs('{REMOTE_ROOT}', exist_ok=True); os.chdir('{REMOTE_ROOT}'); "
               f"sys.path.insert(0, '{REMOTE_ROOT}/src')\n")
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(prelude + code)
        path = f.name
    try:
        out = _colab('exec', '-s', SESSION, '-f', path, '--timeout', str(timeout), timeout=timeout + 120)
    finally:
        os.unlink(path)
    return '\n'.join(line for line in out.splitlines() if not line.startswith('[colab]'))


def download(remote: str, local: Path) -> Path:
    local.parent.mkdir(parents=True, exist_ok=True)
    _colab('download', '-s', SESSION, remote, str(local), timeout=1200)
    return local


def stop() -> None:
    """セッションを止める（止めないと compute units を消費し続ける）。"""
    global _UPLOADED
    if _session_alive():
        _colab('stop', '-s', SESSION, timeout=300)
    _UPLOADED = set()


def fetch_cards(day: Optional[dt.date] = None, race_ids: Optional[List[str]] = None,
                venues: Optional[set] = None, with_odds: bool = True) -> List[dict]:
    """出馬表（オッズ・複勝オッズ入り）を Colab で取ってくる。

    day を渡すとその日の中央の全レース、race_ids を渡すとそのレースだけ。
    venues（例: {'東京', '京都'}）で競馬場を絞れる。
    """
    ensure_session()
    day = day or dt.date.today()
    code = f"""
import datetime as dt, json
from keiba import fetch, shutuba
f = fetch.Fetcher()
day = dt.date.fromisoformat('{day.isoformat()}')
ids = {json.dumps(race_ids)} or shutuba.race_ids_on(day, f)
venues = set({json.dumps(sorted(venues or []), ensure_ascii=False)})
cards = []
for rid in ids:
    c = shutuba.fetch_race_card(rid, f, with_odds={with_odds})
    if c and (not venues or c['venue_name'] in venues):
        c['date'] = c.get('date') or day.isoformat()
        cards.append(c)
json.dump(cards, open('/content/cards.json', 'w'), ensure_ascii=False)
print(len(cards), 'レース')
"""
    print(run_python(code).strip())
    with tempfile.TemporaryDirectory() as tmp:
        p = download('/content/cards.json', Path(tmp) / 'cards.json')
        return json.loads(p.read_text())


def collect(start: Optional[dt.date] = None, end: Optional[dt.date] = None) -> int:
    """新しいレースのHTMLを Colab で取り、手元の data/html に置く。保存した件数を返す。

    取得済みかどうかは、手元のウェアハウスのレース一覧とマニフェストで判定する
    （それだけを送る）。取り込んだ後のパースは手元で stages.build() を回す。
    """
    ensure_session()
    _upload_tree([config.WAREHOUSE_DIR / 'races', config.MANIFEST_PATH], 'known')
    args = ', '.join(f"{k}=dt.date.fromisoformat('{v.isoformat()}')" for k, v in (('start', start), ('end', end)) if v)
    code = f"""
import datetime as dt, subprocess
from pathlib import Path
from keiba import stages, config
before = set(p.name for p in config.HTML_DIR.rglob('*.html'))
r = stages.collect({args + ', ' if args else ''}push=False)
new = [p for p in config.HTML_DIR.rglob('*.html') if p.name not in before]
with open('/content/new_html.txt', 'w') as fh:
    fh.write('\\n'.join(str(p.relative_to(config.PROJECT_ROOT)) for p in new))
subprocess.run('cd {REMOTE_ROOT} && tar czf /content/new_html.tgz -T /content/new_html.txt', shell=True, check=True)
print('saved', len(new))
"""
    out = run_python(code)
    print(out.strip())
    n = int(out.strip().split()[-1]) if out.strip().split()[-1].isdigit() else 0
    if n:
        with tempfile.TemporaryDirectory() as tmp:
            p = download('/content/new_html.tgz', Path(tmp) / 'new_html.tgz')
            with tarfile.open(p) as tar:
                tar.extractall(config.PROJECT_ROOT)
    return n
