"""パース済みHTMLの台帳

「どのHTMLを、どの内容で、どのパーサ版で解析したか」を記録する。
これにより HTML を毎回パースし直さずに済む。

再パースが必要と判定されるのは次の場合。

1. マニフェストに無い（新規HTML）
2. HTML の SHA256 が変わった（再取得・差し替え・破損）
3. config.PARSER_VERSION が記録時より新しい（パーサを直した）

race_id だけで判定していると、HTML が更新されても気づけない。
ハッシュまで見るのはそのため。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from . import config

COLUMNS = ['race_id', 'year_dir', 'html_sha256', 'html_bytes',
           'parser_version', 'parsed_at', 'n_horses']


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or config.MANIFEST_PATH
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS).astype({'parser_version': 'int64',
                                                     'html_bytes': 'int64',
                                                     'n_horses': 'int64'})
    return pd.read_parquet(path)


def save(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    path = path or config.MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values('race_id').to_parquet(path, index=False)


@dataclass
class ParseTask:
    """再パースが必要な1ファイル。"""
    race_id: str
    path: Path
    year_dir: str
    html_sha256: str
    html_bytes: int
    reason: str   # 'new' | 'changed' | 'parser_upgraded'


def iter_html_files(html_dir: Optional[Path] = None,
                    years: Optional[Iterable[int]] = None) -> List[Path]:
    html_dir = html_dir or config.HTML_DIR
    if years is None:
        dirs = sorted(d for d in html_dir.iterdir() if d.is_dir())
    else:
        dirs = [html_dir / str(y) for y in sorted(years)]
    files: List[Path] = []
    for d in dirs:
        if d.is_dir():
            files.extend(sorted(d.glob('*.html')))
    return files


def plan(html_dir: Optional[Path] = None,
         years: Optional[Iterable[int]] = None,
         manifest: Optional[pd.DataFrame] = None,
         hash_all: bool = True) -> List[ParseTask]:
    """再パースが必要なファイルを洗い出す。

    Args:
        hash_all: False にすると、サイズが一致する既知ファイルのハッシュ計算を省く。
            55,000件のハッシュ計算は1分ほどかかるため、日常の増分更新では
            False で十分（サイズが同じで中身だけ変わる改竄は想定しない）。
            パーサ改修後の全件再パースでは値に関わらず全件が対象になる。
    """
    man = load() if manifest is None else manifest
    known = {r.race_id: r for r in man.itertuples(index=False)}
    current_version = config.PARSER_VERSION

    tasks: List[ParseTask] = []
    for path in iter_html_files(html_dir, years):
        race_id = path.stem
        size = path.stat().st_size
        rec = known.get(race_id)

        if rec is None:
            tasks.append(ParseTask(race_id, path, path.parent.name,
                                   file_sha256(path), size, 'new'))
            continue

        if int(rec.parser_version) < current_version:
            tasks.append(ParseTask(race_id, path, path.parent.name,
                                   file_sha256(path), size, 'parser_upgraded'))
            continue

        if int(rec.html_bytes) != size:
            tasks.append(ParseTask(race_id, path, path.parent.name,
                                   file_sha256(path), size, 'changed'))
            continue

        if hash_all:
            digest = file_sha256(path)
            if digest != rec.html_sha256:
                tasks.append(ParseTask(race_id, path, path.parent.name,
                                       digest, size, 'changed'))
    return tasks


def upsert(manifest: pd.DataFrame, rows: List[dict]) -> pd.DataFrame:
    """パース結果でマニフェストを更新する（race_id で置換）。"""
    if not rows:
        return manifest
    new = pd.DataFrame(rows, columns=COLUMNS)
    if manifest.empty:
        return new
    kept = manifest[~manifest['race_id'].isin(new['race_id'])]
    return pd.concat([kept, new], ignore_index=True)
