"""パイプラインの連続実行

    from keiba import pipeline
    pipeline.run(1, 5)                       # 取り込み〜今週末の買い目まで通し
    pipeline.run(2, 3)                       # パースと学習だけ
    pipeline.run(1, 1, end=date(2026, 9, 27))  # 取り込みだけ、期間指定

途中の段が失敗したらそこで止める。前の段の出力が次の段の入力になるため、
失敗を無視して進めると壊れたデータで学習してしまう。
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Dict, List, Optional, Union

from . import stages

StageRef = Union[int, str]


def _resolve(ref: StageRef) -> str:
    """1 / '1' / 'collect' のいずれでも受ける。"""
    if isinstance(ref, int):
        if not 1 <= ref <= len(stages.STAGES):
            raise ValueError(f'ステージ番号は 1〜{len(stages.STAGES)} です: {ref}')
        return stages.STAGES[ref - 1]
    s = str(ref).strip()
    if s.isdigit():
        return _resolve(int(s))
    if s not in stages.STAGES:
        raise ValueError(f'不明なステージ: {ref}（{stages.STAGES} のいずれか）')
    return s


def run(start_stage: StageRef = 1, end_stage: StageRef = 5,
        push: bool = True,
        collect_start: Optional[dt.date] = None,
        collect_end: Optional[dt.date] = None,
        dry_run: bool = False,
        lookback_days: Optional[int] = None,
        within_minutes: Optional[int] = None,
        workers: Optional[int] = None) -> Dict[str, Dict]:
    """指定した範囲のステージを順に実行する。

    Args:
        start_stage / end_stage: 1〜4、または 'collect' 'build' 'train' 'strategy'
        push: 各段の成果物を GitHub に保存するか
        collect_start / collect_end: 取り込みの期間。None なら自動
        dry_run: 取り込みで検知だけ行い、ダウンロードしない
        lookback_days: 取り込みの開始日を自動決定するとき、手持ちの最新日から
            さかのぼる日数。netkeiba の反映遅れで取りこぼすのを防ぐ。
        within_minutes: 予想で、発走までこの分数以内のレースだけを対象にする。
        workers: パースの並列数

    Returns:
        ステージ名 -> 結果 の辞書。失敗した段があればそこで止まる。
    """
    first, last = _resolve(start_stage), _resolve(end_stage)
    i, j = stages.STAGES.index(first), stages.STAGES.index(last)
    if i > j:
        raise ValueError(f'開始が終了より後です: {first} → {last}')

    plan = stages.STAGES[i:j + 1]
    print('実行するステージ:')
    for name in plan:
        print(f'  {stages.STAGE_LABELS[name]}')

    results: Dict[str, Dict] = {}
    t0 = time.time()

    for name in plan:
        t = time.time()
        if name == 'collect':
            kw = {} if lookback_days is None else {'lookback_days': lookback_days}
            r = stages.collect(collect_start, collect_end, push=push,
                               dry_run=dry_run, **kw)
        elif name == 'build':
            r = stages.build(push=push, workers=workers)
        elif name == 'train':
            r = stages.train(push=push)
        elif name == 'strategy':
            r = stages.strategy(push=push)
        else:
            r = stages.predict(push=push, within_minutes=within_minutes)
        r['elapsed_sec'] = round(time.time() - t, 1)
        results[name] = r

        if not r.get('ok', True):
            print(f"\n❌ {stages.STAGE_LABELS[name]} で中断しました"
                  f"（理由: {r.get('reason', '不明')}）")
            print('   後続のステージは実行していません。')
            break

    print(f'\n{"=" * 60}')
    print('まとめ')
    print('=' * 60)
    for name in plan:
        r = results.get(name)
        if r is None:
            print(f'  {stages.STAGE_LABELS[name]}: 未実行')
        else:
            mark = '✅' if r.get('ok', True) else '❌'
            print(f"  {mark} {stages.STAGE_LABELS[name]}  {r['elapsed_sec']}秒")
    print(f'  合計 {time.time() - t0:.0f} 秒')
    return results
