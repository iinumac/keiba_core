"""ウェアハウスを構築する。

    python3 scripts_build.py [--years 2025 2026] [--workers 8] [--fast]

--fast は既知ファイルのハッシュ計算を省く（日常の増分更新向け）。
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--fast', action='store_true', help='既知ファイルのハッシュ計算を省く')
    args = ap.parse_args()

    from keiba import build

    t0 = time.time()

    def prog(done, total, phase):
        if phase == 'parse' and done % 5000 == 0:
            el = time.time() - t0
            rate = done / el if el else 0
            rest = (total - done) / rate / 60 if rate else 0
            print(f'  {done:,}/{total:,}  {el:.0f}s  {rate:.0f}件/秒  残り約{rest:.1f}分', flush=True)
        elif phase == 'plan':
            print('  対象を洗い出し中...', flush=True)

    r = build.build(years=args.years, workers=args.workers,
                    hash_all=not args.fast, progress=prog)

    print('\n=== 完了 ===')
    print(f"  パース: {r['parsed']:,} 件  (内訳: {r['reasons']})")
    print(f"  races : {r['races']:,}")
    print(f"  results: {r['results']:,}")
    print(f"  失敗  : {r['failed']:,}")
    print(f"  更新した年: {r['partitions']}")
    print(f"  所要: {time.time() - t0:.0f} 秒")


if __name__ == '__main__':
    main()
