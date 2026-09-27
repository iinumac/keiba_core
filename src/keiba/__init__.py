"""keiba_core — 競馬データの収集・蓄積・分析基盤

    from keiba import build, store, discovery, fetch, features

典型的な流れ:

    # 1. 新しいレースを検知して取得
    f = fetch.Fetcher()
    new_ids = discovery.discover(start, end, f, known=store.existing_race_ids())
    for rid in new_ids:
        f.download_race(rid, config.HTML_DIR)

    # 2. 増えた分だけパースしてウェアハウスへ
    build.build()

    # 3. SQL で分析
    con = store.connect()
    con.sql("SELECT venue_name, count(*) FROM race_results GROUP BY 1")
"""

from . import (config, manifest, store, build, discovery, fetch,  # noqa: F401
               features, parse, gitpush, strategy)

__all__ = ['config', 'manifest', 'store', 'build', 'discovery', 'fetch',
           'features', 'parse', 'gitpush', 'strategy']
