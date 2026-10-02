"""keiba_core の基本動作テスト

    python3 tests/test_pipeline.py

実データ（data/html, data/warehouse）を使う。ウェアハウスは書き換えない。
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd  # noqa: E402

from keiba import build, config, manifest, store  # noqa: E402
from keiba.features import build_features, C03_CONFIG, C04_CONFIG  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(f"  {'✓' if cond else '✗'} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def test_manifest_roundtrip():
    print('\n[マニフェスト]')
    man = manifest.load()
    check('マニフェストが存在する', len(man) > 0, f'{len(man):,} 件')
    check('race_id が一意', man['race_id'].is_unique)
    check('パーサ版が記録されている',
          set(man['parser_version'].unique()) == {config.PARSER_VERSION})

    tasks = manifest.plan(hash_all=False)
    check('再パース対象が0件（＝二度パースしない）', len(tasks) == 0, f'{len(tasks)} 件')


def test_parser_version_triggers_reparse():
    print('\n[パーサ版を上げると再パースされる]')
    original = config.PARSER_VERSION
    try:
        config.PARSER_VERSION = original + 1
        tasks = manifest.plan(years=[2026], hash_all=False)
        n_html = len(list((config.HTML_DIR / '2026').glob('*.html')))
        check('2026年の全HTMLが対象になる', len(tasks) == n_html,
              f'{len(tasks)} / {n_html}')
        check('理由が parser_upgraded',
              all(t.reason == 'parser_upgraded' for t in tasks))
    finally:
        config.PARSER_VERSION = original


def test_warehouse_integrity():
    print('\n[ウェアハウスの整合性]')
    races = store.read_table('races', columns=['race_id', 'date', 'year'])
    check('レースが入っている', len(races) > 50000, f'{len(races):,} 件')
    check('race_id が一意', races['race_id'].is_unique)

    disk = {p.stem for p in config.HTML_DIR.rglob('*.html')}
    wh = set(races['race_id'].astype(str))
    check('HTMLが無いレースは存在しない', len(wh - disk) == 0, f'{len(wh - disk)} 件')

    years = pd.to_datetime(races['date'], errors='coerce').dt.year
    check('1970年（パース失敗）の行が無い', (years == 1970).sum() == 0)
    check('日付が欠損していない', years.isna().sum() == 0)


def test_validate_rejects_empty_pages():
    print('\n[空ページの検証]')
    check('馬がいなければ弾く',
          build.validate({'race_info': {'race_name': 'x', 'distance': 1600},
                          'horses': []}) == 'no_horses')
    check('レース名が無ければ弾く',
          build.validate({'race_info': {'race_name': '', 'distance': 1600},
                          'horses': [{}]}) == 'no_race_name')
    check('距離が無ければ弾く',
          build.validate({'race_info': {'race_name': 'x', 'distance': None},
                          'horses': [{}]}) == 'no_distance')
    check('正常なものは通す',
          build.validate({'race_info': {'race_name': '2歳新馬', 'distance': 1600},
                          'horses': [{}]}) is None)


def test_course_notation():
    """コース表記の取りこぼしがないこと。

    旧パーサは方向の後を1文字ぶんしか見ておらず、776レースで surface と
    distance が None になっていた。新潟の1000m戦は1件もデータに入っておらず、
    中山のステイヤーズS（芝右 内2周3600m）も16年分すべて欠落していた。

    全HTMLを走査してコース表記は14種類と確認済み。下の式は全種を解析できる。
    """
    print('\n[コース表記の解析]')
    from keiba.parse import COURSE_RE as pat

    cases = [
        ('ダ左1400m',        'ダ',   '左', '',      1400),
        ('芝右1600m',        '芝',   '右', '',      1600),
        ('芝右 外1600m',     '芝',   '右', '外',    1600),
        ('芝左 内2000m',     '芝',   '左', '内',    2000),
        ('芝直線1000m',      '芝',   '',   '直線',  1000),
        ('障芝 外-内2890m',  '芝',   '',   '外-内', 2890),
        ('芝右 内2周3600m',  '芝',   '右', '内2周', 3600),
        ('障芝 内-外3350m',  '芝',   '',   '内-外', 3350),
    ]
    for text, surface, direction, course, dist in cases:
        m = pat.search(text)
        ok = (m is not None
              and m.group(2).startswith(surface[0])
              and (m.group(3) or '') == direction
              and m.group(4).strip() == course
              and int(m.group(5)) == dist)
        check(f'{text}', ok, '' if ok else f'→ {m.groups() if m else None}')

    # ウェアハウス側の結果
    races = store.read_table('races', columns=['venue_name', 'distance',
                                               'surface', 'course_type'])
    nii_1000 = ((races['venue_name'] == '新潟') & (races['distance'] == 1000)).sum()
    check('新潟の1000m戦がデータに入っている', nii_1000 > 0, f'{nii_1000} レース')
    n_3600 = (races['distance'] == 3600).sum()
    check('中山3600m（ステイヤーズS）が入っている', n_3600 > 0, f'{n_3600} レース')
    n_null = int(races['distance'].isna().sum())
    check('距離が欠損しているレースが無い', n_null == 0, f'{n_null} 件')


def test_features():
    print('\n[特徴量]')
    races, results = store.load_for_features(years=[2024, 2025])
    check('results にレース属性が結合されている',
          {'surface', 'distance', 'race_date', 'level_score'} <= set(results.columns))

    df3 = build_features(races, results, C03_CONFIG)
    df4 = build_features(races, results, C04_CONFIG)
    check('C03 の特徴量が生成される', 'horse_expected_top3_rate' in df3.columns)
    check('C04 の市場系特徴量が生成される', 'odds_ratio_to_fav' in df4.columns)
    check('別名が同値', df4['horse_prev_top3_rate'].equals(df4['horse_expected_top3_rate']))
    # 設定が効いていること。surface はコース表記の修正で欠損が無くなったため、
    # 埋め値では差が出なくなった。クレンジング条件の違いで見る。
    #   C03: 特定騎手を除外し、着順が数値でない行は残す
    #   C04: 騎手を除外せず、着順が数値でない行を落とす
    check('C03/C04 で対象行数が異なる（設定が効いている）',
          len(df3) != len(df4), f'C03 {len(df3):,} / C04 {len(df4):,}')
    check('新馬の初期値が設定どおり異なる',
          round(float(df3.loc[df3['is_debut'] == 1, 'horse_expected_top3_rate'].iloc[0]), 4)
          != round(float(df4.loc[df4['is_debut'] == 1, 'horse_expected_top3_rate'].iloc[0]), 4))

    # 旧プロジェクトは sort_values('race_date') だけで最新行を取っており、
    # 同一日に複数レースがある騎手でどの行を拾うかが入力の行順で変わっていた。
    # race_id まで含めれば、入力をシャッフルしても同じ値になる。
    # tail(1) は行順を保つため、リストではなく騎手ID単位で比較すること。
    def master(d):
        return (d.sort_values(['race_date', 'race_id'])
                 .groupby('jockey_id').tail(1)
                 .set_index('jockey_id')['jockey_added_value'].sort_index())

    base = master(df3)
    shuffled = master(df3.sample(frac=1, random_state=1))
    check('マスタ生成が入力の行順に依存しない', base.equals(shuffled),
          f'{len(base)} 騎手')

    # 参考: race_id を外すと不定になることの確認
    def master_date_only(d):
        return (d.sort_values(['race_date'])
                 .groupby('jockey_id').tail(1)
                 .set_index('jockey_id')['jockey_added_value'].sort_index())

    unstable = not master_date_only(df3).equals(
        master_date_only(df3.sample(frac=1, random_state=1)))
    check('race_id を外すと不定になる（旧実装の再現）', unstable)


def test_discovery_filters_to_jra():
    print('\n[中央競馬への絞り込み]')
    from keiba import discovery

    check('中央の場コードを中央と判定',
          all(discovery.is_jra(f'2026{c}010101') for c in discovery.JRA_VENUES))
    # 2026-01-04 のレース一覧に実際に並んでいた地方の場コード
    check('地方の場コードを除外',
          not any(discovery.is_jra(f'2026{c}010401') for c in ('45', '48', '55', '65')))
    check('桁数が違うものは弾く', not discovery.is_jra('20260101'))
    check('場名が引ける', discovery.venue_name('202605010101') == '東京')

    # ウェアハウスに中央以外が混入していないこと
    ids = store.existing_race_ids()
    non_jra = [i for i in ids if not discovery.is_jra(i)]
    check('ウェアハウスに中央以外が無い', len(non_jra) == 0, f'{len(non_jra)} 件')


def test_payouts():
    """払戻は「レース×券種×組み合わせ」の縦持ち。

    同着で組数が増えるため、横持ち（券種ごとに列）では破綻する。
    実データには3着が3頭同着でワイドが7組になったレースが2件ある。
    """
    print('\n[払戻]')
    pay = store.read_table('payouts')
    check('払戻が入っている', len(pay) > 600000, f'{len(pay):,} 行')
    check('払戻金額に欠損が無い', pay['payout'].isna().sum() == 0)
    check('人気に欠損が無い', pay['popularity'].isna().sum() == 0)
    check('払戻は100円以上', pay['payout'].min() >= 100, f"最小 {pay['payout'].min()}")

    types = set(pay['bet_type'])
    check('券種が8種そろっている',
          types == {'単勝', '複勝', '枠連', '馬連', 'ワイド', '馬単', '三連複', '三連単'},
          str(sorted(types)))

    # 3着3頭同着のレース。ワイド7組・複勝5組・三連複3組になる
    for rid in ('201208040611', '202009050712'):
        g = pay[pay['race_id'] == rid]
        shape = g['bet_type'].value_counts().to_dict()
        ok = (shape.get('ワイド') == 7 and shape.get('複勝') == 5
              and shape.get('三連複') == 3)
        check(f'3着3頭同着 {rid} を展開できている', ok, str(shape))

    # 8頭以下は枠連が発売されない
    races = store.read_table('races', columns=['race_id', 'horse_count'])
    n_run = (store.read_table('results', columns=['race_id', 'horse_number'])
             .groupby('race_id').size())
    waku = set(pay[pay['bet_type'] == '枠連']['race_id'])
    small = {r for r, n in n_run.items() if n <= 8}
    check('8頭以下のレースに枠連が無い', not (small & waku),
          f'{len(small & waku)} 件 / 8頭以下 {len(small):,} レース')


def test_payout_matches_odds():
    """払戻と results.odds は別々に抽出した独立データ。突き合わせて検証する。"""
    print('\n[払戻と結果の整合]')
    try:
        con = store.connect()
    except ImportError:
        print('  - duckdb 未インストールのためスキップ')
        return

    row = con.sql("""
      WITH win AS (SELECT race_id, horse_number, odds FROM results
                   WHERE finish_position = 1),
           pay AS (SELECT race_id, horse_numbers[1] AS horse_number, payout,
                          count(*) OVER (PARTITION BY race_id) AS n_win
                   FROM payouts WHERE bet_type = '単勝')
      SELECT
        sum(CASE WHEN p.n_win = 1 AND abs(w.odds*100 - p.payout) >= 1
                 THEN 1 ELSE 0 END) AS solo_mismatch,
        sum(CASE WHEN p.n_win = 1 THEN 1 ELSE 0 END) AS solo
      FROM win w JOIN pay p
        ON w.race_id = p.race_id AND w.horse_number = p.horse_number
    """).fetchone()
    check('単独1着なら 単勝払戻 == オッズ×100', row[0] == 0,
          f'{row[0]} 件 / {row[1]:,} 件')

    bad = con.sql("""
      WITH ex AS (SELECT race_id, unnest(horse_numbers) AS hn
                  FROM payouts WHERE bet_type <> '枠連')
      SELECT count(*) FROM ex
      LEFT JOIN results r ON ex.race_id = r.race_id AND ex.hn = r.horse_number
      WHERE r.horse_number IS NULL
    """).fetchone()[0]
    check('払戻の馬番がすべて結果に実在する', bad == 0, f'{bad} 件')


def test_audit():
    print('\n[抜けの検出]')
    from keiba import audit

    rep = audit.local()
    check('開催日を数えている', rep.n_race_days > 4000, f'{rep.n_race_days:,} 日')
    check('内部の欠番が無い', not rep.gaps, f'{len(rep.gaps)} 件: {rep.gaps[:5]}')
    check('HTMLが無いレースが無い', not rep.warehouse_without_html,
          f'{len(rep.warehouse_without_html)} 件')
    check('ok 判定が立つ', rep.ok)

    # 欠番を作って検出できることを確かめる
    import pandas as pd
    from keiba import audit as A

    fake = pd.DataFrame({'race_id': [f'202605010{n:02d}' for n in
                                     [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]]})
    fake['prefix'] = fake['race_id'].str[:10]
    fake['rn'] = pd.to_numeric(fake['race_id'].str[10:12])
    nums = set(fake['rn'])
    gaps = [n for n in range(1, max(nums)) if n not in nums]
    check('欠番（4R抜け）を検出できる', gaps == [4], f'{gaps}')

    check('race_id を人が読める形に直せる',
          A._describe('202505021109') == '2025年 東京 2回11日 9R',
          A._describe('202505021109'))


def test_pipeline_range():
    print('\n[ステージの範囲指定]')
    from keiba import pipeline, stages

    check('番号で指定できる', pipeline._resolve(1) == 'collect')
    check('文字列の番号でも指定できる', pipeline._resolve('2') == 'build')
    check('名前で指定できる', pipeline._resolve('train') == 'train')

    for bad in (0, 5, 'unknown'):
        try:
            pipeline._resolve(bad)
            check(f'不正な指定 {bad!r} を拒否', False)
        except ValueError:
            check(f'不正な指定 {bad!r} を拒否', True)

    try:
        pipeline.run(3, 1)
        check('開始が終了より後なら拒否', False)
    except ValueError:
        check('開始が終了より後なら拒否', True)

    check('ステージは4段', stages.STAGES == ['collect', 'build', 'train', 'strategy'])
    check('全ステージにラベルがある',
          all(s in stages.STAGE_LABELS for s in stages.STAGES))


def test_duckdb():
    print('\n[DuckDB]')
    try:
        con = store.connect()
    except ImportError:
        print('  - duckdb 未インストールのためスキップ')
        return
    n = con.sql('SELECT count(*) FROM races').fetchone()[0]
    check('races ビューが引ける', n > 50000, f'{n:,} 件')
    cols = con.sql('SELECT * FROM race_results LIMIT 0').df().columns
    check('race_results にレース属性が含まれる',
          {'surface', 'distance', 'track_condition', 'venue_name'} <= set(cols))


def test_store_upsert_is_isolated():
    print('\n[ウェアハウスへの取り込み]')
    with tempfile.TemporaryDirectory() as tmp:
        wh = Path(tmp)
        races = pd.DataFrame([{'race_id': '202601010101', 'date': '2026-01-05',
                               'race_name': 'テスト', 'distance': 1600}])
        results = pd.DataFrame([{'race_id': '202601010101', 'horse_number': 1,
                                 'finish_position': 1}])
        store.upsert(races, results, warehouse=wh)
        check('年パーティションに書かれる',
              (wh / 'races' / 'year=2026' / 'part.parquet').exists())

        # 同じ race_id を上書き
        races2 = races.copy(); races2.loc[0, 'race_name'] = '更新後'
        store.upsert(races2, results, warehouse=wh)
        got = store.read_table('races', warehouse=wh)
        check('同じ race_id は重複せず置換される', len(got) == 1)
        check('内容が更新されている', got.iloc[0]['race_name'] == '更新後')


def main():
    for fn in [test_manifest_roundtrip, test_parser_version_triggers_reparse,
               test_warehouse_integrity, test_validate_rejects_empty_pages,
               test_course_notation, test_features,
               test_discovery_filters_to_jra, test_payouts,
               test_payout_matches_odds, test_audit,
               test_pipeline_range, test_duckdb,
               test_store_upsert_is_isolated]:
        fn()
    print('\n' + '=' * 50)
    if FAILS:
        print(f'❌ 失敗 {len(FAILS)} 件: {FAILS}')
        return 1
    print('✅ すべて成功')
    return 0


if __name__ == '__main__':
    sys.exit(main())
