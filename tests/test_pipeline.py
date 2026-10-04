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


def test_backtest():
    """点数の数え方と、同着の合算。"""
    print('\n[バックテスト]')
    import pandas as pd
    from keiba import backtest

    # 着順が関係する券種を組み合わせ数で数えると点数を過少に見積もる。
    # 実際にこれで三連単の回収率が439%と出た。
    check('三連複 5頭BOX = 10点', backtest.points_for('三連複', 5) == 10)
    check('三連単 5頭BOX = 60点（順列）', backtest.points_for('三連単', 5) == 60,
          f"{backtest.points_for('三連単', 5)}")
    check('馬連 5頭BOX = 10点', backtest.points_for('馬連', 5) == 10)
    check('馬単 5頭BOX = 20点（順列）', backtest.points_for('馬単', 5) == 20)

    # 同着のレースは当たりが複数行。合計する必要がある
    te = pd.DataFrame({
        'race_id': ['R1'] * 8,
        'horse_number': [1, 2, 3, 4, 5, 6, 7, 8],
        'finish_position': [1, 2, 3, 4, 5, 6, 7, 8],
        'score': [8, 7, 6, 5, 4, 3, 2, 1],
    })
    pay = pd.DataFrame({
        'race_id': ['R1', 'R1'],
        'bet_type': ['三連複', '三連複'],
        'horse_numbers': [[1, 2, 3], [1, 2, 4]],   # 3着同着のつもり
        'payout': [1000, 2000],
    })
    r = backtest.run(te, pay, label='t', column='score', n_pick=4,
                     bet_type='三連複', min_runners=8)
    check('同着の払戻を合算する', r.payout == 3000, f'{r.payout}')
    check('購入額は点数×100', r.cost == 4 * 100, f'{r.cost}')

    # 買い目に含まれない組み合わせは当たりにしない
    r2 = backtest.run(te, pay, label='t', column='score', n_pick=3,
                      bet_type='三連複', min_runners=8)
    check('買い目外は当たりにしない', r2.payout == 1000, f'{r2.payout}')

    check('必要レース数は赤字なら None', r2.races_to_confirm is None)

    # 買わないレースは資金が減らない
    te2 = pd.concat([te, te.assign(race_id='R2')], ignore_index=True)
    pay2 = pd.concat([pay, pay.assign(race_id='R2')], ignore_index=True)
    only_r1 = backtest.run(te2, pay2, label='t', column='score', n_pick=4,
                           bet_type='三連複', min_runners=8, bet_races={'R1'})
    check('対象レースは2、購入は1',
          only_r1.candidates == 2 and only_r1.races == 1,
          f'{only_r1.candidates}/{only_r1.races}')
    check('参加率が50%', abs(only_r1.coverage - 50.0) < 1e-9)
    check('買わない分は購入額に含めない', only_r1.cost == 400, f'{only_r1.cost}')

    # 信頼区間
    both = backtest.run(te2, pay2, label='t', column='score', n_pick=4,
                        bet_type='三連複', min_runners=8)
    check('信頼区間が出る', both.roi_ci95 is not None)
    lo, hi = both.roi_ci95
    check('区間が回収率を挟む', lo <= both.roi <= hi, f'{lo:.0f}〜{hi:.0f} / {both.roi:.0f}')


def test_speed_features():
    """走破タイム・上がりの集計。レース内正規化とリーク防止。"""
    print('\n[速度特徴量]')
    import numpy as np
    import pandas as pd
    from keiba.features import add_speed_features

    # 同じ馬が3走。タイムは条件で水準が違っても、レース内で正規化されるはず
    d = pd.DataFrame({
        'race_id': ['R1', 'R1', 'R1', 'R2', 'R2', 'R2', 'R3', 'R3', 'R3'],
        'horse_id': ['H', 'A', 'B', 'H', 'A', 'B', 'H', 'A', 'B'],
        'race_date': pd.to_datetime(
            ['2024-01-01'] * 3 + ['2024-02-01'] * 3 + ['2024-03-01'] * 3),
        # R1は短距離（速い）、R2は長距離（遅い）。水準が違っても相対化される
        'time_seconds': [70.0, 71.0, 72.0, 130.0, 131.0, 132.0, 70.5, 70.0, 71.0],
        'last_3f': [34.0, 35.0, 36.0, 36.0, 37.0, 38.0, 35.0, 34.0, 34.5],
    })
    out = add_speed_features(d).sort_values(['horse_id', 'race_date'])
    h = out[out['horse_id'] == 'H']

    check('レース内で正規化される（水準差が消える）',
          abs(h['time_z'].iloc[0] - h['time_z'].iloc[1]) < 1e-9,
          f"{h['time_z'].iloc[0]:.4f} / {h['time_z'].iloc[1]:.4f}")
    check('速いほど time_z が大きい',
          out[out['race_id'] == 'R1'].sort_values('time_seconds')['time_z'].is_monotonic_decreasing)

    # 初出走は過去が無いので欠損
    check('初出走の集計は欠損', bool(pd.isna(h['r3_time_z'].iloc[0])))
    # 2走目は1走目だけを見る（当該レースを含めない＝リークしない）
    check('2走目の集計は1走目のみを見る',
          abs(h['r3_time_z'].iloc[1] - h['time_z'].iloc[0]) < 1e-9,
          f"{h['r3_time_z'].iloc[1]:.4f} vs {h['time_z'].iloc[0]:.4f}")
    check('過去最速の上がりは絶対値で取る',
          abs(h['best_last_3f'].iloc[2] - 34.0) < 1e-9,
          f"{h['best_last_3f'].iloc[2]}")

    from keiba import stages
    check('市場フリーの特徴量に速度が入っている',
          set(stages.SPEED_FEATURES) <= set(stages.MARKET_FREE_FEATURES))
    check('重複していた horse_expected_top3_rate は外してある',
          'horse_expected_top3_rate' not in stages.MARKET_FREE_FEATURES)


def test_recency_features():
    """前走・前々走の人気と着順のズレ。"""
    print('\n[市場の過剰反応]')
    import pandas as pd
    from keiba.features import add_recency_features
    from keiba import segments as sg, stages

    d = pd.DataFrame({
        'race_id': ['R1', 'R2', 'R3'],
        'horse_id': ['H'] * 3,
        'race_date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        'popularity': [1, 13, 5],
        'finish_position': [12, 2, 4],
    })
    out = add_recency_features(d).sort_values('race_date')

    check('初出走は前走が欠損', bool(pd.isna(out['p1_gap'].iloc[0])))
    # R2 から見た前走は R1（1番人気で12着）→ gap = 12 - 1 = 11
    check('人気より着順が悪いと gap は正', out['p1_gap'].iloc[1] == 11,
          f"{out['p1_gap'].iloc[1]}")
    # R3 から見た前走は R2（13番人気で2着）→ gap = 2 - 13 = -11
    check('人気より着順が良いと gap は負', out['p1_gap'].iloc[2] == -11,
          f"{out['p1_gap'].iloc[2]}")
    check('surprise は絶対値', out['p1_surprise'].iloc[2] == 11)
    check('前々走も取れる', out['p2_gap'].iloc[2] == 11, f"{out['p2_gap'].iloc[2]}")

    # 過大評価グループの判定
    cand = pd.DataFrame({
        'p1_pop': [1, 1, 13, 13, 5, 5],
        'p1_fin': [10, 3, 15, 2, 10, 2],
    })
    want = [True, False, True, True, False, False]
    got = list(sg.overvalued_after_last_race(cand))
    check('1番人気で大敗 / 人気薄で激走・大敗 を検出', got == want, str(got))

    check('recency は市場フリーに入れない',
          not (set(stages.RECENCY_FEATURES) & set(stages.MARKET_FREE_FEATURES)))

    # 着差の表記 → 馬身
    from keiba.features import margin_to_lengths as ml
    for text, want in [('ハナ', 0.05), ('クビ', 0.2), ('1/2', 0.5),
                       ('1.1/4', 1.25), ('2.1/2', 2.5), ('3', 3.0), ('大', 10.0)]:
        got = ml(text)
        check(f'着差 {text} → {want}', abs(got - want) < 1e-9, f'{got}')
    check('空欄は欠損', pd.isna(ml('')))

    # 市場が買いすぎ / 見落としている馬の判定
    cand = pd.DataFrame({
        'p1_pop': [1, 5, 5, 5, 5, 5],
        'p1_fin': [10, 1, 5, 2, 5, 5],
        'p2_fin': [5, 1, 5, 5, 5, 5],
        'p1_margin': [1.0, 1.0, 10.0, 0.1, 0.1, 0.1],
        'days_since_last': [30, 30, 30, 50, 400, 50],
    })
    over = list(sg.overvalued(cand))
    check('1番人気で大敗 → 買いすぎ', over[0])
    check('2走連続勝利 → 買いすぎ', over[1])
    check('前走5馬身以上の大敗 → 買いすぎ', over[2])
    check('1年以上の休み明け → 買いすぎ', over[4])
    under = list(sg.undervalued(cand))
    check('適度な間隔＋僅差負け → 見落とし', under[5], str(under))
    check('休み明けは見落としに含めない', not under[4])

    # 実績が薄い馬。欠損のままの値で判定する
    import numpy as np
    thin_df = pd.DataFrame({
        'p1_pop': [np.nan, 3, 3, 3],
        'p1_fin': [np.nan, 2, 7, 2],       # デビュー / 1走で2着 / 1走で7着 / 2走
        'p2_fin': [np.nan, np.nan, np.nan, 6],
        'p1_margin': [np.nan, 0.5, 3.0, 0.5],
        'days_since_last': [np.nan, 30, 30, 30],
    })
    thin = list(sg.thin_record(thin_df))
    check('デビュー馬は実績が薄い', thin[0])
    check('1走だけで3着内は実績が薄い', thin[1])
    check('1走だけでも着外なら対象外', not thin[2])
    check('2走以上あれば対象外', not thin[3])
    check('欠損のままなら overvalued はデビュー馬を拾わない',
          not sg.overvalued(thin_df).iloc[0])

    # 以前のバックテストは 0 埋め後に overvalued を通していた。
    # overvalued | thin_record がそれと全行一致すること
    zero = thin_df.fillna(0)
    explicit = sg.overvalued(thin_df) | sg.thin_record(thin_df)
    check('明示ルールが 0埋め後の判定と一致',
          list(explicit) == list(sg.overvalued(zero)),
          f'{list(explicit)} / {list(sg.overvalued(zero))}')


def test_market():
    """市場の見立てを3着内確率に揃える計算。"""
    print('\n[市場確率]')
    import numpy as np
    from keiba import market

    p = market.implied_win_prob(np.array([2.0, 4.0, 8.0, 16.0]))
    check('勝率の合計が1', abs(p.sum() - 1.0) < 1e-9, f'{p.sum():.6f}')
    check('オッズが低いほど勝率が高い', p[0] > p[1] > p[2] > p[3])

    t3 = market.harville_top3(p)
    check('3着内確率の合計が3（4頭立て）', abs(t3.sum() - 3.0) < 1e-6, f'{t3.sum():.4f}')
    check('3着内確率は勝率以上', bool((t3 >= p - 1e-12).all()))
    check('3着内確率は1以下', bool((t3 <= 1.0 + 1e-12).all()))
    check('順序が保たれる', t3[0] > t3[1] > t3[2] > t3[3])

    # 3頭なら全馬が3着内
    p3 = market.implied_win_prob(np.array([2.0, 3.0, 6.0]))
    t3b = market.harville_top3(p3)
    check('3頭立てなら全馬の3着内確率が1', bool(np.allclose(t3b, 1.0, atol=1e-6)),
          str(np.round(t3b, 4)))

    # 市場フリーの特徴量に市場情報が混ざっていないこと
    from keiba import stages
    banned = {'odds', 'popularity', 'market_implied_win_prob', 'odds_ratio_to_fav',
              'pop_odds_mismatch', 'prev_odds', 'prev_popularity'}
    leaked = banned & set(stages.MARKET_FREE_FEATURES)
    check('市場フリーの特徴量に市場情報が無い', not leaked, str(leaked))

    # 支持率の式。Harville と違い、人気馬を過大評価しない
    odds = np.array([1.8, 4.5, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, 150.0, 200.0])
    p = market.top3_from_support(odds)
    hv = market.harville_top3(market.implied_win_prob(odds))
    check('支持率の式は確率の範囲', ((p > 0) & (p < 1)).all())
    check('支持率の式はオッズに対して単調', (np.diff(p) < 0).all(), f'{np.round(p, 3)}')
    check('人気馬は Harville より低く見積もる', p[0] < hv[0], f'{p[0]:.3f} / {hv[0]:.3f}')
    check('中穴は Harville より高く見積もる', p[5] > hv[5], f'{p[5]:.3f} / {hv[5]:.3f}')
    check('不正なオッズは欠損', np.isnan(market.top3_from_support(np.array([2.0, 0.0, 5.0])))[1])

def test_segments():
    """レースの区分。荒れ具合と回収率は別物なので両方出せること。"""
    print('\n[区分]')
    import pandas as pd
    from keiba import segments as sg

    cases = [('第71回優駿牝馬(G1)', 'S', 'G1'),
             ('きさらぎ賞(G3)', 'B', 'G3'),
             ('3歳未勝利', 'E', '未勝利'),
             ('2歳新馬', 'E', '新馬'),
             # 特別競走は名前にクラスが出ないので賞金ベースで補完する
             ('知床特別', 'D', '1-2勝'),
             ('サンライズステークス', 'C', 'OP・3勝')]
    for name, lv, want in cases:
        got = sg.race_grade(name, lv)
        check(f'{name} → {want}', got == want, '' if got == want else f'→ {got}')

    d = pd.DataFrame({
        'race_id': ['R1'] * 4,
        'horse_number': [1, 2, 3, 4],
        'finish_position': [1, 2, 3, 4],
        'popularity': [1, 3, 7, 14],
        'distance': [1200, 1800, 2200, 2800],
        'race_name': ['3歳未勝利'] * 4,
        'race_level': ['E'] * 4,
    })
    seg = sg.add_segments(d)
    check('距離の区分が付く',
          list(seg['dist_band']) == ['~1500', '1600-1900', '2000-2300', '2600~'],
          str(list(seg['dist_band'])))
    check('人気帯が付く',
          list(seg['pop_band']) == ['1番人気', '2-3', '6-8', '13-'],
          str(list(seg['pop_band'])))

    # 複勝払戻の紐づけ。3着内でなければ0
    pay = pd.DataFrame({'race_id': ['R1'] * 3, 'bet_type': ['複勝'] * 3,
                        'horse_numbers': [[1], [2], [3]], 'payout': [110, 150, 300]})
    att = sg.attach_place_payout(seg, pay)
    check('3着内に払戻が付く', list(att['fuku'])[:3] == [110, 150, 300])
    check('4着は0', list(att['fuku'])[3] == 0)


def test_betting():
    """三連複の買い方。点数の数え方と、買わない判断。"""
    print('\n[買い方]')
    from keiba import betting as bt

    order = [5, 3, 9, 1, 7, 2, 11]
    for bet, name, want in [
            ('三連複', '3頭BOX', 1), ('三連複', '軸2頭→相手3頭', 3),
            ('三連複', '4頭BOX', 4), ('三連複', '軸1頭→相手4頭', 6),
            ('三連複', '5頭BOX', 10),
            # ワイドと馬連は2頭組なので、同じ相手数でも点数が少ない
            ('ワイド', '軸1頭→相手2頭', 2), ('ワイド', '軸1頭→相手3頭', 3),
            ('ワイド', '4頭BOX', 6),
            ('馬連', '軸1頭→相手2頭', 2), ('馬連', '軸1頭→相手3頭', 3)]:
        got = len(bt.SHAPES[bet][name](order))
        check(f'{bet} {name} = {want}点', got == want, f'{got}')
    check('どの買い方も上限10点以内',
          all(len(f(order)) <= bt.MAX_POINTS
              for shapes in bt.SHAPES.values() for f in shapes.values()))
    check('2頭組の券種は3頭流しが3点',
          len(bt.SHAPES['ワイド']['軸1頭→相手3頭'](order))
          < len(bt.SHAPES['三連複']['軸1頭→相手4頭'](order)))

    # 軸流しは軸を必ず含む
    combos = bt.SHAPES['馬連']['軸1頭→相手3頭'](order)
    check('馬連の軸流しは軸を必ず含む', all(order[0] in c for c in combos))
    check('馬連は2頭組', all(len(c) == 2 for c in combos))

    hn = [5, 3, 9, 1, 7, 2, 11, 4]
    # 確信度が高いと馬連、低いと三連複1点
    p = bt.plan_race('R', hn, [.6, .55, .5, .3, .2, .1, .1, .05])
    check('確信度が高いと馬連', p.bet_type == '馬連', f'{p.bet_type} {p.shape}')
    p1 = bt.plan_race('R', hn, [.5, .48, .45, .3, .2, .1, .1, .05])
    check('確信度が低いと三連複1点',
          p1.bet_type == '三連複' and p1.points == 1,
          f'{p1.bet_type} {p1.shape} {p1.points}点')
    # 券種を固定することもできる
    p2 = bt.plan_race('R', hn, [.6, .55, .5, .3, .2, .1, .1, .05],
                      rules=((1.4, 'ワイド', '軸1頭→相手3頭'),))
    check('券種を固定できる', p2.bet_type == 'ワイド' and p2.points == 3,
          f'{p2.bet_type} {p2.points}点')

    # 混戦は買わない。点数を増やすのではない
    p3 = bt.plan_race('R', hn, [.2, .2, .2, .2, .2, .1, .1, .05])
    check('混戦は買わない', p3.shape is None and '混戦' in p3.reason, p3.reason)
    p4 = bt.plan_race('R', hn, [.6, .5, .5, .3, .2, .1, .1, .05], skip=True)
    check('過大評価なら買わない', p4.shape is None, p4.reason)
    p4b = bt.plan_race('R', hn, [.6, .5, .5, .3, .2, .1, .1, .05], skip='実績が薄い')
    check('見送りの理由を渡せる', p4b.shape is None and p4b.reason == '実績が薄い', p4b.reason)
    p4c = bt.plan_race('R', hn, [.6, .55, .5, .3, .2, .1, .1, .05], skip='')
    check('空の理由なら買う', p4c.shape is not None, p4c.reason)
    p5 = bt.plan_race('R', hn[:5], [.6, .5, .5, .3, .2])
    check('少頭数は買わない', p5.shape is None, p5.reason)

    # 券種ごとの払戻表を混ぜても正しく引ける
    plan_t = bt.plan_race('R', hn, [.5, .48, .45, .3, .2, .1, .1, .05])
    plan_u = bt.plan_race('R', hn, [.6, .55, .5, .3, .2, .1, .1, .05])
    nested = {'三連複': {'R': [(next(iter(plan_t.combos)), 5000)]},
              '馬連': {'R': [(next(iter(plan_u.combos)), 800)]}}
    rt = bt.evaluate([plan_t], nested)
    ru = bt.evaluate([plan_u], nested)
    # 平らな払戻表（三連複のつもり）に他の券種を混ぜたら止める。
    # 以前ステージ4がこれで馬連を全部外れ扱いにし、回収率14.9%と出ていた
    try:
        bt.evaluate([plan_u], {'R': [(next(iter(plan_u.combos)), 800)]})
        check('平らな払戻表に馬連を渡すと止まる', False)
    except ValueError:
        check('平らな払戻表に馬連を渡すと止まる', True)

    pdf = pd.DataFrame({'race_id': ['R', 'R', 'R'],
                        'bet_type': ['三連複', '馬連', '単勝'],
                        'horse_numbers': [[1, 2, 3], [1, 2], [1]],
                        'payout': [5000, 800, 300]})
    idx2 = bt.payout_index(pdf)
    check('payout_index は券種ごとに分ける',
          idx2['馬連']['R'] == [(frozenset([1, 2]), 800)]
          and idx2['三連複']['R'] == [(frozenset([1, 2, 3]), 5000)])
    check('payout_index は買わない券種を含めない', '単勝' not in idx2)

    check('券種別の払戻表を引き分ける',
          rt['収支'] == 5000 - plan_t.cost and ru['収支'] == 800 - plan_u.cost,
          f"{rt['収支']} / {ru['収支']}")

    # 同着は合算、買い目外は当たりにしない。
    # スコアが同値のときは馬番の大きい方が先に来るので、軸は [5, 9] になる
    plan = bt.plan_race('R', hn, [.6, .5, .5, .3, .2, .1, .1, .05],
                        rules=((1.4, '三連複', '軸2頭→相手3頭'),))
    axis = sorted({h for c in plan.combos for h in c}
                  & set.intersection(*[set(c) for c in plan.combos]))
    check('同点時の軸が決まっている', axis == [5, 9], str(axis))

    inside = sorted(plan.combos, key=sorted)[:2]
    idx = {'R': [(inside[0], 1000),                 # 買い目に含まれる
                 (inside[1], 2000),                 # これも含まれる（同着）
                 (frozenset([2, 11, 4]), 9999)]}    # 含まれない
    r = bt.evaluate([plan], idx)
    check('同着を合算し、買い目外は除く', r['収支'] == 3000 - plan.cost,
          f"{r['収支']} (cost {plan.cost})")


def test_strategy_stage_order():
    """戦略の3段階が正しく構成されていること。"""
    print('\n[戦略の段階]')
    import inspect
    from keiba import stages

    src = inspect.getsource(stages.strategy)
    for n, label in [(1, '候補のスコアリング'), (2, '市場と独立'), (3, '除外と買い方')]:
        check(f'【{n}】{label} の段がある', f'【{n}】' in src)

    # 市場なしの特徴量に市場情報が混ざっていないこと（段階2の前提）
    banned = {'odds', 'popularity', 'market_implied_win_prob',
              'odds_ratio_to_fav', 'pop_odds_mismatch', 'prev_odds',
              'prev_popularity'}
    check('市場なしモデルの特徴量が独立している',
          not (banned & set(stages.MARKET_FREE_FEATURES)))
    # 市場ありは市場情報を含むこと
    check('市場ありモデルは市場情報を含む',
          bool(banned & set(stages.STRATEGY_FEATURES)))

    # 買い目用モデルの特徴量。ステージ3・4・5で同じものを使う
    cols = (stages.STRATEGY_FEATURES + stages.MARKET_FREE_FEATURES
            + stages.RECENCY_FEATURES + ['unrelated'])
    f = stages.strategy_feature_list(cols)
    check('買い目用モデルは速度特徴量を含む', 'r3_time_z' in f)
    check('買い目用モデルは前走のズレを含む', 'p1_gap' in f)
    check('買い目用モデルは市場を含む', 'odds' in f)
    check('関係ない列は入らない', 'unrelated' not in f)
    check('重複が無い', len(f) == len(set(f)))
    check('ステージ4が共通定義を使う',
          'strategy_feature_list(' in src and 'STRATEGY_ROUNDS' in src)
    check('ステージ4は除外を 0埋めの前に判定する',
          src.index('thin_record') < src.index('.fillna(0)'))

    # model_input は元の df を書き換えない（除外は欠損のままで判定するため）
    import numpy as np
    d = pd.DataFrame({'a': [1.0, np.nan], 'b': [np.inf, 2.0]})
    x = stages.model_input(d, ['a', 'b'])
    check('model_input は欠損と無限大を 0 に', x.values.tolist() == [[1.0, 0.0], [0.0, 2.0]])
    check('model_input は元を書き換えない', pd.isna(d.loc[1, 'a']))

    # ステージ5もバックテストと同じモデル・同じ除外を使う
    src5 = inspect.getsource(stages.score_upcoming)
    check('ステージ5は買い目用モデルを使う', 'model_strategy.pkl' in src5)
    check('ステージ5も実績の薄い馬を除外する', 'thin_record' in src5)


def test_shutuba():
    """出馬表の取得。通信せずに検証できる部分のみ。"""
    print('\n[出馬表]')
    from keiba import shutuba, fetch

    # 文字コードはホストで違う。一律 EUC-JP にすると出馬表が文字化けする
    check('db.netkeiba は EUC-JP',
          fetch.encoding_for('https://db.netkeiba.com/race/202601010101') == 'EUC-JP')
    check('race.netkeiba は UTF-8',
          fetch.encoding_for('https://race.netkeiba.com/race/shutuba.html') == 'UTF-8')

    # グレードのアイコンは2桁のことがある。15 を 1（G1）と誤認しないこと
    check('Icon_GradeType15 はリステッド', shutuba.GRADE_ICON.get('15') == 'L')
    check('Icon_GradeType1 は G1', shutuba.GRADE_ICON.get('1') == 'G1')

    # オッズをそのまま使うと尺度が合わない。3着内確率に直す必要がある
    import numpy as np
    from keiba import market
    odds = np.array([2.0, 4.0, 8.0, 16.0, 30.0, 50.0, 80.0, 100.0])
    naive = sorted(1.0 / odds, reverse=True)[:3]
    proper = sorted(market.harville_top3(market.implied_win_prob(odds)), reverse=True)[:3]
    check('1/オッズ の上位3頭合計は閾値に届かない', sum(naive) < 1.4,
          f'{sum(naive):.2f}')
    check('Harville なら閾値と同じ尺度になる', sum(proper) > 1.4, f'{sum(proper):.2f}')

    check('race_id の抽出', shutuba.RACE_ID_RE.findall(
        'a href="/race/shutuba.html?race_id=202605040111"') == ['202605040111'])

    # 発走までの時間。直前運用で対象レースを絞るのに使う
    import datetime as _dt
    now = _dt.datetime(2026, 10, 3, 10, 0)
    for t, want in [('10:00', 0), ('09:59', -1), ('11:30', 90), ('11:31', 91)]:
        got = shutuba.minutes_to_post(
            {'date': '2026-10-03', 'start_time': t}, now=now)
        check(f'発走{t} → {want}分', got == want, f'{got}')
    check('発走時刻が無ければ None',
          shutuba.minutes_to_post({'date': '2026-10-03'}) is None)

    check('オッズの状態を日本語に', shutuba.ODDS_STATUS.get('final') == '確定')
    check('発売中は前日に限らない', shutuba.ODDS_STATUS.get('middle') == '発売中')
    check('発走後の result も確定', shutuba.ODDS_STATUS.get('result') == '確定')
    # 当日のレース結果ページ。db.netkeiba に反映される前の突き合わせに使う
    html = """<table class="RaceTable01"><tr><th>着順</th></tr>
<tr><td>1</td><td>3</td><td>5</td><td>パンサーズ</td><td>牡3</td><td>56.0</td><td>松若</td>
<td>1:23.6</td><td></td><td>2</td><td>4.7</td></tr>
<tr><td>取消</td><td>1</td><td>1</td><td>トリケシ</td><td>牡3</td><td>56.0</td><td>某</td>
<td></td><td></td><td></td><td></td></tr></table>
<table class="Payout_Detail_Table">
<tr><th>複勝</th><td class="Result"><div><span>5</span></div><div><span></span></div><div><span>4</span></div></td>
<td class="Payout"><span>190円<br>850円</span></td></tr>
<tr><th>ワイド</th><td class="Result"><ul><li><span>4</span></li><li><span>5</span></li><li></li></ul>
<ul><li><span>5</span></li><li><span>7</span></li><li></li></ul></td>
<td class="Payout"><span>3,020円<br>560円</span></td></tr>
<tr><th>3連複</th><td class="Result"><ul><li><span>4</span></li><li><span>5</span></li><li><span>7</span></li></ul></td>
<td class="Payout"><span>14,700円</span></td></tr></table>"""
    res = shutuba.parse_result(html)
    check('結果の着順を読める', res['order'][0]['horse_number'] == 5 and res['order'][0]['odds'] == 4.7)
    check('取消は着順なし', res['order'][1]['rank'] is None)
    pays = {(p['bet_type'], tuple(p['horse_numbers'])): p['payout'] for p in res['payouts']}
    check('複勝を馬ごとに読める', pays.get(('複勝', (4,))) == 850, f'{pays}')
    check('ワイドを組ごとに読める', pays.get(('ワイド', (5, 7))) == 560)
    check('3連複は三連複として読む', pays.get(('三連複', (4, 5, 7))) == 14700)
    check('確定済みと判定', res['confirmed'])
    blurred = html.replace('<td class="Result"><ul>', '<td class="Result"><span class="Bokashi_Img"></span><ul>', 1)
    check('組み合わせが伏せられていれば未確定', not shutuba.parse_result(blurred)['confirmed'])

    # init は1時間近く古いことがある。update を先に試す
    check('オッズは update を先に取る', shutuba.ODDS_ACTIONS[0] == 'update')

    check('斤量をfloatに', shutuba._try_float('57.0 kg') == 57.0)
    check('Rをintに', shutuba._try_int('11R') == 11)
    check('数字が無ければ None', shutuba._try_int('') is None)

    # 性齢・本賞金・馬体重
    check('性齢を分解', shutuba._sex_age('牡2') == ('牡', 2))
    check('セン馬も読める', shutuba._sex_age('セ5') == ('セ', 5))
    check('空なら None', shutuba._sex_age('') == (None, None))
    cond = '4回 東京 1日目 サラ系２歳 未勝利 (混)[指] 馬齢 15頭 本賞金:590,240,150,89,59万円'
    check('1着賞金を取れる', shutuba._first_prize(cond) == 590.0,
          f'{shutuba._first_prize(cond)}')
    check('本賞金が無ければ None', shutuba._first_prize('15頭') is None)

    # 出馬表 → results 形式。学習時の特徴量をそのまま作れる形にする
    card = {'race_id': 'R1', 'date': '2026-10-03', 'race_name': '2歳未勝利',
            'surface': 'ダ', 'distance': 1600, 'condition': cond,
            'horses': [{'horse_number': 1, 'horse_name': 'ア', 'horse_id': 'h1',
                        'jockey_id': 'j1', 'trainer_id': 't1', 'bracket': 1,
                        'sex_age': '牡2', 'impost': 56.0, 'jockey_name': '丹内',
                        'horse_weight': 458.0, 'odds': 4.2, 'popularity': 2}]}
    rows = shutuba.to_result_rows([card])
    r = rows.iloc[0]
    check('馬は horse_id で紐付ける', r['horse_id'] == 'h1')
    check('賞金から level_score を復元', r['level_score'] == 1, f"{r['level_score']}")
    check('性齢を展開', (r['sex'], r['age']) == ('牡', 2))
    check('着順は走る前の印', r['finish_position'] == shutuba.UNRUN)
    check('走後にしか分からない列は空', pd.isna(r['time_seconds']) and pd.isna(r['last_3f']))
    check('未走行の印が付く', bool(r['is_unrun']))

    # 出馬表の表記をウェアハウスに合わせる。「ダ」のままだと芝ダ特徴量が壊れる
    for raw, want in [('ダ', 'ダート'), ('芝', '芝'), ('障', '障害')]:
        got = shutuba.to_result_rows([{**card, 'surface': raw}]).iloc[0]['surface']
        check(f'芝ダ「{raw}」→「{want}」', got == want, got)

    # 馬体重は発走1時間前まで出ない。前走の値で埋める
    from keiba import stages
    hist = pd.DataFrame({'horse_id': ['h1', 'h1', 'h2'],
                         'race_date': pd.to_datetime(['2026-08-01', '2026-09-01', '2026-09-01']),
                         'horse_weight': [470.0, 476.0, 500.0]})
    up = pd.DataFrame({'horse_id': ['h1', 'h2', 'h3'],
                       'horse_weight': [None, 498.0, None]})
    filled = stages._fill_weight_from_last_race(up, hist)['horse_weight'].tolist()
    check('未発表なら直近の前走の馬体重', filled[0] == 476.0, f'{filled}')
    check('発表済みならそのまま', filled[1] == 498.0)
    check('前走が無ければ欠損のまま', pd.isna(filled[2]))

    # 着順 99 は is_top3 を立てない。過去の集計を汚さないため
    from keiba.features import C03_CONFIG, clean, add_horse_features, add_added_value
    races_df = pd.DataFrame({'race_id': ['R1'], 'race_name': ['2歳未勝利']})
    c = clean(races_df, rows.assign(odds=4.2), C03_CONFIG)
    check('未走行の行は is_top3 が立たない', c['is_top3'].iloc[0] == 0)

    # 同じ騎手が同じ日に複数鞍乗っても、未走行の行は平均に混ざらない
    two = pd.concat([rows, rows.assign(race_id='R2')], ignore_index=True)
    c2 = clean(pd.DataFrame({'race_id': ['R1', 'R2'], 'race_name': ['a', 'b']}),
               two, C03_CONFIG)
    c2 = add_added_value(add_horse_features(c2, C03_CONFIG))
    check('未走行の行は騎手の平均に入らない',
          c2['added_value_in_race'].isna().all(),
          f"{c2['added_value_in_race'].tolist()}")


def test_pace():
    """レースのペースの3段階分類。"""
    print('\n[ペース]')
    from keiba import pace
    races = pd.DataFrame({
        'venue_code': ['05'] * 40, 'surface': ['芝'] * 40, 'distance': [1600] * 40,
        'track_condition': ['良'] * 40,
        # 実際の偏差はほぼ正規分布。その分位点で並べる
        'pace_first3f': [34.0 + 0.5 * float(x) for x in
                         __import__('scipy').stats.norm.ppf([(i + 0.5) / 40 for i in range(40)])],
        'pace_last3f': [35.0] * 40})
    r = pace.add_race_pace(races)
    check('前半が速いレースはハイ', r['pace'].iloc[0] == 'ハイ')
    check('前半が遅いレースはスロー', r['pace'].iloc[-1] == 'スロー')
    share = r['pace'].value_counts(normalize=True)
    check('おおむね3等分', all(0.25 < share.get(k, 0) < 0.42 for k in pace.PACE_LABELS), f'{share.to_dict()}')
    few = pace.add_race_pace(races.head(10))
    check('比べるレースが少なければ判定しない', few['pace'].isna().all())


def test_views():
    """目線ごとの上位3頭の表。"""
    print('\n[目線]')
    from keiba import views
    t = pd.DataFrame({'馬名': list('ABCDE'), '人気': [1, 2, 3, 4, 7], '印': ['', '', '', '', ''],
                      '市場あり_順': [1, 2, 3, 4, 5], '指数・近3走_順': [4, 5, 3, 2, 1]},
                     index=[1, 2, 3, 4, 5])
    t['上位の目線の数'] = (t[['市場あり_順', '指数・近3走_順']] <= 3).sum(axis=1)
    t['人気の割に評価が高い'] = (t['人気'] >= 4) & (t['上位の目線の数'] > 0)
    top = views.top3_table(t)
    check('目線ごとに3頭', list(top['目線']) == ['市場あり', '指数・近3走'], f"{list(top['目線'])}")
    check('1番手は順位1の馬', top.iloc[1]['1番手'].startswith('5 E'))
    md = views.to_markdown({'venue_name': '東京', 'race_num': 11, 'race_name': 'テスト'}, t)
    check('人気薄で目線上位の馬を挙げる', '人気の割に評価が高い' in md and '5 E' in md.split('人気の割に評価が高い')[1])


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

    for bad in (0, 6, 'unknown'):
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

    check('ステージは5段',
          stages.STAGES == ['collect', 'build', 'train', 'strategy', 'predict'],
          str(stages.STAGES))
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
               test_payout_matches_odds, test_backtest,
               test_speed_features, test_recency_features, test_market,
               test_segments, test_betting, test_shutuba,
               test_strategy_stage_order,
               test_pace, test_views, test_audit,
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
