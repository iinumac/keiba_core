"""パイプラインの各段

01〜04 のノートブックと、連続実行用の 00_run は**すべてここを呼ぶ**。
処理をノートブック側に書くと、片方だけ直して食い違う事故が起きるため、
実体は必ずこのモジュールに置くこと。

    1. collect  取り込み — netkeiba から未取得のHTMLを収集
    2. build    パース   — HTML をウェアハウスへ
    3. train    学習     — 3着内確率モデルとマスタを作る
    4. strategy 予想     — 三連複の戦略評価

各段は結果を dict で返し、進捗は print する（ノートブックで読む前提）。
`push=True` なら成果物を GitHub に保存する。これを怠ると Colab の
ランタイム終了とともに消える。
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

STAGES = ['collect', 'build', 'train', 'strategy', 'predict']
STAGE_LABELS = {
    'collect': '1. 取り込み（HTML収集）',
    'build': '2. パース（ウェアハウス構築）',
    'train': '3. 学習（モデル・マスタ作成）',
    'strategy': '4. 戦略評価（買い方の検証）',
    'predict': '5. 予想（今週末の出馬表 → 買い目）',
}

TRAIN_END_YEAR = 2024

TAKEOUT_BASELINE = {
    '単勝': 80.0, '複勝': 80.0, '枠連': 77.5, '馬連': 77.5, 'ワイド': 77.5,
    '馬単': 75.0, '三連複': 75.0, '三連単': 72.5,
}
"""100 - JRAの控除率。ランダムに買えば回収率はここに収束する。

回収率を「100%を下回っているから駄目」と見るのではなく、
この基準をどれだけ上回ったかがモデルの実力。
"""

FEATURES_NO_ODDS = [
    'distance', 'surface_encoded', 'level_score', 'impost',
    'horse_expected_top3_rate', 'jockey_added_value', 'trainer_added_value',
    'prev_finish', 'prev_last_3f', 'days_since_last', 'is_jockey_changed',
]
FEATURES_WITH_ODDS = FEATURES_NO_ODDS + ['popularity', 'odds']

SPEED_FEATURES = [
    'r3_time_z', 'avg_time_z', 'best_time_z',
    'r3_l3f_z', 'avg_l3f_z', 'best_l3f_z', 'best_last_3f',
]
"""走破タイム・上がり3F をレース内で正規化し、馬ごとに集計したもの。

`r3_time_z`（直近3走のタイム偏差）が市場フリーモデルの最重要特徴量。
prev_finish を上回る。
"""

MARKET_FREE_FEATURES = [
    # horse_expected_top3_rate は horse_prev_top3_rate と完全に同じ値
    # （features.py が別名で出している）なので入れない
    'horse_prev_win_rate', 'horse_prev_top3_rate',
    'prev_finish', 'prev_last_3f', 'days_since_last', 'is_debut',
    'is_jockey_changed', 'jockey_added_value', 'trainer_added_value',
    'distance', 'surface_code', 'impost', 'horse_weight', 'horse_number',
    'level_score', 'age',
] + SPEED_FEATURES
"""市場の情報を一切使わない特徴量。

「市場が見落としているもの」を測るには、市場と独立した推定が要る。
オッズ・人気とその派生はもちろん、prev_odds と prev_popularity も
前走とはいえ市場の評価なので外す。

この構成での AUC は 0.7527（速度特徴量なしだと 0.7417）。
単勝オッズ単独の 0.8189 には及ばない。市場は調教や当日の気配まで
織り込んでいる。詳細は docs/MARKET.md。
"""

RECENCY_FEATURES = [
    'p1_pop', 'p1_fin', 'p1_gap', 'p1_surprise',
    'p2_pop', 'p2_fin', 'p2_gap', 'p2_surprise', 'gap_mean2',
    'p1_margin', 'streak_top3', 'streak_win', 'jockey_rides_delta',
]
"""前走・前々走の人気と着順のズレ。市場の過剰反応を捉える。

前走の人気を使うので市場情報を部分的に含む。MARKET_FREE には入れない。
足すと市場フリーモデルは 0.7527 → 0.7584。
市場込みモデルの AUC は動かないが（+0.0002）、回収率には効く
（過大評価グループを除くと全体 74.5% → 75.8%、人気薄に限れば +2.2pt）。
"""

STRATEGY_FEATURES = [
    'popularity', 'odds', 'market_implied_win_prob', 'odds_ratio_to_fav',
    'pop_odds_mismatch', 'horse_prev_win_rate', 'horse_prev_top3_rate',
    'prev_finish', 'prev_popularity', 'prev_odds', 'prev_last_3f',
    'days_since_last', 'is_debut', 'jockey_added_value', 'trainer_added_value',
    'distance', 'surface_code', 'impost', 'horse_weight', 'horse_number',
]


STRATEGY_PARAMS = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
                   'num_leaves': 63, 'verbose': -1, 'seed': 42}
STRATEGY_ROUNDS = 600
"""買い目用モデルの学習条件。ステージ3（保存）とステージ4（評価）で共有する。"""


def strategy_feature_list(columns) -> List[str]:
    """買い目用モデルの特徴量。市場あり＋市場なし＋前走のズレ。

    ステージ3の model_with_odds（13特徴量）では、買い目に使うと回収率が
    80% にとどまり、単勝オッズだけ（79%）と変わらなかった。タイム偏差や
    前走のズレまで入れたこの構成で 93〜95% になる。
    """
    cols = set(columns)
    return [c for c in dict.fromkeys(STRATEGY_FEATURES + MARKET_FREE_FEATURES
                                     + RECENCY_FEATURES) if c in cols]


def model_input(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """モデルに渡す形に整える。数値化し、無限大と欠損を 0 にする。

    元の df は書き換えない。除外の判定（segments.thin_record など）は
    欠損のままの値で行う必要があるため。
    """
    import numpy as np
    return (df[features].apply(pd.to_numeric, errors='coerce')
            .replace([np.inf, -np.inf], np.nan).fillna(0))


def _rule(title: str) -> None:
    print(f'\n{"=" * 60}\n{title}\n{"=" * 60}', flush=True)


# ---------------------------------------------------------------------------
# 1. 取り込み
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 14
"""自動で開始日を決めるとき、手持ちの最新日から何日さかのぼるか。

netkeiba の db サイトは結果の反映が遅れる。実際に 2026-09-27 時点で、
9/26（土）のレース一覧には地方の33件しか載っておらず、中央は0件だった。

「最新日の翌日から」にすると、9/27 の分を取り込んだ時点で 9/26 が
二度と見られなくなる。数日さかのぼって確認すれば取りこぼさない。
すでに取得済みの race_id は known で除外されるので、余分な
ダウンロードは発生しない（増えるのは開催日ぶんのリクエストだけ）。
"""


def collect(start: Optional[dt.date] = None, end: Optional[dt.date] = None,
            push: bool = True, dry_run: bool = False,
            lookback_days: int = LOOKBACK_DAYS,
            race_ids: Optional[List[str]] = None) -> Dict:
    """未取得のレースHTMLを収集して GitHub に保存する。

    Args:
        start: 収集開始日。None なら「手持ちの最新レース日 - lookback_days」。
        end: 収集終了日。None なら今日。
        dry_run: 検知だけ行い、ダウンロードはしない。
        lookback_days: 自動決定時にさかのぼる日数。反映遅れ対策。
        race_ids: 指定すると日付による検知を行わず、この race_id だけを取得する。
            `audit.local()` が見つけた欠番を埋めるときに使う。
    """
    import pandas as pd
    from collections import Counter
    from . import config, discovery, fetch, gitpush, manifest, store

    _rule(STAGE_LABELS['collect'])

    fetcher = fetch.Fetcher()
    diag = fetcher.diagnose()
    print(f'疎通確認: {diag}')
    if not diag.startswith('ok'):
        print('⚠️ この環境からは取得できません。中断します。')
        return {'stage': 'collect', 'ok': False, 'reason': diag}

    if race_ids is not None:
        return _collect_ids(list(race_ids), fetcher, push=push, dry_run=dry_run)

    races = store.read_table('races', columns=['race_id', 'date'])
    known = set(races['race_id'].astype(str)) if not races.empty else set()
    known |= set(manifest.load()['race_id'].astype(str))

    auto = start is None
    if auto:
        if races.empty:
            start = dt.date(2010, 1, 1)
        else:
            latest = pd.to_datetime(races['date']).max().date()
            start = latest - dt.timedelta(days=lookback_days)
    end = end or dt.date.today()

    print(f'手持ち: {len(known):,} レース')
    print(f'収集期間: {start} 〜 {end}'
          + (f'（最新日から{lookback_days}日さかのぼって確認）' if auto else ''))
    if start > end:
        print('収集対象の期間がありません。')
        return {'stage': 'collect', 'ok': True, 'found': 0, 'saved': 0}

    W = '月火水木金土日'
    excluded = [0]

    def on_month(y, m, days):
        print(f'  {y}年{m:2d}月: 開催 {len(days)} 日', flush=True)

    def on_day(r):
        excluded[0] += r.excluded
        new = len([x for x in r.race_ids if x not in known])
        print(f'    {r.day} ({W[r.day.weekday()]})  中央 {len(r.race_ids):2d} レース'
              f'（未取得 {new} / 地方を {r.excluded} 件除外）', flush=True)

    new_ids = discovery.discover(start, end, fetcher, known=known,
                                 on_month=on_month, on_day=on_day)
    print(f'\n未取得: {len(new_ids)} 件（地方 {excluded[0]} 件を除外）')

    if dry_run or not new_ids:
        return {'stage': 'collect', 'ok': True, 'found': len(new_ids), 'saved': 0}

    outcomes: Counter = Counter()
    for i, race_id in enumerate(new_ids, 1):
        outcomes[fetcher.download_race(race_id, config.HTML_DIR)] += 1
        if i % 50 == 0:
            print(f'  {i}/{len(new_ids)}  {dict(outcomes)}', flush=True)

    saved = outcomes.get(fetch.Outcome.SAVED, 0)
    print(f'\n取得結果: ' + ', '.join(f'{k.value}={v}' for k, v in outcomes.items()))
    if outcomes.get(fetch.Outcome.BLOCKED):
        print('⚠️ 403 が発生しました。この環境のIPがブロックされている可能性があります。')

    if push and saved:
        gitpush.push(['data/html'], f'Add {saved} race HTML files')

    return {'stage': 'collect', 'ok': True, 'found': len(new_ids), 'saved': saved,
            'outcomes': {k.value: v for k, v in outcomes.items()}}


def _collect_ids(race_ids: List[str], fetcher, push: bool = True,
                 dry_run: bool = False) -> Dict:
    """race_id を指定して取得する（欠番の穴埋め用）。"""
    from collections import Counter
    from . import config, fetch, gitpush

    print(f'指定された {len(race_ids)} 件を取得します')
    if dry_run:
        for r in race_ids:
            print(f'  {r}')
        return {'stage': 'collect', 'ok': True, 'found': len(race_ids), 'saved': 0}

    outcomes: Counter = Counter()
    for race_id in race_ids:
        o = fetcher.download_race(race_id, config.HTML_DIR)
        outcomes[o] += 1
        print(f'  {race_id}: {o.value}', flush=True)

    saved = outcomes.get(fetch.Outcome.SAVED, 0)
    if push and saved:
        gitpush.push(['data/html'], f'Add {saved} race HTML files (gap fill)')
    return {'stage': 'collect', 'ok': True, 'found': len(race_ids), 'saved': saved,
            'outcomes': {k.value: v for k, v in outcomes.items()}}


# ---------------------------------------------------------------------------
# 2. パース
# ---------------------------------------------------------------------------
def build(push: bool = True, workers: Optional[int] = None) -> Dict:
    """増えた分だけパースしてウェアハウスを更新する。"""
    import time
    from . import build as build_mod
    from . import config, gitpush, manifest, store

    _rule(STAGE_LABELS['build'])

    man = manifest.load()
    print(f'マニフェスト登録済み: {len(man):,} 件（パーサ版 {config.PARSER_VERSION}）')

    t0 = time.time()

    def prog(done, total, phase):
        if phase == 'parse' and done % 2000 == 0:
            el = time.time() - t0
            print(f'  {done:,}/{total:,}  {el:.0f}s  {done / el:.0f}件/秒', flush=True)

    result = build_mod.build(workers=workers, hash_all=False, progress=prog)

    print(f"\nパース: {result['parsed']:,} 件  {result['reasons']}")
    print(f"  races {result['races']:,} / results {result['results']:,}"
          f" / payouts {result.get('payouts', 0):,}")
    print(f"  取り込まなかった（結果の無い空ページ）: {result['invalid']}")
    print(f"  失敗: {result['failed']}")

    store.build_duckdb()
    print('DuckDB のビューを更新しました')

    from . import audit
    print()
    rep = audit.local()
    rep.print_report(max_list=10)
    result['audit_gaps'] = rep.gaps

    if push and result['parsed']:
        gitpush.push(['data/warehouse'], 'Update warehouse')

    result['stage'] = 'build'
    result['ok'] = result['failed'] == 0
    return result


# ---------------------------------------------------------------------------
# 3. 学習
# ---------------------------------------------------------------------------
def train(push: bool = True) -> Dict:
    """3着内確率モデルと、予測用マスタを作る。"""
    import pickle
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from . import config, gitpush, store
    from .features import build_features, C03_CONFIG

    _rule(STAGE_LABELS['train'])

    races, results = store.load_for_features()
    df = build_features(races, results, C03_CONFIG)
    print(f'レース {len(races):,} / 出走馬 {len(results):,} / 特徴量 {len(df):,} 行')

    d = df[df['is_debut'] == 0].copy()
    d['level_score'] = d['level_score'].fillna(0)
    d['impost'] = d['impost'].fillna(55.0)
    for c in FEATURES_WITH_ODDS:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan).fillna(0)

    tr = d[d['year'] <= TRAIN_END_YEAR]
    te = d[d['year'] > TRAIN_END_YEAR]
    print(f'学習 {len(tr):,} 行 / 検証 {len(te):,} 行')

    models, aucs = {}, {}
    for name, feats in [('model_no_odds', FEATURES_NO_ODDS),
                        ('model_with_odds', FEATURES_WITH_ODDS)]:
        X_tr, X_val, y_tr, y_val = train_test_split(
            tr[feats], tr['is_top3'], test_size=0.2, random_state=42)
        m = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
             'num_leaves': 63, 'verbose': -1, 'seed': 42},
            lgb.Dataset(X_tr, y_tr), num_boost_round=2000,
            valid_sets=[lgb.Dataset(X_tr, y_tr), lgb.Dataset(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
        auc = roc_auc_score(te['is_top3'], m.predict(te[feats]))
        print(f'  {name}: テストAUC {auc:.4f} (best_iter {m.best_iteration})')
        models[name], aucs[name] = m, auc

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        with open(config.MODEL_DIR / f'{name}.pkl', 'wb') as f:
            pickle.dump(m, f)

    # 買い目用のモデル。ステージ4【1】と同じ作りで、ステージ5はこれで採点する。
    # 上の2つは特徴量が少なく、買い目に使うと回収率が市場だけと変わらない
    from .features import C04_CONFIG
    df4 = build_features(races, results, C04_CONFIG)
    f_strat = strategy_feature_list(df4.columns)
    X4 = model_input(df4, f_strat)
    is_tr = (df4['year'] <= TRAIN_END_YEAR).to_numpy()
    m4 = lgb.train(STRATEGY_PARAMS, lgb.Dataset(X4[is_tr], df4.loc[is_tr, 'is_top3']),
                   num_boost_round=STRATEGY_ROUNDS)
    aucs['model_strategy'] = roc_auc_score(df4.loc[~is_tr, 'is_top3'],
                                           m4.predict(X4[~is_tr]))
    print(f"  model_strategy: テストAUC {aucs['model_strategy']:.4f}"
          f'（買い目用。{len(f_strat)}特徴量。ステージ5が使う）')
    with open(config.MODEL_DIR / 'model_strategy.pkl', 'wb') as f:
        pickle.dump({'model': m4, 'features': f_strat, 'config': 'C04',
                     'train_end_year': TRAIN_END_YEAR}, f)
    del df4, X4

    # 最新評価値のスナップショット。同一日に複数レースがある場合に拾う行が
    # 定まるよう race_id まで含めて並べる（旧実装は行順依存だった）。
    order = ['race_date', 'race_id'] if 'race_date' in df.columns else ['date', 'race_id']
    snap = df.sort_values(order)
    master_dir = config.DATA_DIR / 'master'
    master_dir.mkdir(parents=True, exist_ok=True)
    snap.groupby('horse_id').tail(1)[
        ['horse_id', 'horse_name', 'horse_expected_top3_rate', 'prev_finish',
         'prev_last_3f', 'prev_odds', 'prev_popularity']].to_csv(
        master_dir / 'horse.csv', index=False)
    snap.groupby('jockey_id').tail(1)[
        ['jockey_id', 'jockey_name', 'jockey_added_value']].to_csv(
        master_dir / 'jockey.csv', index=False)
    snap.groupby('trainer_id').tail(1)[
        ['trainer_id', 'trainer_name', 'trainer_added_value']].to_csv(
        master_dir / 'trainer.csv', index=False)
    print(f'モデルとマスタを保存しました')

    if push:
        gitpush.push(['models', 'data/master'],
                     f"Update models (AUC {aucs['model_with_odds']:.4f})")

    return {'stage': 'train', 'ok': True, 'auc': aucs,
            'n_train': len(tr), 'n_test': len(te)}


# ---------------------------------------------------------------------------
# 4. 予想（戦略評価）
# ---------------------------------------------------------------------------
def strategy(push: bool = False, train_end_year: int = TRAIN_END_YEAR) -> Dict:
    """買い目戦略を3段階で評価する。

    各段階で測る指標が違う。混ぜると判断を誤る。

      1. 候補のスコアリング（市場ありモデル）     … 的中率で測る
      2. 市場と独立した見方との突き合わせ（市場なし）… 回収率で測る
      3. 除外と買い方                             … 回収率と参加率で測る

    ## なぜこの順番か

    当初は「市場なしモデルで候補6頭に絞ってから市場ありで評価する」案を
    検討したが、実測で劣った。市場なしモデル（AUC 0.7527）は市場
    （0.8189）より弱く、絞り込みに使うと情報を捨てるだけになる。

      上位6頭に3着内の3頭すべてを含む割合
        市場ありモデル 49.7%  /  市場なしモデル 42.7%

      端から端までの回収率（検証期間）
        市場ありで直接          96.7%
        市場なしで候補6頭→市場あり 88.2%

    市場なしモデルの価値は「候補を絞ること」ではなく
    **「市場と違う意見を出すこと」**にある。全馬をスコアリングしてから、
    乖離の大きいところを見つけ、過大評価を除外する。
    """
    import numpy as np
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from . import backtest, betting, segments as sg, store
    from .features import build_features, C04_CONFIG

    _rule(STAGE_LABELS['strategy'])

    races, results = store.load_for_features()
    df = build_features(races, results, C04_CONFIG)
    payouts = store.read_table('payouts')
    payouts['race_id'] = payouts['race_id'].astype(str)
    df['race_id'] = df['race_id'].astype(str)

    f_market = strategy_feature_list(df.columns)
    f_free = [c for c in dict.fromkeys(MARKET_FREE_FEATURES) if c in df.columns]

    # 除外の判定は欠損のままの値で行う。0 で埋めた後だと、過去走の無い馬が
    # 「2走連続3着内」（0 ≦ 3）に当たってしまう。以前はその偶然で除外が
    # 効いていたので、同じ結果になるよう thin_record として明示してある
    df['_over'] = sg.overvalued(df)
    df['_thin'] = sg.thin_record(df)

    for c in set(f_market + f_free):
        df[c] = pd.to_numeric(df[c], errors='coerce').replace(
            [np.inf, -np.inf], np.nan).fillna(0)
    tr = df[df['year'] <= train_end_year]
    te = df[df['year'] > train_end_year].copy()

    params = STRATEGY_PARAMS

    # ---------------- 1. 候補のスコアリング ----------------
    print('【1】候補のスコアリング（市場ありモデル）— 的中率で測る')
    m_market = lgb.train(params, lgb.Dataset(tr[f_market], tr['is_top3']),
                         num_boost_round=STRATEGY_ROUNDS)
    te['score'] = m_market.predict(te[f_market])
    auc = roc_auc_score(te['is_top3'], te['score'])
    print(f'  AUC {auc:.4f}   学習 {len(tr):,} / 検証 {len(te):,} 行')

    cover = []
    for n in (4, 5, 6, 7):
        ok = tot = 0
        for _, g in te.groupby('race_id'):
            actual = set(g.loc[g['finish_position'] <= 3, 'horse_number'])
            if len(actual) < 3 or len(g) < 8:
                continue
            tot += 1
            ok += actual <= set(g.nlargest(n, 'score')['horse_number'])
        cover.append({'上位N頭': n, '3頭すべてを含む割合%': round(100 * ok / tot, 1)})
    print(pd.DataFrame(cover).to_string(index=False))

    # ---------------- 2. 市場と独立した見方 ----------------
    print('\n【2】市場と独立した見方との突き合わせ（市場なしモデル）— 回収率で測る')
    m_free = lgb.train(params, lgb.Dataset(tr[f_free], tr['is_top3']),
                       num_boost_round=STRATEGY_ROUNDS)
    te['score_free'] = m_free.predict(te[f_free])
    print(f'  AUC {roc_auc_score(te["is_top3"], te["score_free"]):.4f}'
          f'（市場ありより低いのは当然。絞り込みには使わない）')

    # 乖離。市場ありモデルより高く見ている馬＝市場が見落としている候補
    te['divergence'] = te['score_free'] - te['score']
    place = sg.attach_place_payout(te, payouts)
    if not place.empty:
        rows = []
        for q, lab in [(0.0, '全頭'), (0.75, '乖離 上位25%'), (0.9, '乖離 上位10%')]:
            s = place[place['divergence'] >= place['divergence'].quantile(q)]
            rows.append({'条件': lab, '頭数': len(s),
                         '複勝回収率%': round(s['fuku'].mean(), 1)})
        print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- 3. 除外と買い方 ----------------
    print('\n【3】除外と買い方 — 回収率と参加率で測る')
    ranked = te.sort_values('score', ascending=False)
    ranked['_exclude'] = ranked['_over'] | ranked['_thin']
    top3 = ranked.groupby('race_id').head(3).groupby('race_id')
    skip = set(top3['_exclude'].sum().pipe(lambda x: x[x > 0]).index)
    any_over = top3['_over'].sum() > 0
    any_thin = top3['_thin'].sum() > 0
    n_over, n_thin = int(any_over.sum()), int(any_thin.sum())
    # 見送りの理由はステージ5と同じ書き方にする（過大評価を優先）
    reason = {rid: '過大評価の馬が上位にいる' for rid in any_over[any_over].index}
    for rid in any_thin[any_thin].index:
        reason.setdefault(rid, '実績の薄い馬が上位にいる（デビュー馬・1走のみ）')
    print(f'  上位3頭に除外対象の馬がいるレース: {len(skip):,}'
          f'（過大評価 {n_over:,} / 実績が薄い {n_thin:,}、重複あり）')

    # 券種を混ぜて買うので、払戻も券種ごとに引く
    idx = betting.payout_index(payouts)

    plans = [betting.plan_race(rid, g['horse_number'].astype(int).tolist(),
                               g['score'].tolist(), skip=reason.get(rid, ''))
             for rid, g in te.groupby('race_id')]
    result = betting.evaluate(plans, idx)
    print(pd.DataFrame([result]).to_string(index=False))

    reasons = pd.Series([p.reason for p in plans if not p.shape]).value_counts()
    if len(reasons):
        print('\n  買わなかった理由:')
        print('    ' + reasons.to_string().replace('\n', '\n    '))
    shapes = pd.Series([f'{p.bet_type} {p.shape}' for p in plans if p.shape]).value_counts()
    if len(shapes):
        print('\n  選ばれた買い方:')
        print('    ' + shapes.to_string().replace('\n', '\n    '))

    print('\n  ※ 回収率は信頼区間で判断すること。'
          '参加率を絞るほどレース数が減り、確認は難しくなる。')

    return {'stage': 'strategy', 'ok': True, 'auc': auc,
            'model_market': m_market, 'model_free': m_free,
            'features_market': f_market, 'features_free': f_free,
            'betting': result, 'plans': plans, 'skip_races': skip,
            'feature_df': df, 'test': te, 'payout_index': idx}


# ---------------------------------------------------------------------------
# 5. 予想（今週末）
# ---------------------------------------------------------------------------
_UPCOMING_CACHE: Dict[str, 'pd.DataFrame'] = {}
"""race_id ごとの特徴量の控え。2回目以降の採点を速くする。

`FEATURES_WITH_ODDS` のうちオッズで変わるのは odds と popularity だけで、
残りは過去走から決まる。発走直前に何度も回すとき、重い部分（全履歴の
特徴量計算、60秒ほど）をやり直す必要はない。

レース単位で持つのは、「発走まで90分以内」で絞ると時間とともに対象の
組み合わせが変わるため。計算の重さは対象レース数にほぼよらない
（全履歴を通す部分が支配的）ので、初回に当日の全レースを入れておく。
"""


def _fill_weight_from_last_race(rows: pd.DataFrame,
                                results: pd.DataFrame) -> pd.DataFrame:
    """未発表の馬体重を前走の馬体重で埋める。

    馬体重は発走の約1時間前にならないと出馬表に出ない。欠損のままだと
    特徴量の作成で 0 に埋まるが、学習データに 0 は1件も無い（レース後は
    必ず分かる）。前走の値ならほぼ同じ水準になる。発表されたら
    `score_upcoming` が差し替える。
    """
    known = pd.to_numeric(results['horse_weight'], errors='coerce')
    last = (results.assign(_w=known).dropna(subset=['_w'])
            .sort_values('race_date').groupby('horse_id')['_w'].last())
    w = pd.to_numeric(rows['horse_weight'], errors='coerce')
    return rows.assign(horse_weight=w.fillna(rows['horse_id'].map(last)))


def score_upcoming(cards: List[Dict], use_cache: bool = True,
                    warm: Optional[List[Dict]] = None) -> Dict[str, Dict]:
    """出馬表に 3着内確率のスコアを付け、買わないレースを判定する。

    ステージ4のバックテストと**同じモデル・同じ除外**を使う:

      モデル  model_strategy.pkl（ステージ3が保存。ステージ4【1】と同じ作り）
      除外    上位3頭に overvalued または thin_record の馬がいれば買わない

    出馬表の行を results の末尾に足して `build_features` を通す。学習時の
    特徴量は shift()/expanding() で過去走だけから作られるので、これだけで
    予測用の特徴量になる。馬の紐付けは horse_id なので同名馬で取り違えない。
    この方式が通常のバッチ計算と一致することは実測で確認してある。

    新馬戦は全頭がデビュー馬なので、thin_record で必ず見送りになる。

    Args:
        cards: 採点したい出馬表（オッズ入り）
        use_cache: False なら控えを使わず計算し直す
        warm: 一緒に特徴量を作っておく出馬表（オッズ不要）。
            次に呼ばれたとき、ここに含まれるレースは計算が要らない。

    Returns:
        race_id -> {'horse_number': [...], 'score': [...], 'skip': str}
        skip は買わない理由（買うなら空文字）。
    """
    import pickle
    from . import config, store, segments, shutuba
    from .features import build_features, C04_CONFIG

    path = config.MODEL_DIR / 'model_strategy.pkl'
    if not path.exists():
        # 市場オッズで代用すると回収率が控除率そのまま（79%）になるので、
        # 黙って代用せず止める
        raise FileNotFoundError(
            f'{path} がありません。先にステージ3（学習）を回してください。')
    with open(path, 'rb') as f:
        bundle = pickle.load(f)

    if not use_cache:
        _UPCOMING_CACHE.clear()
    want = {str(c['race_id']) for c in cards}
    pool = {str(c['race_id']): c for c in (warm or [])}
    pool.update({str(c['race_id']): c for c in cards})
    todo = [c for rid, c in pool.items() if rid not in _UPCOMING_CACHE]

    if todo and want - set(_UPCOMING_CACHE):
        rows = shutuba.to_result_rows(todo)
        if not rows.empty:
            races, results = store.load_for_features()
            rows = _fill_weight_from_last_race(rows, results)
            df = build_features(races, pd.concat([results, rows], ignore_index=True),
                                C04_CONFIG)
            new = df[df['race_id'].astype(str).isin(rows['race_id'].astype(str))].copy()
            # 除外の判定は過去走だけで決まる。欠損のままの値で一度だけ出す
            new['_over'] = segments.overvalued(new)
            new['_thin'] = segments.thin_record(new)
            for rid in rows['race_id'].astype(str).unique():
                _UPCOMING_CACHE[rid] = new[new['race_id'].astype(str) == rid]

    parts = [_UPCOMING_CACHE[r] for r in want if r in _UPCOMING_CACHE]
    up = pd.concat(parts) if parts else pd.DataFrame()
    if up.empty:
        return {}

    # オッズは発走まで動く。毎回入れ直し、オッズ由来の特徴量も作り直す。
    # 馬体重も発走の約1時間前に発表されるので、出ていれば差し替える
    fresh = {(str(c['race_id']), h.get('horse_number')): h
             for c in cards for h in c.get('horses', [])}
    pairs = list(zip(up['race_id'].astype(str), up['horse_number']))
    announced = pd.to_numeric(pd.Series(
        [fresh.get(k, {}).get('horse_weight') for k in pairs], index=up.index),
        errors='coerce')
    up = up.assign(horse_weight=announced.fillna(up['horse_weight']))
    up = up.assign(
        odds=pd.to_numeric(pd.Series([fresh.get(k, {}).get('odds') for k in pairs],
                                     index=up.index), errors='coerce'),
        popularity=pd.to_numeric(pd.Series([fresh.get(k, {}).get('popularity') for k in pairs],
                                           index=up.index), errors='coerce'))
    up = up[up['odds'].notna() & (up['odds'] > 0)]
    if up.empty:
        return {}
    from .features import add_market_features
    up = add_market_features(up.copy(), C04_CONFIG)

    up = up.assign(score=bundle['model'].predict(model_input(up, bundle['features'])))

    out = {}
    for rid, g in up.groupby('race_id'):
        g = g.sort_values('score', ascending=False)
        top = g.head(3)
        skip = ('過大評価の馬が上位にいる' if top['_over'].any()
                else '実績の薄い馬が上位にいる（デビュー馬・1走のみ）' if top['_thin'].any()
                else '')
        out[str(rid)] = {
            'horse_number': g['horse_number'].astype(int).tolist(),
            'score': g['score'].to_numpy(float),
            'skip': skip,
        }
    return out


# ---------------------------------------------------------------------------
def predict(push: bool = False, days_ahead: int = 10,
            save_json: bool = True, within_minutes: Optional[int] = None,
            race_ids: Optional[List[str]] = None) -> Dict:
    """今週末の出馬表を取ってきて、買い目を出す。

    JRA公式（jra.go.jp）は POST にトークンを渡す方式でURLを直接叩けないため、
    netkeiba から race_id で取得する。詳細は docs/SHUTUBA.md。

    **オッズは発走直前まで動く。** 直前に取るほど市場の評価を正しく反映するので、
    発走が近いレースだけを対象にして何度も回せるようにしてある。

    Args:
        within_minutes: 発走までこの分数以内のレースだけを対象にする。
            None なら期間内すべて。直前運用では 60〜90 あたり。
        race_ids: レースを直接指定する。オッズだけ取り直したいときに使う。
        days_ahead: 何日先まで見るか。

    スコアはステージ3の学習済みモデル（model_with_odds）で出し、
    上位3頭に過大評価の馬がいるレースは買わない。バックテストと同じ方式。

    **市場オッズだけで代用してはいけない。** 単勝オッズ → Harville で
    スコアを作ると、履歴データで購入率90%・回収率79.0%（控除率そのまま）。
    閾値を上げても 85.1% が頭打ちだった。

    1回目は全履歴から特徴量を作るので60秒ほどかかる。2回目以降は
    オッズを入れ直すだけなので一瞬で終わる（`_UPCOMING_CACHE`）。

    新馬は学習対象外（過去走が無い）なので買わない。
    """
    import pickle
    from . import betting, config, fetch, shutuba

    _rule(STAGE_LABELS['predict'])

    fetcher = fetch.Fetcher()

    # race_id を指定して取り直すとき、出馬表には開催日が載っていない。
    # 1回目の絞り込みで分かった日付を覚えておき、2回目の card に書き戻す。
    # これが無いと minutes_to_post が None を返し、発走までの分数が出ない。
    race_days: Dict[str, str] = {}
    all_cards: List[Dict] = []     # 絞り込み前の全レース。採点の控えを作るのに使う

    if race_ids is None:
        days = shutuba.upcoming_race_days(fetcher, days_ahead=days_ahead)
        if not days:
            print(f'  今後 {days_ahead} 日に開催はありません')
            return {'stage': 'predict', 'ok': True, 'races': 0, 'plans': []}
        print('  開催日: ' + ', '.join(f'{d:%m/%d}' for d in days))

        if within_minutes is not None:
            # 発走時刻を知るには出馬表が要るので、まずオッズ無しで軽く取る
            print(f'  発走まで {within_minutes} 分以内のレースに絞ります')
            race_ids = []
            for day in days:
                for rid in shutuba.race_ids_on(day, fetcher):
                    card = shutuba.fetch_race_card(rid, fetcher, with_odds=False)
                    if not card:
                        continue
                    card['date'] = day.isoformat()
                    race_days[rid] = card['date']
                    all_cards.append(card)
                    left = shutuba.minutes_to_post(card)
                    if left is not None and 0 <= left <= within_minutes:
                        race_ids.append(rid)
            print(f'  対象 {len(race_ids)} レース')
            if not race_ids:
                print('  該当なし。発走がもっと近づいてから実行してください')
                return {'stage': 'predict', 'ok': True, 'races': 0, 'plans': []}

    def on_race(day, race_id, card):
        if card:
            if not card.get('date'):
                card['date'] = race_days.get(race_id, '')
            mark = '○' if card.get('odds_available') else '×'
            hs = card.get('horses', [])
            # 馬体重は発走の約1時間前に出る。未発表なら前走の値で代用するが、
            # 履歴で測ると回収率が約2pt下がる（93.6% → 91.0%）
            w = sum(1 for h in hs if h.get('horse_weight'))
            wmark = '○' if hs and w == len(hs) else ('×' if w == 0 else '△')
            left = shutuba.minutes_to_post(card)
            when = f'発走まで{left:>4}分' if left is not None else ''
            print(f'    {card["venue_name"]} {card["race_num"]:>2}R '
                  f'{(card["race_name"] or "")[:16]:18s} オッズ{mark} 馬体重{wmark} {when}',
                  flush=True)

    cards = shutuba.fetch_weekend(fetcher, days_ahead=days_ahead,
                                  on_race=on_race, race_ids=race_ids)
    print(f'  取得 {len(cards)} レース')
    stamps = {c.get('odds_updated_at') for c in cards if c.get('odds_updated_at')}
    if stamps:
        print(f'  オッズ時点: {sorted(stamps)[-1]}'
              f'（{cards[0].get("odds_status") or "不明"}）')

    if save_json and cards:
        out = config.DATA_DIR / 'shutuba'
        out.mkdir(parents=True, exist_ok=True)
        day = cards[0].get('date') or dt.date.today().isoformat()
        path = out / f'{day}_weekend.json'
        path.write_text(shutuba.to_json(cards), encoding='utf-8')
        print(f'  保存: {path}')

    # オッズが出ているレースだけ買い目を出す。
    #
    # スコアは「3着内確率」でなければならない。確信度の閾値（1.4）は
    # 上位3頭の3着内確率の合計で較正してある。1/オッズ をそのまま使うと
    # 尺度が違い（全馬の合計が約1.25）、常に閾値を下回ってしまう。
    #
    # **市場オッズだけでは戦略にならない。** 単勝オッズ → Harville で
    # スコアを作ると、履歴データで購入率90%・回収率79.0%、ほぼ控除率
    # そのままだった。閾値を 1.8 まで上げても 85.1% が頭打ちで、
    # 市場だけで市場に勝つことはできない。バックテストの 94.1% は
    # 「学習済みモデルのスコア」と「過大評価の除外」の両方があって
    # 成立する数字なので、ここでも同じものを使う。
    #
    # 学習時の特徴量は shift()/expanding() で過去走だけから作られるので、
    # 出馬表を results の末尾に足して build_features を通せば、そのまま
    # 予測用の特徴量になる（馬の紐付けは horse_id）。
    t0 = dt.datetime.now()
    scored = score_upcoming(cards, warm=all_cards)
    took = (dt.datetime.now() - t0).total_seconds()
    print(f'  採点 {len(scored)} レース（{took:.0f}秒'
          + ('。2回目以降はオッズの入れ直しだけで済みます' if took > 5 else '') + '）')

    plans = []
    no_odds_races = 0
    for card in cards:
        hs = [h for h in card['horses'] if h.get('odds')]
        if len(hs) < 3:
            no_odds_races += 1
            continue
        g = scored.get(str(card['race_id']))
        if g is None:
            # モデルで出せなかったレースは買わない。市場だけで代用すると
            # 人気馬をなぞるだけになり、控除率ぶん負ける
            plans.append((card, betting.Plan(
                race_id=card['race_id'], confidence=0.0, shape=None, combos=set(),
                reason='モデルでスコアを出せない')))
            continue
        plans.append((card, betting.plan_race(
            card['race_id'], g['horse_number'], g['score'], skip=g['skip'])))

    plans.sort(key=lambda cp: shutuba.minutes_to_post(cp[0]) or 99999)

    bought = [(c, p) for c, p in plans if p.shape]
    print(f'\n  買い目を出せたレース: {len(bought)} / {len(plans)}')
    if bought:
        total = sum(p.cost for _, p in bought)
        kinds = pd.Series([p.bet_type for _, p in bought]).value_counts().to_dict()
        print(f'  合計 {total:,} 円 / {sum(p.points for _, p in bought)} 点  {kinds}')
    for card, plan in bought:
        combos = sorted(tuple(sorted(c)) for c in plan.combos)
        left = shutuba.minutes_to_post(card)
        print(f'    [{card["start_time"]}] {card["venue_name"]} {card["race_num"]:>2}R '
              f'{(card["race_name"] or "")[:14]:16s} 確信度{plan.confidence:.2f} '
              f'{plan.bet_type} {plan.shape}（{plan.points}点）'
              + (f'  発走まで{left}分' if left is not None else ''))
        print(f'      {combos}')

    skipped = pd.Series([p.reason for _, p in plans if not p.shape]).value_counts()
    if len(skipped):
        print('\n  見送り:')
        print('    ' + skipped.to_string().replace('\n', '\n    '))

    no_odds = no_odds_races
    if no_odds:
        print(f'\n  ※ オッズ未発表 {no_odds} レース。'
              f'発走が近づいてから再実行してください')

    if push and save_json and cards:
        from . import gitpush
        gitpush.push(['data/shutuba'], f'Add race cards for {cards[0]["date"]}')

    return {'stage': 'predict', 'ok': True, 'races': len(cards),
            'cards': cards, 'plans': plans, 'bought': bought}
