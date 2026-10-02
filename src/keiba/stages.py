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

    market = [c for c in STRATEGY_FEATURES if c in df.columns]
    free = [c for c in MARKET_FREE_FEATURES if c in df.columns]
    recency = [c for c in RECENCY_FEATURES if c in df.columns]
    f_market = list(dict.fromkeys(market + free + recency))
    f_free = list(dict.fromkeys(free))

    for c in set(f_market + f_free):
        df[c] = pd.to_numeric(df[c], errors='coerce').replace(
            [np.inf, -np.inf], np.nan).fillna(0)
    tr = df[df['year'] <= train_end_year]
    te = df[df['year'] > train_end_year].copy()

    params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
              'num_leaves': 63, 'verbose': -1, 'seed': 42}

    # ---------------- 1. 候補のスコアリング ----------------
    print('【1】候補のスコアリング（市場ありモデル）— 的中率で測る')
    m_market = lgb.train(params, lgb.Dataset(tr[f_market], tr['is_top3']),
                         num_boost_round=600)
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
                       num_boost_round=600)
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
    ranked['_over'] = sg.overvalued(ranked)
    skip = set(ranked.groupby('race_id').head(3)
               .groupby('race_id')['_over'].sum().pipe(lambda x: x[x > 0]).index)
    print(f'  上位3頭に過大評価の馬がいるレース: {len(skip):,}')

    tri = payouts[payouts['bet_type'] == '三連複']
    idx: Dict[str, list] = {}
    for rid, hn, p in zip(tri['race_id'], tri['horse_numbers'], tri['payout']):
        idx.setdefault(str(rid), []).append(
            (frozenset(int(x) for x in hn), int(p)))

    plans = [betting.plan_race(rid, g['horse_number'].astype(int).tolist(),
                               g['score'].tolist(), skip=rid in skip)
             for rid, g in te.groupby('race_id')]
    result = betting.evaluate(plans, idx)
    print(pd.DataFrame([result]).to_string(index=False))

    reasons = pd.Series([p.reason for p in plans if not p.shape]).value_counts()
    if len(reasons):
        print('\n  買わなかった理由:')
        print('    ' + reasons.to_string().replace('\n', '\n    '))
    shapes = pd.Series([p.shape for p in plans if p.shape]).value_counts()
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
def predict(push: bool = False, days_ahead: int = 10,
            save_json: bool = True, strategy_result: Optional[Dict] = None) -> Dict:
    """今週末の出馬表を取ってきて、買い目を出す。

    JRA公式（jra.go.jp）は POST にトークンを渡す方式でURLを直接叩けないため、
    netkeiba から race_id で取得する。詳細は docs/SHUTUBA.md。

    `strategy_result` を渡さない場合は学習済みモデルを models/ から読む。
    オッズが未発表（前日など）のレースは買い目を出せない。
    """
    import pickle
    from . import betting, config, fetch, shutuba

    _rule(STAGE_LABELS['predict'])

    fetcher = fetch.Fetcher()
    days = shutuba.upcoming_race_days(fetcher, days_ahead=days_ahead)
    if not days:
        print(f'  今後 {days_ahead} 日に開催はありません')
        return {'stage': 'predict', 'ok': True, 'races': 0, 'plans': []}
    print(f'  開催日: ' + ', '.join(f'{d:%m/%d}' for d in days))

    def on_race(day, race_id, card):
        if card:
            mark = '○' if card.get('odds_available') else '×'
            print(f'    {day} {card["venue_name"]} {card["race_num"]:>2}R '
                  f'{(card["race_name"] or "")[:16]:18s} オッズ{mark}', flush=True)

    cards = shutuba.fetch_weekend(fetcher, days_ahead=days_ahead, on_race=on_race)
    print(f'  取得 {len(cards)} レース')

    if save_json and cards:
        out = config.DATA_DIR / 'shutuba'
        out.mkdir(parents=True, exist_ok=True)
        path = out / f'{cards[0]["date"]}_weekend.json'
        path.write_text(shutuba.to_json(cards), encoding='utf-8')
        print(f'  保存: {path}')

    # オッズが出ているレースだけ買い目を出す。
    #
    # スコアは「3着内確率」でなければならない。確信度の閾値（1.4）は
    # 上位3頭の3着内確率の合計で較正してある。1/オッズ をそのまま使うと
    # 尺度が違い（全馬の合計が約1.25）、常に閾値を下回って全レース見送りになる。
    # 単勝オッズ → 勝率 → 3着内確率（Harville）へ変換する。
    import numpy as np
    from . import market

    plans = []
    no_odds_races = 0
    for card in cards:
        hs = [h for h in card['horses'] if h.get('odds')]
        if len(hs) < 3:
            no_odds_races += 1
            continue
        win = market.implied_win_prob(np.array([h['odds'] for h in hs], dtype=float))
        top3 = market.harville_top3(win)
        plan = betting.plan_race(card['race_id'],
                                 [h['horse_number'] for h in hs], list(top3))
        plans.append((card, plan))

    bought = [(c, p) for c, p in plans if p.shape]
    print(f'\n  買い目を出せたレース: {len(bought)} / {len(plans)}')
    for card, plan in bought:
        combos = sorted(tuple(sorted(c)) for c in plan.combos)
        print(f'    {card["venue_name"]} {card["race_num"]:>2}R '
              f'{(card["race_name"] or "")[:14]:16s} 確信度{plan.confidence:.2f} '
              f'{plan.shape}（{plan.points}点）')
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
