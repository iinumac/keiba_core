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

STAGES = ['collect', 'build', 'train', 'strategy']
STAGE_LABELS = {
    'collect': '1. 取り込み（HTML収集）',
    'build': '2. パース（ウェアハウス構築）',
    'train': '3. 学習（モデル・マスタ作成）',
    'strategy': '4. 予想（戦略評価）',
}

TRAIN_END_YEAR = 2024

FEATURES_NO_ODDS = [
    'distance', 'surface_encoded', 'level_score', 'impost',
    'horse_expected_top3_rate', 'jockey_added_value', 'trainer_added_value',
    'prev_finish', 'prev_last_3f', 'days_since_last', 'is_jockey_changed',
]
FEATURES_WITH_ODDS = FEATURES_NO_ODDS + ['popularity', 'odds']

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
    print(f"  races {result['races']:,} / results {result['results']:,}")
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
def backtest(te, col: str, n_pick: int, ascending: bool = False,
             min_runners: int = 8) -> Tuple[int, int, float]:
    """3着内の3頭を買い目に含められたレースの数を数える。

    回収率ではなく的中率であることに注意。払戻データがまだ無い。
    """
    hits = races_n = 0
    fields: List[int] = []
    for _, g in te.groupby('race_id'):
        if len(g) < min_runners:
            continue
        actual = set(g[g['finish_position'] <= 3]['horse_number'])
        if len(actual) < 3:
            continue
        races_n += 1
        fields.append(len(g))
        picks = set(g.sort_values(col, ascending=ascending).head(n_pick)['horse_number'])
        if actual <= picks:
            hits += 1
    return races_n, hits, (sum(fields) / len(fields) if fields else 0.0)


def strategy(push: bool = False) -> Dict:
    """三連複の戦略を評価する。

    買い目の選抜は**確率**で行う。「AI複勝率 × 単勝オッズ」を期待値として
    使うのは誤りで（単勝オッズは1着の配当、3着内の配当ではない）、
    それで選ぶとランダムより12倍悪くなることを確認済み。
    """
    import numpy as np
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from . import store
    from .features import build_features, C04_CONFIG

    _rule(STAGE_LABELS['strategy'])

    races, results = store.load_for_features()
    df = build_features(races, results, C04_CONFIG)
    feats = [c for c in STRATEGY_FEATURES if c in df.columns]

    d = df.copy()
    for c in feats:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    tr = d[d['year'] <= TRAIN_END_YEAR]
    te = d[d['year'] > TRAIN_END_YEAR].copy()
    print(f'学習 {len(tr):,} 行 / 検証 {len(te):,} 行')

    params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
              'num_leaves': 63, 'verbose': -1, 'seed': 42}
    model_top3 = lgb.train(params, lgb.Dataset(tr[feats], tr['is_top3']),
                           num_boost_round=400)
    model_win = lgb.train(params, lgb.Dataset(tr[feats], tr['is_win']),
                          num_boost_round=400)
    auc3 = roc_auc_score(te['is_top3'], model_top3.predict(te[feats]))
    aucw = roc_auc_score(te['is_win'], model_win.predict(te[feats]))
    print(f'  3着内モデル AUC {auc3:.4f} / 勝率モデル AUC {aucw:.4f}')

    te['pred_top3'] = model_top3.predict(te[feats])
    te['ev_win_odds'] = te['pred_top3'] * te['odds']   # 参考値。選抜には使わない

    print(f'\n{"選び方":<22}{"点数":>5}{"的中":>7}{"的中率":>9}{"ランダム":>9}{"倍率":>7}')
    rows = []
    for label, col, asc in [('AI複勝確率', 'pred_top3', False),
                            ('人気上位', 'popularity', True),
                            ('AI複勝率×単勝オッズ', 'ev_win_odds', False)]:
        for n in (5, 6, 7):
            races_n, hits, field = backtest(te, col, n, ascending=asc)
            pts = math.comb(n, 3)
            rate = 100 * hits / races_n if races_n else 0
            rnd = 100 * pts / math.comb(round(field), 3)
            rows.append({'label': label, 'n_pick': n, 'points': pts,
                         'hits': hits, 'rate': rate, 'random': rnd})
            print(f'{label + f" {n}頭BOX":<22}{pts:>5}{hits:>7}{rate:>8.2f}%'
                  f'{rnd:>8.2f}%{rate / rnd:>6.1f}x')

    print('\n※ 的中率は点数を増やせば必ず上がる。回収率で評価するには')
    print('   払戻データの取り込みが必要（docs/MIGRATION.md 参照）。')

    return {'stage': 'strategy', 'ok': True, 'auc_top3': auc3, 'auc_win': aucw,
            'backtest': rows, 'model_top3': model_top3, 'model_win': model_win,
            'features': feats, 'feature_df': df}
