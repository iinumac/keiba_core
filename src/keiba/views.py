"""いろいろな目線で、出走馬の上位1〜3番手を並べる。

最後に馬を選んで組み合わせるのは人。そのための材料として、物差しの違う評価を
横に並べる。どれか1つで決めるより、人気の外にいる馬が見えやすい。

    目線                 何を見ているか
    市場あり             買い目用モデルの3着内確率（オッズ込みの総合評価）
    市場なし             オッズを使わないモデルの3着内確率（能力だけの評価）
    指数・近3走          馬場・ペースを補正したタイムの直近3走平均（figure）
    指数・最高           同じく過去最高値（能力の天井）
    上がり指数・近3走    上がり3Fを同じく補正したものの直近3走平均
    複勝支持率           複勝オッズから見た、3着内に来ることへの市場の支持
    前走パフォーマンス   前走のレースレベル − 勝ち馬との差（race_level）
    能力                 過去2年の対戦をつないだ地力（race_level）
    対戦比較             直接・間接の対戦から見た、他の馬に先着する確率の平均（h2h）

人気4番以下なのに、どれかの目線で3番手以内に入った馬を「人気の割に評価が高い」とする。

全履歴を使う計算（指数・レースレベル・対戦比較）は重いので、1回目に作って
プロセス内に控える。判定日ごとに作り直すのはレースレベルの解き直しだけ。
"""

from __future__ import annotations

import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

VIEWS = ['市場あり', '市場なし', '指数・近3走', '指数・最高', '上がり指数・近3走',
         '複勝支持率', '前走パフォーマンス', '能力', '対戦比較']

_CACHE: Dict[str, object] = {}


def _history():
    """全履歴から作るもの（指数・レースレベルの走歴・対戦の索引）。"""
    if 'hist' in _CACHE:
        return _CACHE['hist']
    from . import figure, h2h, race_level, store
    races, results = store.load_for_features()
    fig = figure.add_figures(results, races).sort_values(['horse_id', 'race_date', 'race_id'])
    fig['horse_id'] = fig['horse_id'].astype(str)
    raw_races, raw_results = store.read_table('races'), store.read_table('results')
    out = {'fig': fig[['horse_id', 'race_date', 'figure', 'l3f_figure']],
           'runs': race_level.prepare_runs(raw_races, raw_results),
           'h2h': h2h.History(raw_races, raw_results)}
    _CACHE['hist'] = out
    return out


def _free_model():
    """市場なしモデル。無ければ学習して models/model_free.pkl に保存する。"""
    if 'free' in _CACHE:
        return _CACHE['free']
    from . import config, stages, store
    from .features import build_features, C04_CONFIG
    path = config.MODEL_DIR / 'model_free.pkl'
    if path.exists():
        bundle = pickle.load(open(path, 'rb'))
    else:
        import lightgbm as lgb
        races, results = store.load_for_features()
        df = build_features(races, results, C04_CONFIG)
        feats = [c for c in dict.fromkeys(stages.MARKET_FREE_FEATURES) if c in df.columns]
        tr = df[df['year'] <= stages.TRAIN_END_YEAR]
        m = lgb.train(stages.STRATEGY_PARAMS, lgb.Dataset(stages.model_input(tr, feats), tr['is_top3']),
                      num_boost_round=stages.STRATEGY_ROUNDS)
        bundle = {'model': m, 'features': feats, 'config': 'C04',
                  'train_end_year': stages.TRAIN_END_YEAR}
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        pickle.dump(bundle, open(path, 'wb'))
    _CACHE['free'] = bundle
    return bundle


def _figure_summary(fig: pd.DataFrame, horse_ids: List[str], as_of) -> pd.DataFrame:
    f = fig[fig['horse_id'].isin(horse_ids) & (fig['race_date'] < pd.Timestamp(as_of))]
    g = f.groupby('horse_id')
    return pd.DataFrame({
        '指数・近3走': g['figure'].apply(lambda x: x.dropna().tail(3).mean()),
        '指数・最高': g['figure'].max(),
        '上がり指数・近3走': g['l3f_figure'].apply(lambda x: x.dropna().tail(3).mean())})


def race_views(card: Dict, as_of=None) -> pd.DataFrame:
    """1レース分。馬ごとに各目線の値と順位を持つ表を返す。

    card は shutuba.fetch_race_card の結果（オッズ入り。date が入っていること）。
    """
    from . import h2h, race_level, stages
    as_of = pd.Timestamp(as_of or card['date'])
    hs = {h['horse_number']: h for h in card['horses'] if h.get('odds')}
    if not hs:
        return pd.DataFrame()
    nums = sorted(hs)
    t = pd.DataFrame({'馬番': nums,
                      '馬名': [hs[n]['horse_name'] for n in nums],
                      '騎手': [hs[n].get('jockey_name') for n in nums],
                      '人気': [hs[n].get('popularity') for n in nums],
                      '単勝': [hs[n].get('odds') for n in nums],
                      'horse_id': [str(hs[n].get('horse_id')) for n in nums]}).set_index('馬番')
    rid = str(card['race_id'])

    # 市場あり・市場なし・除外の印（score_upcoming の特徴量を使い回す）
    scored = stages.score_upcoming([card]).get(rid)
    t['印'] = ''
    if scored is not None:
        t['市場あり'] = pd.Series(dict(zip(scored['horse_number'], scored['score'])))
        up = stages._UPCOMING_CACHE[rid].set_index('horse_number')
        free = _free_model()
        t['市場なし'] = pd.Series(free['model'].predict(stages.model_input(up.reset_index(), free['features'])),
                                index=up.index)
        t['印'] = [('過大評価' if up.loc[n, '_over'] else '実績が薄い' if up.loc[n, '_thin'] else '')
                  if n in up.index else '' for n in t.index]

    hist = _history()
    fs = _figure_summary(hist['fig'], list(t['horse_id']), as_of)
    for c in fs.columns:
        t[c] = t['horse_id'].map(fs[c])

    pmin = np.array([hs[n].get('place_min') or np.nan for n in nums], float)
    pmax = np.array([hs[n].get('place_max') or np.nan for n in nums], float)
    inv = 1 / np.sqrt(pmin * pmax)
    if np.isfinite(inv).any():
        t['複勝支持率'] = np.clip(3 * inv / np.nansum(inv), 0, 0.99)

    runs = hist['runs']
    level, ability = race_level.fit(runs, as_of)
    past = runs[(runs['date'] < as_of) & runs['horse_id'].isin(t['horse_id'])]
    prev = past.sort_values('date').groupby('horse_id').tail(1).set_index('horse_id')
    perf = prev['race_id'].map(level) - prev['behind']
    t['前走パフォーマンス'] = t['horse_id'].map(perf)
    t['前走'] = t['horse_id'].map(prev['race_name'].str.replace(r'^第\d+回', '', regex=True))
    t['能力'] = t['horse_id'].map(ability)

    table, _ = h2h.rank(hist['h2h'], list(t['horse_id']), as_of)
    t['対戦比較'] = t['horse_id'].map(table.set_index('horse_id')['score'])

    for v in VIEWS:
        if v in t:
            t[f'{v}_順'] = t[v].rank(ascending=False, method='min')
    rank_cols = [f'{v}_順' for v in VIEWS if f'{v}_順' in t]
    t['上位の目線の数'] = (t[rank_cols] <= 3).sum(axis=1)
    t['人気の割に評価が高い'] = (t['人気'] >= 4) & (t['上位の目線の数'] > 0)
    return t.drop(columns=['horse_id'])


def top3_table(t: pd.DataFrame) -> pd.DataFrame:
    """目線ごとの1〜3番手。"""
    rows = []
    for v in VIEWS:
        if f'{v}_順' not in t:
            continue
        top = t.sort_values(f'{v}_順').head(3)
        rows.append({'目線': v, **{f'{i + 1}番手': f"{n} {r['馬名']}（{int(r['人気'])}人気）"
                                   for i, (n, r) in enumerate(top.iterrows())}})
    return pd.DataFrame(rows)


def to_markdown(card: Dict, t: pd.DataFrame) -> str:
    """表示用。目線ごとの上位3頭と、人気の割に評価が高い馬。"""
    head = (f"### [{card.get('start_time')}] {card['venue_name']}{card['race_num']}R "
            f"{card['race_name']}（{card.get('surface')}{card.get('distance')}m・{len(t)}頭）\n")
    top = top3_table(t)
    lines = [head, '| 目線 | 1番手 | 2番手 | 3番手 |', '|---|---|---|---|']
    for _, r in top.iterrows():
        lines.append(f"| {r['目線']} | {r.get('1番手', '')} | {r.get('2番手', '')} | {r.get('3番手', '')} |")
    many = t[t['上位の目線の数'] >= 3].sort_values('上位の目線の数', ascending=False)
    if len(many):
        lines.append('\n**多くの目線で上位**: ' + '、'.join(
            f"{n} {r['馬名']}（{int(r['人気'])}人気・{int(r['上位の目線の数'])}目線）" for n, r in many.iterrows()))
    sleepers = t[t['人気の割に評価が高い']].sort_values('上位の目線の数', ascending=False)
    if len(sleepers):
        def which(r):
            return '／'.join(v for v in VIEWS if r.get(f'{v}_順', 99) <= 3)
        lines.append('\n**人気の割に評価が高い**: ' + '、'.join(
            f"{n} {r['馬名']}（{int(r['人気'])}人気：{which(r)}）" for n, r in sleepers.iterrows()))
    marks = t[t['印'] != '']
    if len(marks):
        lines.append('\n印: ' + '、'.join(f"{n} {r['馬名']}＝{r['印']}" for n, r in marks.iterrows()))
    return '\n'.join(lines)
