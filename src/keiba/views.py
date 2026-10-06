"""いろいろな目線で、出走馬の上位1〜3番手を並べる。

最後に馬を選んで組み合わせるのは人。そのための材料として、物差しの違う評価を
横に並べる。どれか1つで決めるより、人気の外にいる馬が見えやすい。

    目線                 何を見ているか
    市場あり             買い目用モデルの3着内確率（オッズ込みの総合評価）
    市場なし             オッズを使わないモデルの3着内確率（能力だけの評価。レース内の合計を3に揃える）
    指数・近3走          馬場・ペースを補正したタイムの直近3走平均（figure）
    指数・最高           同じく過去最高値（能力の天井）
    上がり指数・近3走    上がり3Fを同じく補正したものの直近3走平均
    市場の見込み         単勝・複勝オッズとレースの形から見た、市場の3着内見込み
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
         '市場の見込み', '前走パフォーマンス', '能力', '対戦比較']

_CACHE: Dict[str, object] = {}


def _history():
    """全履歴から作るもの（指数・レースレベルの走歴・対戦の索引）。"""
    if 'hist' in _CACHE:
        return _CACHE['hist']
    from . import figure, h2h, pace, race_level, store
    races, results = store.load_for_features()
    fig = figure.add_figures(results, races).sort_values(['horse_id', 'race_date', 'race_id'])
    fig['horse_id'] = fig['horse_id'].astype(str)
    raw_races, raw_results = store.read_table('races'), store.read_table('results')
    # フラグ用の走ごとの情報
    pc = pace.add_horse_pace_features(results, races)
    pc['horse_id'] = pc['horse_id'].astype(str); pc['race_id'] = pc['race_id'].astype(str)
    t_ = pd.to_numeric(pc['time_seconds'], errors='coerce')
    fin = pd.to_numeric(pc['finish_position'], errors='coerce')
    t5 = t_.where(fin == 5).groupby(pc['race_id']).transform('min')
    info = pd.DataFrame({'horse_id': pc['horse_id'], 'race_id': pc['race_id'], 'race_date': pc['race_date'],
                         'surface': pc['surface'], 'level_score': pc['level_score'], 'fin': fin,
                         'pop': pd.to_numeric(pc['popularity'], errors='coerce'),
                         'spread5': t5 - t_.groupby(pc['race_id']).transform('min'),
                         'pos': pc['pos'], 'l3f_vs_pos': pc['l3f_vs_pos']})
    fig_k = fig.assign(race_id=fig['race_id'].astype(str))[['horse_id', 'race_id', 'figure']]
    info = info.merge(fig_k, on=['horse_id', 'race_id'], how='left').sort_values(['horse_id', 'race_date', 'race_id'])
    out = {'info': info,
           'fig': fig[['horse_id', 'race_date', 'figure', 'l3f_figure']],
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


PERCENT_VIEWS = {'市場あり', '市場なし', '市場の見込み', '対戦比較'}
"""確率として%で表示する目線。それ以外は1000mあたりの秒。"""


def deviation(x: pd.Series) -> pd.Series:
    """偏差値。そのレースの出走馬の中で、平均50・標準偏差1つぶんが10。

    目線ごとに単位が違っても同じ物差しで比べられる。1番手と2番手の
    偏差値の差を見れば、抜けているのか団子なのかが分かる。
    """
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(50.0, index=x.index).where(x.notna())
    return 50 + 10 * (x - x.mean()) / sd


def _fmt(v: str, x) -> str:
    if pd.isna(x):
        return '—'
    return f'{x * 100:.1f}%' if v in PERCENT_VIEWS else f'{x:+.2f}秒'


def _add_flag_inputs(t: pd.DataFrame, card: Dict, info: pd.DataFrame, as_of, hs: Dict) -> None:
    """flags.COLUMNS の値を t に足す（すべてレースより前の情報）。"""
    from . import market, shutuba
    from .parse import classify_race_level
    t['fig_best_rank'] = t['指数・最高'].rank(ascending=False, method='min')
    t['l3fig_r3_rank'] = t['上がり指数・近3走'].rank(ascending=False, method='min')
    t['fig_r3_gap'] = t['指数・近3走'] - t['指数・近3走'].max()
    past = info[info['horse_id'].isin(set(t['horse_id'])) & (info['race_date'] < as_of)]
    last = past.groupby('horse_id').tail(1).set_index('horse_id')
    t['fig_p1'] = t['horse_id'].map(last['figure'])
    t['style'] = t['horse_id'].map(past.groupby('horse_id')['pos'].mean())
    t['p1_l3f_vs_pos'] = t['horse_id'].map(last['l3f_vs_pos'])
    t['days_since_last'] = (as_of - t['horse_id'].map(last['race_date'])).dt.days
    surf = shutuba.SURFACE_TO_WAREHOUSE.get(card.get('surface'), card.get('surface'))
    prev_surf = t['horse_id'].map(last['surface'])
    t['surface_change'] = (prev_surf != surf).astype(float).where(prev_surf.notna())
    prize = shutuba._first_prize(card.get('condition'))
    level = classify_race_level(prize)[1] if prize is not None else np.nan
    prev_level = t['horse_id'].map(last['level_score'])
    t['level_up'] = (level > prev_level).astype(float).where(prev_level.notna() & pd.notna(level))
    t['prev_fin'] = t['horse_id'].map(last['fin'])
    t['p1_gap'] = t['prev_fin'] - t['horse_id'].map(last['pop'])
    t['p1_spread5'] = t['horse_id'].map(last['spread5'])
    if '市場の見込み' in t:
        pmin = np.array([hs[n].get('place_min') for n in t.index], float)
        pmax = np.array([hs[n].get('place_max') for n in t.index], float)
        t['place_vs_win'] = np.log(market.place_support(pmin, pmax)
                                   / np.clip(market.top3_from_support(t['単勝'].to_numpy(float)), 1e-4, 1))


def race_views(card: Dict, as_of=None) -> pd.DataFrame:
    """1レース分。馬ごとに各目線の値と順位を持つ表を返す。

    card は shutuba.fetch_race_card の結果（オッズ入り。date が入っていること）。
    """
    from . import flags, h2h, market, race_level, stages
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
        raw = pd.Series(free['model'].predict(stages.model_input(up.reset_index(), free['features'])),
                        index=up.index).reindex(t.index)
        # 市場なしの確率は頭数を見ていない（少頭数で低く、多頭数で高く出る）。3着内は3頭なので
        # 合計が3になるよう揃えると、頭数によるずれがほぼ消える（2026年で AUC 0.755→0.777。
        # scripts/prob_vs_deviation.py）
        t['市場なし'] = (raw * 3 / raw.sum()).clip(upper=0.99)
        t['印'] = [('過大評価' if up.loc[n, '_over'] else '実績が薄い' if up.loc[n, '_thin'] else '')
                  if n in up.index else '' for n in t.index]

    hist = _history()
    fs = _figure_summary(hist['fig'], list(t['horse_id']), as_of)
    for c in fs.columns:
        t[c] = t['horse_id'].map(fs[c])

    from . import market
    pmin = np.array([hs[n].get('place_min') or np.nan for n in nums], float)
    pmax = np.array([hs[n].get('place_max') or np.nan for n in nums], float)
    pops = np.array([hs[n].get('popularity') or np.nan for n in nums], float)
    if np.isfinite(pmin).all() and np.isfinite(pmax).all() and np.isfinite(pops).all():
        t['市場の見込み'] = market.market_top3(t['単勝'].to_numpy(float), pmin, pmax, pops)

    runs = hist['runs']
    level, ability = race_level.fit(runs, as_of)
    past = runs[(runs['date'] < as_of) & runs['horse_id'].isin(t['horse_id'])]
    prev = past.sort_values('date').groupby('horse_id').tail(1).set_index('horse_id')
    perf = prev['race_id'].map(level) - prev['behind']
    t['前走パフォーマンス'] = t['horse_id'].map(perf)
    t['前走'] = t['horse_id'].map(prev['race_name'].str.replace(r'^第\d+回', '', regex=True))
    t['能力'] = t['horse_id'].map(ability)

    table, _ = h2h.rank(hist['h2h'], list(t['horse_id']), as_of)
    # 比較できる相手が1頭もいない馬（新馬など）は 0.5 になり、同点の1番手が並んでしまう
    table = table[table['known'] > 0]
    t['対戦比較'] = t['horse_id'].map(table.set_index('horse_id')['score'])

    _add_flag_inputs(t, card, hist['info'], as_of, hs)
    fl, pk = flags.flop_flags(t), flags.pickup_flags(t)
    t['凡走フラグ数'], t['ピックアップフラグ数'] = fl.sum(axis=1), pk.sum(axis=1)
    t['凡走フラグ'] = [flags.describe(r, fl) for _, r in t.iterrows()]
    t['ピックアップフラグ'] = [flags.describe(r, pk) for _, r in t.iterrows()]
    base = t['市場の見込み'] if '市場の見込み' in t else pd.Series(
        market.top3_from_support(t['単勝'].to_numpy(float)), index=t.index)
    t['補正スコア'] = flags.calibrated(base, t['ピックアップフラグ数'], t['凡走フラグ数'])

    for v in VIEWS:
        if v in t:
            t[f'{v}_順'] = t[v].rank(ascending=False, method='min')
            t[f'{v}_偏差値'] = deviation(t[v])
    rank_cols = [f'{v}_順' for v in VIEWS if f'{v}_順' in t]
    t['上位の目線の数'] = (t[rank_cols] <= 3).sum(axis=1)
    t['人気の割に評価が高い'] = (t['人気'] >= 4) & (t['上位の目線の数'] > 0)
    return t.drop(columns=['horse_id'])


def top3_table(t: pd.DataFrame) -> pd.DataFrame:
    """目線ごとの1〜3番手。値と偏差値、その目線の平均・標準偏差も付ける。"""
    rows = []
    for v in VIEWS:
        if f'{v}_順' not in t:
            continue
        top = t.sort_values(f'{v}_順').head(3)
        row = {'目線': v, '平均': _fmt(v, t[v].mean()),
               '標準偏差': (f'{t[v].std() * 100:.1f}pt' if v in PERCENT_VIEWS else f'{t[v].std():.2f}秒')}
        for i, (n, r) in enumerate(top.iterrows()):
            row[f'{i + 1}番手'] = (f"{n} {r['馬名']}（{int(r['人気'])}人気）"
                                 f"{_fmt(v, r[v])}・偏差値{r[f'{v}_偏差値']:.0f}")
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(card: Dict, t: pd.DataFrame) -> str:
    """表示用。全馬の一覧・目線ごとのランキング・ピックアップリスト・要注意リスト。"""
    def pct(x):
        return '—' if pd.isna(x) else f'{x * 100:.1f}%'
    head = (f"### [{card.get('start_time')}] {card['venue_name']}{card['race_num']}R "
            f"{card['race_name']}（{card.get('surface')}{card.get('distance')}m・{len(t)}頭）")
    L = [head, '', '**1. 全馬の一覧**（補正スコア順。確率はすべて3着内）', '',
         '| 馬番 | 馬名 | 人気 | 単勝 | 市場あり | 市場なし | 市場の見込み | 補正スコア | ピックアップ | 凡走 | 印 |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for n, r in t.sort_values('補正スコア', ascending=False).iterrows():
        L.append(f"| {n} | {r['馬名']} | {int(r['人気'])} | {r['単勝']} | {pct(r.get('市場あり'))} | {pct(r.get('市場なし'))} | "
                 f"{pct(r.get('市場の見込み'))} | {pct(r['補正スコア'])} | {int(r['ピックアップフラグ数'])} | "
                 f"{int(r['凡走フラグ数'])} | {r['印']} |")
    L += ['', '**2. 目線ごとのランキング**', '',
          '| 目線 | 平均 | 標準偏差 | 1番手 | 2番手 | 3番手 |', '|---|---|---|---|---|---|']
    for _, r in top3_table(t).iterrows():
        L.append(f"| {r['目線']} | {r['平均']} | {r['標準偏差']} | {r.get('1番手', '')} | "
                 f"{r.get('2番手', '')} | {r.get('3番手', '')} |")
    pick = t[(t['ピックアップフラグ数'] >= PICKUP_MIN) | ((t['人気'] >= 4) & (t['ピックアップフラグ数'] >= PICKUP_MIN - 1))]
    L += ['', f'**3. ピックアップ**（フラグ{PICKUP_MIN}個以上、人気4番以下は{PICKUP_MIN - 1}個以上）', '']
    for n, r in pick.sort_values(['ピックアップフラグ数', '人気'], ascending=[False, True]).iterrows():
        L.append(f"- {n} {r['馬名']}（{int(r['人気'])}人気・{int(r['ピックアップフラグ数'])}個）：{r['ピックアップフラグ']}")
    if pick.empty:
        L.append('- 該当なし')
    warn = t[(t['人気'] <= CAUTION_POP) & (t['凡走フラグ数'] >= CAUTION_MIN)]
    L += ['', f'**4. 要注意**（{CAUTION_POP}番人気以内で凡走フラグ{CAUTION_MIN}個以上）', '']
    for n, r in warn.sort_values(['凡走フラグ数', '人気'], ascending=[False, True]).iterrows():
        L.append(f"- {n} {r['馬名']}（{int(r['人気'])}人気・{int(r['凡走フラグ数'])}個）：{r['凡走フラグ']}")
    if warn.empty:
        L.append('- 該当なし')
    return '\n'.join(L)


PICKUP_MIN = 4
"""ピックアップに載せるフラグの数。人気4番以下は1つ少なくてよい（フラグ3個で3着内16%、0個の倍近い）。"""
CAUTION_POP, CAUTION_MIN = 5, 4
"""要注意に載せる人気の範囲とフラグの数（1〜3番人気でフラグ4個なら凡走率35〜37%）。"""
