"""買い目用モデルにスピード指数を加えると、的中率・回収率はどう変わるか。

    python scripts/figure_model.py [出力CSV]

2024年までで学習し、2025年・2026年で検証する。的中率を先に、回収率を後に見る。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from keiba import betting as bt, figure, segments as sg, stages, store
from keiba.features import build_features, C04_CONFIG

t0 = time.time()
races, results = store.load_for_features()
df = build_features(races, results, C04_CONFIG)
df['race_id'] = df['race_id'].astype(str)
fig = figure.add_horse_figure_features(figure.add_figures(results, races))
fig['race_id'] = fig['race_id'].astype(str)
FIG = [c for c in fig.columns if c.startswith(('fig_', 'l3fig_'))]
df = df.merge(fig[['race_id', 'horse_number'] + FIG], on=['race_id', 'horse_number'], how='left')
df['_over'] = sg.overvalued(df)
df['_thin'] = sg.thin_record(df)
print(f'データ {len(df):,} 行  指数の特徴量 {len(FIG)} 個  {time.time() - t0:.0f}秒', flush=True)

F0 = stages.strategy_feature_list(df.columns)
F1 = F0 + FIG
tr, te = df[df['year'] <= 2024], df[df['year'] >= 2025].copy()
pay = store.read_table('payouts'); pay['race_id'] = pay['race_id'].astype(str)
IDX = bt.payout_index(pay)
WIN = {r: set(h) for r, h in zip(pay[pay['bet_type'] == '単勝']['race_id'],
                                  pay[pay['bet_type'] == '単勝']['horse_numbers'])}

def hit_metrics(score, year):
    t = te[te['year'] == year].assign(score=score[te['year'].to_numpy() == year])
    m = dict(win1=[], top3_1=[], wide=[], umaren=[], trio=[], c4=[], c5=[], c6=[])
    for rid, g in t.groupby('race_id'):
        if len(g) < 8:
            continue
        g = g.sort_values('score', ascending=False)
        o = g['horse_number'].astype(int).tolist()
        actual = set(g.loc[g['fin'] <= 3, 'horse_number'].astype(int))
        if len(actual) < 3:
            continue
        m['win1'].append(int(g['fin'].iloc[0] == 1))
        m['top3_1'].append(int(o[0] in actual))
        m['wide'].append(int({o[0], o[1]} <= actual))
        m['umaren'].append(int(set(g.loc[g['fin'] <= 2, 'horse_number'].astype(int)) == {o[0], o[1]}))
        m['trio'].append(int(set(o[:3]) == actual))
        for k in (4, 5, 6):
            m[f'c{k}'].append(int(actual <= set(o[:k])))
    return {k: np.mean(v) * 100 for k, v in m.items()}

def roi(score, year, lo=1.4, sw=1.55):
    t = te[te['year'] == year].assign(score=score[te['year'].to_numpy() == year])
    plans = []
    for rid, g in t.groupby('race_id'):
        g = g.sort_values('score', ascending=False); top = g.head(3)
        skip = '過大評価' if top['_over'].any() else ('薄い' if top['_thin'].any() else '')
        plans.append(bt.plan_race(rid, g['horse_number'].astype(int).tolist(), g['score'].tolist(),
                                  skip=skip, min_confidence=lo,
                                  rules=((sw, '馬連', '軸1頭→相手3頭'), (lo, '三連複', '3頭BOX'))))
    return bt.evaluate(plans, IDX)

te['fin'] = pd.to_numeric(te['finish_position'], errors='coerce')
rows = []
for name, feats in (('今のモデル', F0), ('＋スピード指数', F1)):
    m = lgb.train(stages.STRATEGY_PARAMS, lgb.Dataset(stages.model_input(tr, feats), tr['is_top3']),
                  num_boost_round=stages.STRATEGY_ROUNDS)
    s = m.predict(stages.model_input(te, feats))
    print(f'\n■ {name}（{len(feats)}特徴量）  学習まで {time.time() - t0:.0f}秒', flush=True)
    for y in (2025, 2026):
        mask = te['year'].to_numpy() == y
        h = hit_metrics(s, y)
        print(f"  {y} AUC {roc_auc_score(te['is_top3'][mask], s[mask]):.4f}  "
              f"1位の勝率 {h['win1']:.1f}%  1位の3着内 {h['top3_1']:.1f}%  "
              f"ワイド上位2頭 {h['wide']:.1f}%  馬連上位2頭 {h['umaren']:.1f}%  三連複上位3頭 {h['trio']:.1f}%  "
              f"上位4/5/6頭に3着内全員 {h['c4']:.1f}/{h['c5']:.1f}/{h['c6']:.1f}%", flush=True)
        rows.append(dict(model=name, year=y, auc=roc_auc_score(te['is_top3'][mask], s[mask]), **h))
    grid = [(lo, sw) for lo in (1.3, 1.35, 1.4, 1.45, 1.5) for sw in (1.5, 1.55, 1.6, 1.7) if sw > lo]
    best = max(((roi(s, 2025, lo, sw)['回収率%'], lo, sw) for lo, sw in grid
                if roi(s, 2025, lo, sw)['購入レース'] >= 300))
    for lab, (lo, sw) in (('今の閾値 1.4/1.55', (1.4, 1.55)), (f'2025で決めた {best[1]}/{best[2]}', best[1:])):
        a, b = roi(s, 2025, lo, sw), roi(s, 2026, lo, sw)
        print(f"  回収率 {lab}: 2025 {a['回収率%']}%（的中{a['的中率%']}%・{a['購入レース']}R）  "
              f"2026 {b['回収率%']}%（的中{b['的中率%']}%・{b['購入レース']}R）[{b.get('95%区間')}]", flush=True)
    imp = pd.Series(m.feature_importance('gain'), index=feats).sort_values(ascending=False)
    print('  重要度 上位10:', ', '.join(f'{k}' for k in imp.index[:10]))

out = sys.argv[1] if len(sys.argv) > 1 else 'figure_model.csv'
pd.DataFrame(rows).to_csv(out, index=False)
print(f'\n合計 {time.time() - t0:.0f}秒')
