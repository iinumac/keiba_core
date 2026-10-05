"""複勝支持率を買い目用モデルに入れると、的中率・回収率はどう変わるか。

    python scripts/place_model.py

複勝オッズがあるのは2022年以降なので、比べる2つのモデルはどちらも
2022〜2024年で学習し、2025〜2026年で評価する。
"""

import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from keiba import betting as bt, market, segments as sg, stages, store
from keiba.features import build_features, C04_CONFIG

t0 = time.time()
o = store.read_table('odds')
o['race_id'] = o['race_id'].astype(str)
for c in ('place_min', 'place_max'):
    o[c] = pd.to_numeric(o[c], errors='coerce')

races, results = store.load_for_features()
df = build_features(races, results, C04_CONFIG)
df['race_id'] = df['race_id'].astype(str)
df = df[df['year'] >= 2022].merge(o[['race_id', 'horse_number', 'place_min', 'place_max']],
                                  on=['race_id', 'horse_number'], how='left')
df['_over'] = sg.overvalued(df); df['_thin'] = sg.thin_record(df)
odds = pd.to_numeric(df['odds'], errors='coerce')
inv = 1 / odds
df['win_support'] = inv / inv.groupby(df['race_id']).transform('sum')
df['n_eff'] = 1 / (df['win_support'] ** 2).groupby(df['race_id']).transform('sum')
pi = 1 / np.sqrt(df['place_min'] * df['place_max'])
df['place_support'] = (3 * pi / pi.groupby(df['race_id']).transform('sum')).clip(1e-4, .99)
wf = np.full(len(df), np.nan)
for idx in df.groupby('race_id').indices.values():
    o_ = odds.iloc[idx].to_numpy(float)
    if np.isfinite(o_).all() and (o_ > 0).all():
        wf[idx] = market.top3_from_support(o_)
df['win_formula'] = wf
lg = lambda p: np.log(np.clip(p, 1e-4, .9999) / (1 - np.clip(p, 1e-4, .9999)))
ok = df['place_support'].notna() & df['win_formula'].notna()
Xc = lambda d: np.column_stack([lg(d['win_formula']), lg(d['place_support']), np.log(d['n_eff']),
                                np.log(pd.to_numeric(d['popularity'], errors='coerce').clip(1)),
                                lg(d['place_support']) * np.log(d['n_eff'])])
tr_c = df[ok & (df['year'] <= 2024)]
comb = LogisticRegression(max_iter=1000).fit(Xc(tr_c), tr_c['is_top3'])
df['market_top3'] = np.nan
df.loc[ok, 'market_top3'] = comb.predict_proba(Xc(df[ok]))[:, 1]
df['pop_rank'] = pd.to_numeric(df['popularity'], errors='coerce')
print(f'複勝オッズあり {ok.mean() * 100:.1f}% / {len(df):,}行  {time.time() - t0:.0f}秒', flush=True)

# 市場の3着内見込み：今の式 / 複勝支持率 / 組合せ（2022〜2024で作り、2025〜2026で評価）
ev = df[ok & (df['year'] >= 2025)].copy()
srt = ev.sort_values(['race_id', 'win_support'], ascending=[True, False])
srt['k'] = srt.groupby('race_id').cumcount() + 1
P = srt[srt['k'] <= 4].pivot(index='race_id', columns='k', values='win_support')
ev['pattern'] = ev['race_id'].map(pd.Series(np.select(
    [P[1] / P[2] >= 2, P[2] / P[3] >= 1.8, P[3] / P[4] >= 1.6], ['1強', '2強', '3強'], '混戦'), index=P.index))
from sklearn.metrics import log_loss, brier_score_loss
print('\n【市場の3着内見込み】2022〜2024で作り、2025〜2026で評価')
for name, col in (('今の式（単勝換算）', 'win_formula'), ('複勝支持率', 'place_support'), ('組合せ＋レースの形', 'market_top3')):
    p_ = ev[col].clip(1e-4, .9999)
    print(f"  {name:16s} 対数損失 {log_loss(ev['is_top3'], p_):.4f}  Brier {brier_score_loss(ev['is_top3'], p_):.4f}  AUC {roc_auc_score(ev['is_top3'], p_):.4f}")
    for pat in ('1強', '2強', '3強', '混戦'):
        y_ = ev[ev['pattern'] == pat]
        print(f"      {pat}（{y_['race_id'].nunique()}R） " + '  '.join(
            f"{k}人気 {(y_[y_['pop_rank'] == k]['is_top3'].mean() - y_[y_['pop_rank'] == k][col].mean()) * 100:+5.1f}"
            for k in (1, 2, 3, 4, 5)), flush=True)

F0 = stages.strategy_feature_list(df.columns)
F1 = F0 + ['place_support', 'win_formula', 'market_top3', 'n_eff', 'pop_rank']
tr, te = df[df['year'] <= 2024], df[df['year'] >= 2025].copy()
te['fin'] = pd.to_numeric(te['finish_position'], errors='coerce')
pay = store.read_table('payouts'); pay['race_id'] = pay['race_id'].astype(str)
IDX = bt.payout_index(pay)


def hits(t):
    m = dict(win1=[], top3_1=[], wide=[], umaren=[], trio=[], c5=[])
    for rid, g in t.groupby('race_id'):
        if len(g) < 8:
            continue
        g = g.sort_values('score', ascending=False)
        o_ = g['horse_number'].astype(int).tolist()
        act = set(g.loc[g['fin'] <= 3, 'horse_number'].astype(int))
        if len(act) < 3:
            continue
        m['win1'].append(g['fin'].iloc[0] == 1); m['top3_1'].append(o_[0] in act)
        m['wide'].append({o_[0], o_[1]} <= act)
        m['umaren'].append(set(g.loc[g['fin'] <= 2, 'horse_number'].astype(int)) == {o_[0], o_[1]})
        m['trio'].append(set(o_[:3]) == act); m['c5'].append(act <= set(o_[:5]))
    return {k: np.mean(v) * 100 for k, v in m.items()}


def roi(t, lo=1.4, sw=1.55):
    plans = []
    for rid, g in t.groupby('race_id'):
        g = g.sort_values('score', ascending=False); top = g.head(3)
        skip = '過大評価' if top['_over'].any() else ('薄い' if top['_thin'].any() else '')
        plans.append(bt.plan_race(rid, g['horse_number'].astype(int).tolist(), g['score'].tolist(), skip=skip,
                                  min_confidence=lo, rules=((sw, '馬連', '軸1頭→相手3頭'), (lo, '三連複', '3頭BOX'))))
    return bt.evaluate(plans, IDX)


for name, feats in (('A 今の特徴量', F0), ('B ＋複勝支持率・組合せの見込み・レースの形', F1)):
    m = lgb.train(stages.STRATEGY_PARAMS, lgb.Dataset(stages.model_input(tr, feats), tr['is_top3']),
                  num_boost_round=stages.STRATEGY_ROUNDS)
    te['score'] = m.predict(stages.model_input(te, feats))
    print(f'\n■ {name}（{len(feats)}特徴量）', flush=True)
    for y in (2025, 2026):
        t = te[te['year'] == y]; h = hits(t); r = roi(t)
        print(f"  {y} AUC {roc_auc_score(t['is_top3'], t['score']):.4f}  1位の勝率 {h['win1']:.1f}%  1位の3着内 {h['top3_1']:.1f}%"
              f"  ワイド上位2頭 {h['wide']:.1f}%  馬連上位2頭 {h['umaren']:.1f}%  三連複上位3頭 {h['trio']:.1f}%"
              f"  上位5頭に3着内全員 {h['c5']:.1f}%  | 今のルール 回収{r['回収率%']}% 的中{r['的中率%']}% {r['購入レース']}R", flush=True)
    imp = pd.Series(m.feature_importance('gain'), index=feats).sort_values(ascending=False)
    print('  重要度 上位8:', ', '.join(imp.index[:8]))
print(f'\n合計 {time.time() - t0:.0f}秒')
