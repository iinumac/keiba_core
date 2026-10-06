"""3着内確率は、確率のまま使うのと、レース内の偏差値にするのと、どちらが実際の3着内に近いか。

    python scripts/prob_vs_deviation.py

市場あり（model_strategy.pkl）と市場なし（model_free.pkl）の両方で比べる。
どちらも2024年までで学習しているので、2025年以降だけを使う。
2025年で「値 → 3着内率」の対応を1本の式（ロジスティック）で当て、2026年で確かめる。
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from keiba import config, stages, store
from keiba.features import build_features, C04_CONFIG

races, results = store.load_for_features()
df = build_features(races, results, C04_CONFIG)
df = df[pd.to_datetime(df['race_date']) >= '2025-01-01'].copy()
df['fin'] = pd.to_numeric(df['finish_position'], errors='coerce')
df = df[df['fin'].notna()]
df['top3'] = (df['fin'] <= 3).astype(int)
df['n'] = df.groupby('race_id')['horse_number'].transform('size')
for name, file in (('市場あり', 'model_strategy.pkl'), ('市場なし', 'model_free.pkl')):
    b = pickle.load(open(config.MODEL_DIR / file, 'rb'))
    p = b['model'].predict(stages.model_input(df, b['features']))
    df[name] = p
    g = df.groupby('race_id')[name]
    df[name + '_偏差値'] = 50 + 10 * (df[name] - g.transform('mean')) / g.transform('std')
    # 1レースで3着内に入るのは3頭なので、合計が3になるよう揃えた確率
    df[name + '_合計3'] = (df[name] * 3 / g.transform('sum')).clip(upper=0.99)
df = df[df['n'] >= 5]
test = pd.to_datetime(df['race_date']) >= '2026-01-01'
print(f"2025年 {df[~test]['race_id'].nunique():,}レース / 2026年 {df[test]['race_id'].nunique():,}レース（検証）\n")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_eval(x_tr, y_tr, x_te, y_te):
    m = LogisticRegression(C=1e6).fit(x_tr, y_tr)
    q = m.predict_proba(x_te)[:, 1]
    return log_loss(y_te, q), brier_score_loss(y_te, q), roc_auc_score(y_te, q)


for name in ('市場あり', '市場なし'):
    tr, te = df[~test], df[test]
    y_tr, y_te = tr['top3'], te['top3']
    dev = name + '_偏差値'
    rows = {
        '確率そのまま（補正なし）': (log_loss(y_te, np.clip(te[name], 1e-4, 1 - 1e-4)),
                         brier_score_loss(y_te, te[name]), roc_auc_score(y_te, te[name])),
        '確率（1本の式で補正）': fit_eval(logit(tr[[name]].to_numpy()), y_tr, logit(te[[name]].to_numpy()), y_te),
        '偏差値（1本の式で確率に）': fit_eval(tr[[dev]], y_tr, te[[dev]], y_te),
        '確率を合計3に揃える': (log_loss(y_te, np.clip(te[name + '_合計3'], 1e-4, 1 - 1e-4)),
                       brier_score_loss(y_te, te[name + '_合計3']), roc_auc_score(y_te, te[name + '_合計3'])),
        '偏差値＋頭数': fit_eval(np.c_[tr[dev], np.log(tr['n'])], y_tr, np.c_[te[dev], np.log(te['n'])], y_te),
    }
    print(f'■ {name}（2026年で検証。対数損失・ブライアは小さいほど、AUCは大きいほど良い）')
    print(pd.DataFrame(rows, index=['対数損失', 'ブライア', 'AUC']).T.round(4).to_string(), '\n')

    # 同じ値の馬が、頭数によって実際どれだけ3着内に来たか
    te = te.assign(頭数=pd.cut(te['n'], [4, 10, 14, 18], labels=['5-10頭', '11-14頭', '15-18頭']))
    te['確率の帯'] = pd.cut(te[name], [0, .1, .2, .3, .4, .5, .6, 1])
    te['合計3の帯'] = pd.cut(te[name + '_合計3'], [0, .1, .2, .3, .4, .5, .6, 1])
    te['偏差値の帯'] = pd.cut(te[dev], [0, 40, 45, 50, 55, 60, 65, 70, 100])
    for col in ('確率の帯', '合計3の帯', '偏差値の帯'):
        t = te.pivot_table(index=col, columns='頭数', values='top3', aggfunc='mean') * 100
        c = te.pivot_table(index=col, columns='頭数', values='top3', aggfunc='size')
        t = t.round(1).astype(str) + '%(' + c.astype(str) + ')'
        sp = (te.pivot_table(index=col, columns='頭数', values='top3', aggfunc='mean') * 100)
        t['頭数による差'] = (sp.max(axis=1) - sp.min(axis=1)).round(1)
        mean_p = te.groupby(col)[name].mean() * 100
        t['確率の平均'] = mean_p.round(1)
        print(f'  {col}ごとの実際の3着内率（頭数別）')
        print(t.to_string(), '\n')
