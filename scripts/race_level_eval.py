"""レースレベル（keiba.race_level）が次のレースの着順をどれだけ言い当てるか。

    python scripts/race_level_eval.py [開始日=2025-01-01]

週ごとに、その週の最初の開催日より前のデータだけでレベルを解き直し、
その週のレースの各出走馬に「前走の情報」を付ける。

    前走レベル        前走のレースのレベル
    前走パフォーマンス 前走レベル − 前走の勝ち馬との差（その馬が前走で見せた強さ）
    前走クラス        races.level_score（G1 / G2 / OP … の格）
    前走着順
    前走メンバーのその後  前走の出走馬の次走3着内率（判定日より前に走った分だけ）
    能力              同じ解で得られる馬の地力（前走に限らず過去2年すべて）

同じレースの2頭の組み合わせで、どちらが先着したかを比べる。
- 単独の的中率（各指標で上の馬が先着した割合）
- 「格上のレースで負けた馬」と「格下のレースで好走した馬」がぶつかる組での的中率
- オッズに上乗せした情報があるか（2025年で学習、2026年で評価したロジスティック回帰）
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from keiba import race_level as rl, store

START = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else '2025-01-01')

t0 = time.time()
runs = rl.prepare_runs(store.read_table('races'), store.read_table('results'))
runs = runs.sort_values(['horse_id', 'date'])
for c in ['race_id', 'date', 'behind', 'level_score', 'finish_position']:
    runs['prev_' + c] = runs.groupby('horse_id')[c].shift(1)
runs = runs.sort_values(['date', 'race_id', 'finish_position'])

# 前走メンバーのその後（次走3着内率）を、判定日より前に走った分だけで数えるための準備
nxt = runs.sort_values(['horse_id', 'date'])
nxt = nxt.assign(next_date=nxt.groupby('horse_id')['date'].shift(-1),
                 next_top3=nxt.groupby('horse_id')['finish_position'].shift(-1) <= 3)

test = runs[runs['date'] >= START].copy()
test['week'] = test['date'].dt.to_period('W-SUN')
parts = []
for week, g in test.groupby('week'):
    as_of = g['date'].min()
    level, ability = rl.fit(runs, as_of)
    g = g.assign(prev_level=g['prev_race_id'].map(level), ability=g['horse_id'].map(ability))
    seen = nxt[(nxt['race_id'].isin(set(g['prev_race_id'].dropna())))
               & (nxt['next_date'] < as_of)]
    after = seen.groupby('race_id')['next_top3'].agg(['mean', 'size'])
    after = after[after['size'] >= 3]['mean']
    g['prev_after'] = g['prev_race_id'].map(after)
    parts.append(g)
df = pd.concat(parts)
df['prev_perf'] = df['prev_level'] - df['prev_behind']
df['logodds'] = -np.log(df['odds'])
df = df.dropna(subset=['prev_level', 'odds'])
print(f'出走 {len(df):,}  レース {df["race_id"].nunique():,}  {time.time() - t0:.0f}秒', flush=True)

# 同じレースの2頭の組（前走が別のレースの組だけ）
cols = ['race_id', 'horse_id', 'finish_position', 'prev_race_id', 'prev_level', 'prev_perf',
        'prev_behind', 'prev_level_score', 'prev_finish_position', 'prev_after', 'ability',
        'logodds', 'level_score', 'date']
x = df[cols]
p = x.merge(x, on='race_id', suffixes=('_a', '_b'))
p = p[(p['horse_id_a'] < p['horse_id_b']) & (p['prev_race_id_a'] != p['prev_race_id_b'])]
p['y'] = (p['finish_position_a'] < p['finish_position_b']).astype(int)
for f in ['prev_level', 'prev_perf', 'prev_behind', 'prev_level_score', 'prev_finish_position',
          'prev_after', 'ability', 'logodds']:
    p['d_' + f] = p[f + '_a'] - p[f + '_b']
# 小さいほど良い指標は符号を反転して「大きいほど a が上」にそろえる
p['d_prev_behind'] *= -1
p['d_prev_finish_position'] *= -1
print(f'組 {len(p):,}  {time.time() - t0:.0f}秒\n')

NAMES = {'d_logodds': 'オッズ（人気）', 'd_ability': '能力（過去2年）',
         'd_prev_perf': '前走パフォーマンス', 'd_prev_level': '前走レベル',
         'd_prev_level_score': '前走クラス', 'd_prev_after': '前走メンバーのその後',
         'd_prev_behind': '前走の勝ち馬との差', 'd_prev_finish_position': '前走着順'}


def hit_table(q):
    rows = []
    for f, name in NAMES.items():
        s = q[q[f].notna() & (q[f] != 0)]
        rows.append({'指標': name, '組数': len(s), '的中率': ((s[f] > 0) == (s['y'] == 1)).mean()})
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.3f}')


pd.set_option('display.unicode.east_asian_width', True)
print('■ 全レース: 各指標で上の馬が先着した割合')
print(hit_table(p))

graded = p[p['level_score_a'] >= 4]
print(f'\n■ 重賞（level_score 4以上）だけ  レース {graded["race_id"].nunique()}')
print(hit_table(graded))

# 格上で負けた馬 vs 格下で好走した馬: 前走レベルが高い方が、前走の勝ち馬との差では劣る組
conflict = p[(np.sign(p['d_prev_level']) * np.sign(p['d_prev_behind']) < 0)
             & (p['d_prev_level'].abs() >= 0.1)]
print(f'\n■ 「格上で負けた馬」vs「格下で好走した馬」（前走レベル差0.1以上）  組 {len(conflict):,}')
hi_won = ((conflict['d_prev_level'] > 0) == (conflict['y'] == 1)).mean()
print(f'  前走レベルが高かった方が先着: {hi_won:.1%}')
print(f'  前走パフォーマンスで上の方が先着: '
      f'{((conflict["d_prev_perf"] > 0) == (conflict["y"] == 1)).mean():.1%}')
print(f'  人気が上の方が先着: {((conflict["d_logodds"] > 0) == (conflict["y"] == 1)).mean():.1%}')
cg = conflict[conflict['level_score_a'] >= 4]
print(f'  うち重賞（組 {len(cg):,}）: 前走レベルが高い方 {((cg["d_prev_level"] > 0) == (cg["y"] == 1)).mean():.1%}'
      f'  前走パフォーマンス {((cg["d_prev_perf"] > 0) == (cg["y"] == 1)).mean():.1%}'
      f'  人気 {((cg["d_logodds"] > 0) == (cg["y"] == 1)).mean():.1%}')

print('\n■ オッズに上乗せできるか（2025年で学習 → 2026年で評価）')
q = p.dropna(subset=['d_prev_perf', 'd_ability', 'd_logodds'])
tr, te = q[q['date_a'] < '2026-01-01'], q[q['date_a'] >= '2026-01-01']
MODELS = {
    'オッズのみ': ['d_logodds'],
    '+前走パフォーマンス': ['d_logodds', 'd_prev_perf'],
    '+前走レベル・差': ['d_logodds', 'd_prev_level', 'd_prev_behind'],
    '+能力': ['d_logodds', 'd_ability'],
    '+能力・前走パフォーマンス': ['d_logodds', 'd_ability', 'd_prev_perf'],
}
for name, fs in MODELS.items():
    m = LogisticRegression(fit_intercept=False).fit(tr[fs], tr['y'])
    pr = m.predict_proba(te[fs])[:, 1]
    coef = '  '.join(f'{f[2:]}={c:+.2f}' for f, c in zip(fs, m.coef_[0]))
    print(f'  {name:<16} logloss {log_loss(te["y"], pr):.4f}  AUC {roc_auc_score(te["y"], pr):.4f}  {coef}')
for name, fs in [('前走パフォーマンスのみ', ['d_prev_perf']), ('能力のみ', ['d_ability'])]:
    m = LogisticRegression(fit_intercept=False).fit(tr[fs], tr['y'])
    pr = m.predict_proba(te[fs])[:, 1]
    print(f'  {name:<16} logloss {log_loss(te["y"], pr):.4f}  AUC {roc_auc_score(te["y"], pr):.4f}')
print(f'\n学習 {len(tr):,} 組 / 評価 {len(te):,} 組  {time.time() - t0:.0f}秒')
