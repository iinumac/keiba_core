"""三連複の期待値予想

旧 C04 のノートブック内にあった予想エンジンを、そのままモジュールへ移したもの。
ノートブックに 11,000 文字のコードを置いておくと、差分が読めず再利用もできない。

`Predictor` が学習済みモデルと最新マスタを保持し、JRA出馬表JSON から
買い目までを出す。マスタは 03_train が作るものと同じ考え方で、
`features.build_features()` の出力から最新行を取って渡す。
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tabulate import tabulate
except ImportError:  # tabulate が無い環境向けの簡易フォールバック
    def tabulate(rows, headers=(), tablefmt=''):
        out = [' | '.join(str(h) for h in headers)]
        out += [' | '.join(str(c) for c in r) for r in rows]
        return '\n'.join(out)


def parse_jra_json_to_races(json_input):
    '''
    JRAの出走馬一覧JSON（文字列または辞書）から全レース情報をパースする
    '''
    if isinstance(json_input, str):
        try:
            json_input = json.loads(json_input)
        except Exception as e:
            print(f'JSONパースエラー: {e}')
            return []
            
    content = json_input.get('content', [])
    all_races = []
    race_counter = 1
    
    for elem in content:
        if elem.get('type') != 'list':
            continue
            
        for item in elem.get('items', []):
            text = item.get('text', '')
            if '発走時刻：' not in text or '単勝オッズ' not in text:
                continue
                
            # レース情報
            m_date = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
            race_date = f'{m_date.group(1)}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}' if m_date else '2026-08-22'
            
            m_venue = re.search(r'\d+回([^\d\s]+)\d+日', text)
            venue = m_venue.group(1) if m_venue else '札幌'
            
            m_time = re.search(r'発走時刻：\s*(\d{1,2}時\d{1,2}分)', text)
            start_time = m_time.group(1) if m_time else ''
            
            m_course = re.search(r'コース：([\d,]+)\s*メートル\s*（(芝|ダート|ダ)\s*[・\s]*(右|左|直)?）', text)
            distance = int(m_course.group(1).replace(',', '')) if m_course else 1600
            surface = 'ダート' if (m_course and 'ダ' in m_course.group(2)) else '芝'
            surface_code = 1 if surface == 'ダート' else 0
            
            m_name = re.search(r'発走時刻：\s*\d{1,2}時\d{1,2}分\s*(\S+)', text)
            race_name = m_name.group(1) if m_name else f'第{race_counter}レース'
            
            # 出走馬トークン分割
            table_part = text.split('単勝オッズ')[-1].replace('ページトップへ戻る', '').strip()
            tokens = table_part.split()
            i = 0
            horses = []
            
            while i < len(tokens):
                token = tokens[i]
                if token.isdigit() or token in ['除外', '取消']:
                    num_str = token
                    if i + 1 >= len(tokens):
                        break
                    name = tokens[i+1]
                    i += 2
                    
                    weight = 470.0
                    is_debut = 0
                    if i < len(tokens) and tokens[i].isdigit() and i + 1 < len(tokens) and tokens[i+1] == 'kg':
                        weight = float(tokens[i])
                        i += 2
                        if i < len(tokens) and tokens[i].startswith('('):
                            i += 1
                    elif i < len(tokens) and '(初出走)' in tokens[i]:
                        is_debut = 1
                        i += 1
                        if i < len(tokens) and tokens[i].isdigit():
                            weight = float(tokens[i])
                            i += 1
                            if i < len(tokens) and tokens[i] == 'kg':
                                i += 1
                    elif i < len(tokens) and tokens[i].isdigit() and i + 2 < len(tokens) and tokens[i+1] == 'kg' and '(初出走)' in tokens[i+2]:
                        weight = float(tokens[i])
                        is_debut = 1
                        i += 3
                        
                    sex_age = ''
                    if i < len(tokens) and re.match(r'^(?:牡|牝|せん|セ)\d+$', tokens[i]):
                        sex_age = tokens[i]
                        i += 1
                        
                    impost = 55.0
                    if i < len(tokens) and re.match(r'^\d+(\.\d+)?$', tokens[i]):
                        impost = float(tokens[i])
                        i += 1
                        if i < len(tokens) and tokens[i] == 'kg':
                            i += 1
                            
                    if i < len(tokens) and tokens[i] in ['▲', '☆', '◇', '★']:
                        i += 1
                        
                    jt_tokens = []
                    odds = 0.0
                    is_excluded = False
                    while i < len(tokens):
                        if re.match(r'^\d+\.\d+$', tokens[i]):
                            odds = float(tokens[i])
                            i += 1
                            break
                        elif tokens[i] in ['除外', '取消']:
                            is_excluded = True
                            i += 1
                            break
                        else:
                            jt_tokens.append(tokens[i])
                            i += 1
                            
                    if not is_excluded and num_str.isdigit():
                        if len(jt_tokens) >= 4:
                            jockey = ''.join(jt_tokens[:2])
                            trainer = ''.join(jt_tokens[2:])
                        elif len(jt_tokens) == 2:
                            jockey = jt_tokens[0]
                            trainer = jt_tokens[1]
                        elif len(jt_tokens) == 3:
                            if '.' in jt_tokens[0]:
                                jockey = jt_tokens[0]
                                trainer = ''.join(jt_tokens[1:])
                            else:
                                jockey = ''.join(jt_tokens[:2])
                                trainer = jt_tokens[2]
                        else:
                            jockey = ' '.join(jt_tokens)
                            trainer = ''
                            
                        horses.append({
                            'horse_number': int(num_str),
                            'horse_name': name,
                            'jockey_name': jockey,
                            'trainer_name': trainer,
                            'impost': impost,
                            'horse_weight': weight,
                            'odds': odds,
                            'is_debut': is_debut,
                            'sex_age': sex_age
                        })
                else:
                    i += 1
                    
            if len(horses) > 0:
                all_races.append({
                    'race_num': race_counter,
                    'race_date': race_date,
                    'venue_name': venue,
                    'start_time': start_time,
                    'distance': distance,
                    'surface': surface,
                    'surface_code': surface_code,
                    'race_name': race_name,
                    'horses': pd.DataFrame(horses)
                })
                race_counter += 1
                
    return all_races


@dataclass
class Predictor:
    """学習済みモデルと最新マスタを束ねた予想器。

    Args:
        model_top3: 3着内確率モデル
        model_win: 勝率モデル
        features: モデルが使う特徴量の列名リスト
        latest_horse_master: horse_name を index にした最新行
        latest_jockey_master: jockey_name -> jockey_added_value
        latest_trainer_master: trainer_name -> trainer_added_value
    """

    model_top3: Any
    model_win: Any
    features: List[str]
    latest_horse_master: pd.DataFrame
    latest_jockey_master: Dict[str, float]
    latest_trainer_master: Dict[str, float]

    @classmethod
    def from_features(cls, feature_df: pd.DataFrame, model_top3, model_win,
                      features: List[str]) -> 'Predictor':
        """build_features() の出力から最新マスタを組み立てて生成する。

        同一日に複数レースがある場合に拾う行が定まるよう、race_id まで含めて
        並べてから最新行を取る。
        """
        order = [c for c in ('date', 'race_date') if c in feature_df.columns][:1] + ['race_id']
        snap = feature_df.sort_values(order)
        return cls(
            model_top3=model_top3,
            model_win=model_win,
            features=features,
            latest_horse_master=snap.groupby('horse_name').tail(1).set_index('horse_name'),
            latest_jockey_master=snap.groupby('jockey_name').tail(1)
                .set_index('jockey_name')['jockey_added_value'].to_dict(),
            latest_trainer_master=snap.groupby('trainer_name').tail(1)
                .set_index('trainer_name')['trainer_added_value'].to_dict(),
        )

    def predict(self, race_dict):
        '''
        出走馬表とオッズから過去実績を自動結合し、AI三連複予想を出力する
        '''
        rdf = race_dict['horses'].copy()
        if len(rdf) == 0:
            return
        
        # 人気順位の自動付与（オッズ昇順）
        rdf['popularity'] = rdf['odds'].rank(method='min').astype(int)
        rdf['distance'] = race_dict['distance']
        rdf['surface_code'] = race_dict['surface_code']
    
        # 過去データから馬実績をルックアップ
        horse_stats = []
        for _, row in rdf.iterrows():
            h_name = row['horse_name']
            if h_name in self.latest_horse_master.index:
                h_info = self.latest_horse_master.loc[h_name]
                if isinstance(h_info, pd.DataFrame):
                    h_info = h_info.iloc[-1]
                p_win = h_info.get('horse_prev_win_rate', 0.08)
                p_top3 = h_info.get('horse_prev_top3_rate', 0.24)
                p_fin = h_info.get('prev_finish', 10)
                p_pop = h_info.get('prev_popularity', 10)
                p_odds = h_info.get('prev_odds', 30.0)
                p_l3f = h_info.get('prev_last_3f', 36.5)
                days = h_info.get('days_since_last', 60)
                debut = 0
            else:
                p_win, p_top3, p_fin, p_pop, p_odds, p_l3f, days, debut = 0.08, 0.20, 10, 10, 30.0, 36.5, 90, 1
            
            j_added = self.latest_jockey_master.get(row['jockey_name'], 0.0)
            t_added = self.latest_trainer_master.get(row['trainer_name'], 0.0)
        
            horse_stats.append({
                'horse_prev_win_rate': p_win,
                'horse_prev_top3_rate': p_top3,
                'prev_finish': p_fin,
                'prev_popularity': p_pop,
                'prev_odds': p_odds,
                'prev_last_3f': p_l3f,
                'days_since_last': days,
                'is_debut': 1 if row['is_debut'] == 1 else debut,
                'jockey_added_value': j_added,
                'trainer_added_value': t_added
            })
        
        stat_df = pd.DataFrame(horse_stats)
        for col in stat_df.columns:
            rdf[col] = stat_df[col].values
        
        # 市場歪み特徴量
        rdf['market_implied_win_prob'] = 0.8 / rdf['odds']
        min_o = rdf['odds'].min()
        rdf['odds_ratio_to_fav'] = rdf['odds'] / (min_o + 1e-5)
        rdf['pop_odds_mismatch'] = rdf['popularity'] * 2.5 - np.log(rdf['odds'] + 1)
    
        # 欠損補完 & 推論
        X = rdf[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        raw_top3 = self.model_top3.predict(X)
        raw_win = self.model_win.predict(X)
    
        # 確率のレース内正規化
        rdf['pred_top3_prob'] = np.clip(raw_top3 * (3.0 / (raw_top3.sum() + 1e-9)), 0.02, 0.98)
        rdf['pred_win_prob'] = raw_win / (raw_win.sum() + 1e-9)
    
        # 複勝推定オッズ & 期待値(EV)
        rdf['fukusho_odds_est'] = np.clip(1.0 + (rdf['odds'] - 1.0) * 0.28, 1.1, 30.0)
        rdf['ev_top3'] = rdf['pred_top3_prob'] * rdf['fukusho_odds_est']
        rdf['ev_win'] = rdf['pred_win_prob'] * rdf['odds']
    
        # 総合スコアと印の割り当て
        rdf['score'] = rdf['pred_top3_prob'] * 0.65 + (rdf['ev_top3'] / rdf['ev_top3'].max()) * 0.35
        rdf = rdf.sort_values(by='score', ascending=False).reset_index(drop=True)
    
        marks = ['◎ (本命)', '◯ (対抗)', '▲ (単穴)', '△ (連下)', '△ (連下)', '☆ (特注穴)']
        rdf['mark'] = [marks[i] if i < len(marks) else '―' for i in range(len(rdf))]
    
        # 出力ヘッダー
        r_num = race_dict.get('race_num', 1)
        r_name = race_dict['race_name']
        r_time = race_dict['start_time']
        r_date = race_dict['race_date']
        venue = race_dict['venue_name']
        course_str = f"{race_dict['surface']}{race_dict['distance']}m"
    
        print('=' * 75)
        print(f'🏇 【第{r_num}R {r_name}】 {r_date} {venue} {course_str} (発走: {r_time})')
        print('=' * 75)
    
        disp_df = rdf[['mark', 'horse_number', 'horse_name', 'sex_age', 'impost', 'jockey_name', 'popularity', 'odds', 'pred_top3_prob', 'ev_top3']].copy()
        disp_df['pred_top3_prob'] = (disp_df['pred_top3_prob'] * 100).map('{:.1f}%'.format)
        disp_df['ev_top3'] = disp_df['ev_top3'].map('{:.2f}'.format)
        disp_df['impost'] = disp_df['impost'].map('{:.1f}kg'.format)
        disp_df['odds'] = disp_df['odds'].map('{:.1f}'.format)
    
        print(tabulate(
            disp_df.values,
            headers=['印', '馬番', '馬名', '性齢', '斤量', '騎手', '人気', '単勝オッズ', 'AI複勝率', '複勝EV'],
            tablefmt='github'
        ))
    
        # 三連複 買い目提案
        axis = rdf.iloc[0]
        h_axis = int(axis['horse_number'])
        sub_opp = rdf.iloc[1:6]
        h_opps = [int(x) for x in sub_opp['horse_number'].tolist()]
    
        print('\n🎯 【AI推奨 三連複買い目】')
        print(f'  ① 【軸1頭ながし】 (10点)')
        print(f'     軸馬 (◎): [{h_axis:02d}] {axis["horse_name"]} (AI複勝率 {axis["pred_top3_prob"]*100:.1f}%, 期待値 {axis["ev_top3"]:.2f})')
        print(f'     相手 (◯▲△☆): {[f"{h:02d}" for h in h_opps]}')
    
        if len(rdf) >= 5:
            h_box = [int(x) for x in rdf.iloc[:5]['horse_number'].tolist()]
            print(f'  ② 【高期待値 5頭BOX】 (10点)')
            print(f'     買い目: {[f"{h:02d}" for h in h_box]}')
        print('=' * 75 + '\n')
