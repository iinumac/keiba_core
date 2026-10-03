# 週末の出馬表を取る

## JRA公式サイトは使えない

jra.go.jp はリンクが

```html
<a href="javascript:void(0);" onclick="doAction('/JRADB/accessS.html','pw01sde...')">
```

という形で、POST にトークンを渡して遷移する。**URLを直接叩けない。**
トークンはセッションに紐づくため、スクレイピングするなら
フォームの状態を保ったまま段階的に辿る必要があり、壊れやすい。

同じ出馬表は netkeiba から race_id で直接取れるので、そちらを使う。
race_id は既に `discovery` で検知できている。

## 取得の流れ

| # | 取得元 | 得るもの |
|---|---|---|
| 1 | `race.netkeiba.com/top/calendar.html?year=&month=` | 開催日（**未来も返る**） |
| 2 | `race.netkeiba.com/top/race_list_sub.html?kaisai_date=` | その日の race_id |
| 3 | `race.netkeiba.com/race/shutuba.html?race_id=` | 出走馬・枠番・斤量・騎手 |
| 4 | `race.netkeiba.com/api/api_get_jra_odds.html?race_id=&type=1` | 単勝オッズと人気 |

```python
from keiba import shutuba, fetch

f = fetch.Fetcher()
cards = shutuba.fetch_weekend(f, days_ahead=10)
```

リクエストは「月1 + 開催日数 + レース数×2」。週末48レースで約100回、
1秒間隔なので2分ほど。

## つまずきどころ

### `race_list.html` は空

一覧ページは JavaScript で後からデータを読む。HTMLを取っても race_id は
1件も入っていない。実データは **`race_list_sub.html`** にある。

### 文字コードがホストで違う

| ホスト | 文字コード |
|---|---|
| `db.netkeiba.com`（結果DB） | EUC-JP |
| `race.netkeiba.com`（出馬表・オッズ） | **UTF-8** |

一律 EUC-JP を指定すると出馬表が文字化けする。`fetch.encoding_for` が振り分ける。

### 結果DBには未来のレースが無い

`db.netkeiba.com/race/list/` は終わったレースしか持たない。
今週末の race_id を取るには出馬表側（`race_list_sub.html`）を使う。

### ページ上部のナビに同じクラス名が出る

出馬表ページの上部には全レースのナビがあり、そこにも `RaceName` や
`RaceData01` が現れる。`soup.find()` で取ると**別のレースの情報を拾う**。
当該レースの情報は `div.RaceList_NameBox` の中に絞って取る。

また `RaceName` は `div` ではなく `h1`。タグを決め打ちしないこと。

### グレードのアイコンは2桁がある

`Icon_GradeType15`（リステッド）を `Icon_GradeType(\d)` で取ると
`1` にマッチして **G1 と誤判定する**。数字全体を取ること。

### オッズは前日だとまだ出ていない

未発表の間は `---.-`。`card['odds_available']` で判定できる。

**オッズは発走直前まで動く。直前に取るほど市場の評価を正しく反映する。**
いつ時点のものかが分かるよう、API が返す更新時刻も持たせてある。

```python
card['odds_updated_at']   # '2026-10-02 23:42:16'
card['odds_status']       # '発売前' / '発売中' / '確定'
```

直前に何度も回すなら [`05_predict`](../notebooks/05_predict.ipynb) を使う。
生HTMLを取得しないので（62MB）起動が速い。ウェアハウスと学習済みモデルは
リポジトリにあるものを使う。採点は1回目に60秒ほど、2回目以降は一瞬。
`within_minutes` で発走が近いレースだけに絞れる。

```python
stages.predict(within_minutes=90)   # 発走まで90分以内のレースだけ
stages.predict(race_ids=[...])      # レースを指定して取り直し
```

## 出力

```python
{
  'race_id': '202605040111',
  'date': '2026-10-03',
  'venue_name': '東京', 'race_num': 11,
  'race_name': 'グリーンチャンネルC', 'grade': 'L',
  'surface': 'ダ', 'distance': 1600, 'horse_count': 15,
  'start_time': '15:45', 'weather': '曇', 'track_condition': '稍',
  'odds_available': True,
  'horses': [
    {'bracket': 1, 'horse_number': 1, 'horse_name': 'ルヴァレドクール',
     'horse_id': '2022105123', 'jockey_id': '01091', 'trainer_id': '01141',
     'sex_age': 'セ4', 'impost': 57.0, 'jockey_name': '横山和',
     'horse_weight': 482.0,           # 当日まで None
     'odds': 9.1, 'popularity': 5},
    ...
  ]
}
```

`shutuba.to_json(cards)` でJSON文字列になる。

## 過去走との紐付け

馬名で引くと同名馬や表記ゆれで取り違えるので、出馬表の
`/horse/{id}/` リンクから `horse_id` を取る。騎手・調教師も同じ。

`shutuba.to_result_rows(cards)` で results と同じ形の行に直せる。
これを results の末尾に足して `features.build_features` を通せば、
学習時と同じ特徴量になる（特徴量は過去走だけから作られるため）。

| 列 | 出馬表での扱い |
|---|---|
| `level_score` | 条件欄の「本賞金:590,…万円」から復元。推定ではなく正確 |
| `horse_weight` | 当日になれば出る。前日は None |
| `finish_position` | `UNRUN`（99）。is_top3 を立てず、過去の集計を汚さない |
| `is_unrun` | True。騎手・調教師の平均にこの行を入れないための印 |
| タイム・上がり・着差 | None（走ってみないと分からない） |

通常のバッチ計算と比べて、最終開催日24レースで最大差 0.002、
レース内の順位は全レースで一致した。
