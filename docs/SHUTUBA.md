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
確定オッズが要るなら発走が近づいてから取る。

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
     'sex_age': 'セ4', 'impost': 57.0, 'jockey_name': '横山和',
     'odds': 9.1, 'popularity': 5},
    ...
  ]
}
```

`shutuba.to_json(cards)` でJSON文字列になる。
