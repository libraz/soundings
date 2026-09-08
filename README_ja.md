# soundings

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/libraz/soundings/blob/main/LICENSE)
[![Data](https://img.shields.io/badge/data-CC0--1.0-blue)](https://github.com/libraz/soundings/blob/main/data/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](https://github.com/libraz/soundings)

`soundings` は、ハードウェア MIDI 音源の MIDI 制御面と音響上の振る舞いを測定します。Python 製のコマンドライン・ハーネスと、個体ごとの測定結果を収録したバージョン管理済みのアーカイブで構成されています。

仕様書の転記ではなく、実機で観測した振る舞いを記録します。対象は、読み出せるアドレス、受け付ける値、リセット後・電源投入後の状態、メッセージの別名、利用可能な音色とエフェクト、音響測定です。結果は特定の個体と測定環境に限定されます。

## 対象範囲と現状

現在のアーカイブには、Roland SC-8850 1 台の測定結果が `data/units/roland-sc8850-01/` に含まれます。アドレスマップと各領域の境界、各アドレスが受け付ける値と自分の値を保持するかどうか、電源投入直後の状態、各リセットが復元するもの、メッセージ族ごとの保存先、音色とインサーションエフェクトのカタログ、そして音響測定 — 繰り返しの安定性、単一アドレスでの可聴差とそれをブロック単位に集約したもの、インサーションエフェクト個別の型のパラメータ、エフェクトの減衰と変調です。

測定結果は、同じ型番やファームウェアのすべての個体に当てはまることを示すものではありません。`meta.json` には個体の識別情報、個体から取得できたファームウェア情報、MIDI と音声の測定経路を記録しています。

## インストール

必要なもの:

- Python 3.11 以上（[Rye](https://rye.astral.sh/) で管理）
- MIDI 測定用の双方向 MIDI インターフェース
- 対象となる音源。音響測定にはオーディオインターフェースも必要です

```sh
rye sync
rye run soundings devices
```

利用可能なコマンドは次で確認できます。

```sh
rye run soundings --help
```

## 実機を測定する前に

データを取る前に selftest を実行してください。

```sh
rye run soundings selftest --audio "<オーディオインターフェース>"
```

返信 SysEx のチェックサム、既知のアドレスに対する繰り返し読み出し、音声キャプチャの時間軸を検証します。ただし、この検証は未知の機器に対して安全にプローブできることを保証しません。実機に対してコマンドを実行する前に、ヘルプと既存の測定結果を確認してください。要求によっては応答が返らない、または電源を入れ直すまで機器が応答しなくなることがあります。

MIDI インターフェースの選択には `--port`、SysEx デバイス ID の指定には `--device-id` を使います。どちらもグローバルオプションなので、サブコマンドより前に置きます。結果データを書き出すコマンドは、必要に応じて `--out` を受け取ります。

## コマンド

| コマンド | 用途 |
|---|---|
| `devices` / `selftest` | MIDI とオーディオのデバイスを一覧し、測定経路を検証します。 |
| `identity` / `read "40 01 30" 16` | Identity Request を送信し、Roland RQ1 でアドレスを読み出します。 |
| `sweep` / `boundary` / `offsets` | アドレス空間を地図化し、領域が地図どおりの位置で終わるかを確かめ、1 バイト読みでしか届かないアドレスを探します。 |
| `write-probe` / `hold-probe` / `window-probe` | アドレスが受け付ける値、自分の値を保持するかどうか、他のアドレスの値を映しているかどうかを測定します。 |
| `power-on` / `reset-probe` | 電源投入直後の状態を取得し、各リセットが復元するものを測定します。 |
| `tone-map` / `efx-map` | 利用可能な音色またはインサーションエフェクトを調べます。 |
| `alias-scan` | MIDI メッセージで変化する保存先を特定します。 |
| `port-send` / `port-read` | 応答を返さない MIDI 入力へ送り、それが機体に残したものを読み出します。 |
| `repeat` / `contrast` / `verdict` | 繰り返しの安定性と可聴差を測定し、以前に録ったテイクを判定します。 |
| `transfer` / `motion` / `decay` / `vibrato` | 音声経路、時間変化するエフェクト、エフェクトの減衰、テイクのピッチ変調を解析します。 |
| `balance` | パラメータがチャンネル間のレベル差に与えた影響を測定します。 |
| `plan` / `block` / `efx-params` | ブロックを問う値を決め、保存済みの記録をアドレスごと・エフェクトパラメータごとの判定 1 つに集約します。 |
| `efx-motion` / `efx-sort` | エフェクトが時間とともに何をするかを追い、静止する型としない型に仕分けます。 |
| `index` / `complete` | 個体の記録を一覧し、完成の基準に対する到達度を数えます。 |

保存済みの記録を読むコマンド — `motion`、`decay`、`verdict`、`balance`、`vibrato`、`plan`、`block`、`efx-params`、`efx-motion`、`efx-sort`、`complete`、`index` — は実機を必要としないので、別の測定が機材を占有している間も実行できます。

## データ形式

個体ディレクトリは、測定段階ごとに 1 つのサブディレクトリを持ちます。名前は、その中の記録を書いたコマンドです。記録は、何についてのものか — その実行が問うたアドレス、ブロック、エフェクト型、コントローラ — で名付けられ、アドレス空間の一部ではなく全体を覆う実行は `whole-map.json` です。

```text
data/units/<manufacturer>-<model>-<n>/
  meta.json                 個体の識別情報と測定経路
  measurements.json         記録した観測事項とスポットチェック
  index.json                以下すべての生成された一覧

  sweep/                    読み出せるアドレス領域
  boundary/                 領域が地図どおりの位置で終わるかどうか
  offsets/                  1 バイト読みでしか届かないアドレス
  write-probe/              受け付けた書き込みと読み戻し結果
  hold-probe/               隣接アドレスが自分の値を保持するかどうか
  window-probe/             他のアドレスの値を映しているアドレス
  power-on/                 電源投入直後に取得した状態
  reset-probe/              リセットで変化する状態
  tone-map/                 利用可能な音色
  efx-map/                  利用可能なインサーションエフェクト
  alias-scan/               メッセージとアドレスの対応
  port-send/  port-read/    応答を返さない MIDI 入力が機体に残したもの

  repeat/                   機体がテイクをどれだけ再現するか。以下の音響比較
                            はすべてこれを基準に読む
  contrast/                 単一アドレスまたはコントローラでの音響比較
  block/                    その比較をアドレスごとの判定 1 つに集約したもの
  efx-params/               インサーションエフェクト 1 型のパラメータ
  balance/                  パラメータがチャンネル間のレベル差に与えた影響
  transfer/                 掃引正弦波に対して経路が行ったこと
  motion/  decay/  vibrato/ エフェクトの変調、減衰、テイクのピッチ
  efx-motion/  efx-sort/    静止するエフェクト型としない型
```

`soundings index <個体ディレクトリ>` が `index.json` を再生成します。この一覧は、各記録を、それを書いた段階と、記録自身が述べる対象とともに挙げます。

結果ファイルには、各測定の手法、設定、限界を保存します。仕様書や特許に基づく仮説は、測定結果ではなく測定の背景として記録します。

## 再現性

`meta.json` には、MIDI・オーディオインターフェース、配線、キャプチャ方法、レベル、関連する機器設定を記録します。データを追加する場合も、他者が測定を再現したり個体差を評価したりできる情報を記録してください。

## ライセンス

測定ハーネスは [MIT](LICENSE)、`data/` 以下のデータは [CC0 1.0](data/LICENSE) です。引用は歓迎しますが、条件ではありません。

`documents/` はそのどちらでもなく、そのために `data/` の外にあります。公表された文書から読み取った事実の表を保持し、文書の記載を実測と突き合わせられるようにするためのものです。文書自体は各発行元のものであり、ここでは配布していません。[documents/LICENSE](documents/LICENSE) を参照してください。

本プロジェクトは、いかなる楽器メーカーとも提携・関連・公認の関係にありません。製品名は測定した機材を識別するために記載しています。
