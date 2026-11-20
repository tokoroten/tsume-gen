# tsume-gen: 盤面と正解手順を同時に生成する詰将棋の自動作問

[English](README.md) · AI エージェント向けの手順: [AGENTS.md](AGENTS.md)（英語）

深層学習による詰将棋の生成器と、それを判定する厳密な詰将棋ソルバのコードです。論文「盤面と正解手順を同時に生成する詰将棋の自動作問：MaskGIT と厳密ソルバによる強化学習」（第 68 回プログラミング・シンポジウム）の §6「公開」にあたります。

- 生成器は MaskGIT をもとにした双方向の Transformer で、盤面・攻方の持駒・正解手順の最初の 3 手を、マスクした状態から同時に埋めて詰将棋の図を作ります。
- ソルバは Rust で書いた df-pn の詰将棋ソルバです。生成した図が、余詰・駒余り・関与しない駒がなく、正解手順を保ったまま外せる駒もない詰将棋（論文でいう極小完全作）かどうかを判定します。
- 人間が創作した詰将棋作品は、学習にも生成にも使っていません。

## 論文 §6 との対応

| 論文 §6 | 置き場 | 状態 |
| --- | --- | --- |
| データセット（v4（2026-11-01）で約 3,330 万図、CC BY 4.0） | <https://huggingface.co/datasets/tokoroten-jp/generated-tsume-shogi> | 公開済み |
| 本実験の学習データ（約 805 万件、CC BY 4.0） | <https://huggingface.co/datasets/tokoroten-jp/tsume-gen-m17-pretraining> | 公開済み |
| 生成器の重み（小・中・大の事前学習と、大にソルバの判定を報酬とする強化学習 2,000 ステップ） | <https://huggingface.co/tokoroten-jp/tsume-gen> | 公開済み |
| ソルバ | このリポジトリの `solver/` | 公開済み |
| 生成器を動かす Python のサンプル | このリポジトリの `examples/` | 公開済み |
| 生成器の学習のコード（事前学習・強化学習） | このリポジトリ | 今後足す |
| 量産の仕組み（生成・判定・縮約・駒足しの流れ） | このリポジトリ | 今後足す |
| ブラウザでの作問 | <https://tsume.quest/generate> | 公開済み（このリポジトリの対象外） |

## 中身

| 置き場 | 中身 |
| --- | --- |
| [`solver/`](solver/README.md) | Rust の詰将棋ソルバ（crate `tsume-solver`）。厳密な詰み探索、作品の規約、図を判定する JSON Lines の段 `validate`・`solve`・`reduce`。データセットの図を作ったサブコマンド（`distill`・`extend`・`mine`）もソルバの一部として入っています。 |
| [`generator/generate_min.py`](generator/generate_min.py) | 生成器の推論のコード（PyTorch）を 1 つのファイルにしたもの。モデル、重みの読み込み、条件の入力、マスクを外していく復号、SFEN と主線の最初の 3 手の出力。単独でも動きます（`python generator/generate_min.py --help`）。学習のコードは入っていません。 |
| [`examples/generate_and_judge.py`](examples/generate_and_judge.py) | サンプル。Hugging Face から重みを取得し、図を生成し、ソルバで判定します。 |
| [`requirements.txt`](requirements.txt) | `generator/` とサンプルが使う Python のパッケージ。 |
| [`training/restore_m17.py`](training/restore_m17.py) | tokoroten-jp/tsume-gen-m17-pretraining から、事前学習に読ませたファイルをバイト単位で同じに作り直します（パッケージは [`requirements-training.txt`](requirements-training.txt)）。 |
| [`LICENSE`](LICENSE) | MIT。 |

## 使い方

### ソルバ

```bash
cd solver
cargo build --release            # target/release/tsume-solver
./target/release/tsume-solver --help
./target/release/tsume-solver solve-sfen "4k4/9/4P4/9/9/9/9/9/9 b G 1" --max-ply 3
cd ..                            # リポジトリのルートに戻る
```

### 生成のサンプル

ソルバのビルドには Rust が要ります。`solver/rust-toolchain.toml` が stable のツールチェーンを指定しています（`rustup` があれば `cargo` が自動で選びます）。x86_64 では x86-64-v3（AVX2・BMI2）向けにビルドします。段と、それぞれが確かめる規約は [solver/README.md](solver/README.md) にあります。

サンプルには Python 3.12 と [requirements.txt](requirements.txt) のパッケージ（PyTorch・NumPy・safetensors・huggingface_hub）、上でビルドしたソルバが要ります。次のコマンドはリポジトリのルートで実行します。

```bash
python3 -m venv .venv                                   # Windows: py -3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python -m pip install -r requirements.txt
.venv/bin/python examples/generate_and_judge.py --model branch-rl-large --ply 3 --count 8
```

最初の準備では rustup・crates.io・PyPI からダウンロードします。そのあとのネットワークの利用は、Hugging Face からの重みのダウンロードだけです。

[`examples/generate_and_judge.py`](examples/generate_and_judge.py) は [tokoroten-jp/tsume-gen](https://huggingface.co/tokoroten-jp/tsume-gen) から `<model>/model.safetensors` と `<model>/config.json` を匿名で取得し（`--weights-dir` で手元のファイルを使うこともできます）、目標の手数 `--ply` を条件に、論文の §4.2・§4.4 の復号（24 ラウンド、温度 1.0、ランダムな順にマスクを外す、反則のマスの引き直し、top-p 0.95、玉方の玉は 1〜5 筋）で `--count` 個の図を生成し、すべての図をソルバで判定します: `validate`、`solve`（規約 A〜C・F・G）、`reduce`（規約 D）。図ごとに SFEN、判定（`sound`・`reducible`・`mate`・`rejected`・`invalid`・`unknown`・`undecodable`）、詰みの手数と主線を標準出力に出し、要約の 2 行を標準エラーに出します。`--json` を付けると図ごとに 1 つの JSON を出します。`sound` はすべての規約を満たした図です。`mate` は `solve` を通ったが規約 D を調べていない図（`--skip-reduce`）、`unknown` はソルバが判定の前に予算を使い切った図です。

- `--model`: `sft-small`・`sft-medium`・`sft-large`・`branch-rl-large`（既定、論文で使った重み）・`branch-rl-large-ema`。
- `--ply`（1・3・5・7・9・11）、`--count`、`--seed`、`--device`（`cpu`・`cuda`・`cuda:N`・`mps`）。`--defenders`（`0`〜`5`・`6+`）は玉方の盤上の駒の数（玉を除く）を条件にします（既定は指定なし）。
- 復号: `--rounds`（既定 24）、`--temperature`（既定 1.0）、`--top-p`（既定 0.95。1.0 で無効）、`--no-redo-check`（規則を破ったラウンドの引き直しをやめる）。`--plain` は `--top-p 1.0 --no-redo-check` と同じです。既定は論文の復号です。
- `--solver`（または環境変数 `TSUME_SOLVER_BIN`）でソルバの実行ファイルを指定します。`--skip-reduce` で規約 D を飛ばします。`--threads` はソルバのスレッド数です（既定はすべてのコア）。
- `--max-ply` はソルバの探索の上限です。既定の 13 は速さのためで、論文の判定は 15 手までです。`--max-ply 15` で同じ上限になります。合うのは判定の上限だけで、論文の評価の数値は、そこで使った標本・条件・復号にもよるので、これだけでは再現しません。13 のままだと、15 手でしか詰まない図は「詰まない」と出ます。
- `--sfen SFEN` を付けると、生成のかわりに与えた局面を判定します（何回でも付けられます。モデルは読み込みません）。
- 同じ seed と同じ種類のデバイスなら、生成は決定的です。デバイスが違うと、同じ seed でも違う図が出ます。

CPU で `branch-rl-large` を `--ply 3 --count 8` で動かすと、読み込みが 1 秒未満（ダウンロードの後）、生成が 7〜8 秒、判定が 14〜19 秒で、メモリのピークはソルバを含めて 0.6〜1.3 GB です（時間は Windows の PC 1 台で、メモリは Windows と macOS で測りました）。初回は重みを 343 MB ダウンロードします（huggingface_hub がキャッシュします）。GPU（`--device cuda`、CUDA 版の PyTorch が要ります）でのメモリは測っていません。1〜2 GB ほどの見込みです。`--model sft-small`（約 30 MB）でも動き、より速く終わります。

## 例: 15 手詰と、ソルバの解き方

このプロジェクトの生成器が作り、ソルバの判定を通った問題です。<https://tsume.quest> でも出題しました。攻方（先手、▲）が先に指し、持駒は金と銀です。玉方（後手、△）は残りの駒を全部持ちます。

```
後手の持駒：飛二 角 金二 桂四 香四 歩十七
  ９ ８ ７ ６ ５ ４ ３ ２ １
+---------------------------+
| ・ ・ ・ ・ ・v玉 ・v全 ・|一
| ・ ・ ・ ・v金 ・ 歩 ・ ・|二
| ・ ・ ・ ・ ・ ・ ・ ・ 銀|三
| ・ ・ ・ ・ ・ 銀 ・ ・ ・|四
| 角 ・ ・ ・ ・ ・ ・ ・ ・|五
| ・ ・ ・ ・ ・ ・ ・ ・ ・|六
| ・ ・ ・ ・ ・ ・ ・ ・ ・|七
| ・ ・ ・ ・ ・ ・ ・ ・ ・|八
| ・ ・ ・ ・ ・ ・ ・ ・ ・|九
+---------------------------+
先手の持駒：金 銀
```

SFEN: `5k1+s1/4g1P2/8S/5S3/B8/9/9/9/9 b GS 1`（玉方の持駒は省いてよい。ソルバが残りの駒を全部玉方に持たせる）。

正解（15 手）:

▲3一金 △同成銀 ▲同歩成 △同玉 ▲2二銀打 △4一玉 ▲3二銀打 △同玉 ▲3三銀成 △4一玉 ▲3一銀成 △同玉 ▲2二銀成 △4一玉 ▲3二成銀寄 まで15手

USI では `G*3a 2a3a 3b3a+ 4a3a S*2b 3a4a S*3b 4a3b 4d3c+ 3b4a 2b3a+ 4a3a 1c2b+ 3a4a 2b3b`。

### 判定のしかた

```bash
python examples/generate_and_judge.py --sfen "5k1+s1/4g1P2/8S/5S3/B8/9/9/9/9 b GS 1" --max-ply 15
```

```
 1 sound       mate in 15  5k1+s1/4g1P2/8S/5S3/B8/9/9/9/9 b GS 1
   main line: G*3a 2a3a 3b3a+ 4a3a S*2b 3a4a S*3b 4a3b 4d3c+ 3b4a 2b3a+ 4a3a 1c2b+ 3a4a 2b3b
```

- `--sfen` を付けると、生成のかわりに、与えた局面を判定します。何回でも付けられます。モデルは読み込まないので、重みのダウンロードもありません。
- `--max-ply 15` は論文の上限です。サンプルの既定の 13 のままだと、この問題は「上限の内に詰まない」と出ます。
- 判定の全体は、PC（Ryzen 9 5950X）で約 4 秒でした。

中ではソルバの 3 つの段を順に通します。どの段も、1 行に 1 つの JSON の記録を読み書きします。

1. **`validate`**: 詰将棋の局面として正しいかを確かめ（攻方の玉が無い、など）、玉方の持駒を埋めます。
2. **`solve --max-ply 15`**: まず df-pn の探索で詰みを証明します（証明の木は 463 ノード）。次に 15 手の内で、問題として正しいための規約を調べます。調べるのは、手数がちょうどであること、主線に余詰が無いこと、駒余りが無いこと、関与しない駒が無いこと、無駄合を数えないことです。最後に、上の主線を書きます。約 120 万ノード、0.8 秒でした。
3. **`reduce --max-ply 15`**: 駒を 1 つずつ外して解き直し、外せる駒が無いことを確かめます（`reduction.minimality: verified`）。約 3 秒でした。

ただの `tsume-solver solve-sfen "<SFEN>" --max-ply 15` は、すべての手順を調べ尽くす探索で、短い問題のためのものです。この問題では既定の 10 万ノードの予算（`--node-budget`）で止まり、`"outcome":"UNKNOWN"` になります。長い問題は、サンプルと同じく段を通して解いてください。

## データの由来

生成器の学習データは、データセット tokoroten-jp/generated-tsume-shogi（<https://huggingface.co/datasets/tokoroten-jp/generated-tsume-shogi>）として公開している図のデータベースから、2026-11-03 の時点で抜き出しました。これはデータセットの v4（2026-11-01）より後の状態で、学習データの行の約 82 % は v4 にあり、残りは二つの日付の間にデータベースに足した行です。学習データはそこから選んだ部分集合で、公開データセットそのものではありません。学習に使ったままの形を tokoroten-jp/tsume-gen-m17-pretraining（<https://huggingface.co/datasets/tokoroten-jp/tsume-gen-m17-pretraining>）として公開しています。`training/restore_m17.py` で、学習に読ませたファイルをバイト単位で同じに作り直せます。そこには次の 2 つの第三者のデータから作った図が含まれます。

- **やねうら王 実戦詰み局面集**（2020）: 将棋ソフトやねうら王の自己対局から、3〜11 手の詰みが見つかった局面を集めたもの。配布ページ <https://yaneuraou.yaneu.com/2020/12/25/christmas-present/> に「著作権は主張しませんのでご自由にお使いください。」とあります。局面から攻方の玉と不要な駒を取り除き（盤面蒸留）、逆算で手数を延ばした図などを作りました。
- **floodgate**（<https://wdoor.c.u-tokyo.ac.jp/shogi/>）: コンピュータ将棋の連続対局サーバの棋譜。終盤の局面から盤面蒸留した図を作りました。floodgate の棋譜には著作権の規定がないので、そこから作った図は注意して使ってください。

このほか、ランダムな配置から探した図と、過去の生成器が生成した図を、ソルバで判定して使っています。データセットでは、各図の由来を行ごとに記録しています。生成器が生成する図にも、学習データを通じてこれらの由来があります。

## ライセンス

- コード: MIT（[LICENSE](LICENSE)）。
- 生成器の重み・データセット: CC BY 4.0（それぞれの Hugging Face のページ）。
- CC BY 4.0 は、私たちが作った重みとデータセットのライセンスです。元の資料のライセンスではありません。やねうら王の局面集は、上に引いた作者の文のもとで使っています。floodgate には著作権の規定がありません（上記）。

## 引用

このリポジトリ・重み・データセットを使った場合は、次の論文を引用してください。

```bibtex
@inproceedings{nakayama2027tsume,
  author    = {中山 心太},
  title     = {盤面と正解手順を同時に生成する詰将棋の自動作問：{MaskGIT} と厳密ソルバによる強化学習},
  booktitle = {第68回プログラミング・シンポジウム予稿集},
  year      = {2027}
}
```

データセットだけを使った場合は、データセットのページにある CITATION.cff の書き方でも構いません。

## 今後

- 生成器の学習のコード（データセットからの学習例の抽出、事前学習、強化学習）。
- 量産の仕組み（生成、ソルバでの判定、盤面縮約、余詰を消す駒足し）。
- 論文の完全版（arXiv）。
