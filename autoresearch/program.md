# Titanic autoresearch

## 目的

scikit-learn のアルゴリズム、特徴量、前処理をコーディングエージェントが仮説に基づいて変更し、固定したローカル評価の accuracy を改善する。まずはレビュー可能な小規模実験として運用する。

初期モデルの出典:
- https://github.com/ftnext/2019_slides/blob/master/spz_Oct_titanic_handson4/PITCHME.md
- https://www.kaggle.com/code/ftnext/kaggle-spzcolab-201910

初期設定は Age / Pclass / Sex / Embarked の4特徴量と `LogisticRegression(random_state=1, solver="liblinear")`。Age は平均、Embarked は最頻値で補完。Sex は female=0 / male=1、Embarked はダミー変数化する。標準化や派生特徴量はまだ追加しない。

## 準備とコマンド

このディレクトリで実行する。

```bash
# データがない場合のみ。既存 prepare.sh が Kaggle CLI を呼び出す。
./prepare.sh
uv sync
# 必ず最初に変更前のベースラインを評価する。
uv run train.py > baseline.log 2>&1
# 変更後の1候補を評価する。
uv run train.py > run.log 2>&1
```

`data/train.csv` と `data/test.csv` を使う。データパスは train.py の位置から解決される。認証やコンペ参加が未完了でデータ取得に失敗したら、その事実を報告する。秘密情報は表示・記録しない。

`uv sync` が作る uv.lock は保存し、実験中は依存関係を固定する。

## 編集してよい範囲

- `train.py` の EXPERIMENT AREA 内の `build_pipeline()`、必要な import、同領域に追加する特徴量変換クラス。
- モデル選択、ハイパーパラメータ、特徴量の追加・削除、欠損補完、エンコーディング、標準化。
- 派生特徴量は Pipeline 内で作る。学習を伴う処理は `fit()` で学習し、`transform()` で適用する。scikit-learn の clone に対応させる。
- 入力には生の予測用列が渡る。使う列は Pipeline 側で選ぶ。Survived と PassengerId は入力されない。

## 固定するもの

- `train.py` の FIXED EVALUATION 以下、データ、分割、seed、指標、時間上限、pyproject.toml、uv.lock、prepare.sh、この program.md。
- 分割前に平均・最頻値・特徴選択などを学習しない。検証foldを使って前処理をfitしない。
- 特徴変換からCSVを再読込しない。外部データ、公開済み正解、生存者の照合、test.csvを使った学習は対象外。
- 毎回のKaggle提出、提出スコアによる選択、ネットワークアクセス、追加パッケージは探索に含めない。
- 確率を使うモデルの閾値を調整する場合も、検証foldの正解でそのfoldの閾値を決めない。

同じファイルに評価処理があるので、この編集境界は運用上の約束であり、権限による隔離ではない。各変更で固定領域の差分がないことを確認する。

## 評価と予算（初版の暫定判断）

- ラベル付きデータの20%を stratify=y / random_state=1 で最終確認用に確保する。
- 残り80%に固定の層化5分割CVを適用する。主指標は `cv_accuracy`（fold平均、高いほどよい）。`cv_std` はfold間の標準偏差であり、信頼区間ではない。
- 当時のNotebookの単一7:3分割とは評価方法が異なるため、当時のスコアとの直接比較はしない。
- 前処理は各fold内でfitする。初期特徴量とモデルは引き継ぐが、当時の分割前補完は引き継がない。
- 1実行は120秒で打ち切る。UnixのSIGALRMを利用するためmacOS/Linux向け。プロセス起動・依存解決はこの上限に含まれない。外側でも経過時間を監視し、終了しなければ停止する。
- 1セッションはベースラインを含め最大20評価試行、または総経過30分。先に達した方で終了する。失敗・修正再実行も数える。人間からの停止指示でも終了する。
- 0.005以上の平均accuracy改善は採用候補。より小さい改善もfold別の変化と複雑さを見て判断し、理由を残す。0.005は統計的有意差の閾値ではない。
- 同程度なら簡単で速い候補を優先。固定CVへの過適合を認識し、微小な差を追って探索を増やし続けない。

## 実験ループ

1. 既存ファイル、Gitの状態、過去の実験を確認する。ユーザーの変更や過去ログを上書きしない。勝手にブランチを切り替えたり、リポジトリ全体をresetしたりしない。
2. 初回はベースラインを無変更で実行して記録する。失敗時は成功するまで最適化を始めない。
3. 次の仮説、予想される効果、比較元の実験IDを短く記す。1候補の主な仮説は1つにする。
4. `runs/<一意な実験ID>/` に実行する train.py を保存する。評価は元の train.py で行い、全出力を同ディレクトリの run.log に保存する。実行後にコードがスナップショットと同じことを確認する。
5. 終了コード、cv_accuracy、cv_std、fold別スコア、実行時間、コードとデータのハッシュを読む。非ゼロ終了、タイムアウト、非有限値、収束失敗は成功扱いにしない。
6. results.tsv に追記する。不採用・失敗のコードとログも残す。既存ファイルがあればヘッダーと列に合わせ、上書きしない。
7. 採否と理由を記録する。最良候補IDを記録し、不採用時は自分が変更した実験領域だけを最良候補へ戻す。
8. 予算内で次の仮説へ進む。過去の失敗を踏まえ、モデル比較、列の追加・削除、派生特徴量、前処理の変更、有望候補の調整を考える。巨大な総当たりで1試行に多数の候補を隠さない。

新規 results.tsv のヘッダー（タブ区切り）:

```text
experiment_id\tparent_id\tcode_sha256\tdata_sha256\tcv_accuracy\tcv_std\tfold_scores\tseconds\tstatus\thypothesis\tdecision
```

status は keep / discard / crash / timeout。失敗時の指標は NA とし、0をスコアの代用品にしない。仮説・判断は改行やタブを含まない短文にする。

## 終了時

最良候補を train.py に復元し、必要なら予算内で再現を確認する。実験一覧、ベースラインとの差、採用理由、失敗から分かったこと、限界、次の仮説を日本語で報告する。ベースラインより改善しなければ、そのまま報告する。

通常探索では `--audit` を実行しない。最良候補を凍結し、人間が最終確認を依頼した段階で実行する。

```bash
uv run train.py --audit
```

holdout_accuracy は最終確認専用。結果を見てモデルを変えたら、以後このholdoutを未使用の最終評価と呼ばない。ファイル上のラベルは閲覧可能なので、ここも運用上の分離である。

提出CSV作成を依頼されたら、凍結した候補を全ラベル付きデータで再学習する。

```bash
uv run train.py --submission submission.csv
```

PassengerIdとSurvivedのCSVを生成し、既存出力は上書きしない。Kaggleへのアップロードは行わない。
