# Holder–Bayes：追加実験の実行手順

この追加コードは、投稿時のコードに次の2本の実験を追加します。元のソースは変更しません。既存の結果CSVや事後サンプルは不要で、データ生成・推定から実行できます。

1. 論文の既存設定における、頻度論的Hölder（DPD型）最小化との直接比較。
2. 小標本回帰における、誤除去・見逃し・確認費用の比較。頻度論的点推定、H-B点推定、FoD、pairs Bootstrap、および不確実性の比較用設定を含みます。

## 1. 配置と準備

ZIPの中身を、元の `holder-bayes` フォルダの直下に展開してください。`scripts/`、`configs/`、`docs/`、`tests/` の中に新しいファイルが追加され、`hb_experiments/additional/` が作成されます。ZIP自体には元のコードや結果ファイルを含めていません。

以降は元のリポジトリ直下、Python 3.12の環境で実行します。元の実験用環境を準備済みなら、インストールは不要です。

```bash
python -m pip install -e ".[experiments]"
python scripts/run_additional_experiments.py doctor
```

Pythonのコマンド名が `python3` の場合は読み替えてください。シェル版では `PYTHON=python3 bash scripts/run_additional_experiments.sh ...` と指定できます。

## 2. まず動作確認

```bash
python scripts/run_additional_experiments.py run \
  --config configs/additional_smoke.json \
  --output-dir outputs/additional_smoke --finalize
```

合計5設定で、実際の推論・最小化・Bootstrap・集計・作図を通します。**この設定は40 warmup / 40 draws・Bootstrap 4回で、性能評価には使えません。** 収束基準は緩めていないため、通常は `paired_valid=False` になります。生の結果と診断は保存され、収束基準を通った比較がなければ性能図は作らず、完了状況の図を出します。

数値計算と費用計算のテストは次のコマンドです。

```bash
python -m unittest discover -s tests -p test_additional_experiments.py -v
```

## 3. 本実験

本実験用設定は `configs/additional_experiments.json` です。途中で止まっても、**同じコマンドを再実行**すると、完了した設定を再利用し、推論キャッシュやBootstrapのチェックポイントから続けます。

### 実験1：既存設定との直接比較

```bash
python scripts/run_additional_experiments.py run --study direct \
  --config configs/additional_experiments.json \
  --output-dir outputs/additional_full \
  --local-data-dir /absolute/path/to/raw_data --finalize
```

実データは元のローダーが受け取る `cal_housing.csv`、`kin8nm.csv`、`pumadyn32fh.csv`（または `.tsv`、**ヘッダー行あり**、最終列が目的変数）を用意します。`target` という列名がある場合はその列を目的変数として使います。元の論文と同じデータを使ってください。ローカルに用意しない場合は `--local-data-dir ...` の代わりに `--download-data` を指定できます。取得するのは生データで、過去の推定結果ではありません。

シミュレーションだけ先に実行する場合は、設定ファイルをコピーして `direct.benchmark` を `false` にし、**別の出力フォルダ**を指定してください。

### 実験2：確認費用付きシミュレーション

```bash
python scripts/run_additional_experiments.py run --study costs \
  --config configs/additional_experiments.json \
  --output-dir outputs/additional_full --finalize
```

こちらは外部データ不要です。実験1と同じ設定ファイルなら同じ出力先を使えます。`--study both` では2本を順に実行します。

**本実験の規模**：実験1は360設定（密度推定150、合成回帰180、実データ30）、実験2は400設定です。さらに回帰の除去後の再推定、および実験2では各設定200回×複数初期値のBootstrap最小化が走ります。`plan` で実行前に全設定を確認できます。

```bash
python scripts/run_additional_experiments.py plan \
  --config configs/additional_experiments.json > additional_plan.json
```

中間規模の試行用に `configs/additional_pilot.json` も付けています。こちらは本実験とは別の出力先に保存してください。設定を変えた結果を混ぜないため、出力先の設定・ソース・主要依存バージョンが変わると再利用を止めます。

## 4. 結果の場所と読み方

各実験は `OUTPUT/direct/`、`OUTPUT/costs/` に保存されます。

| 出力 | 内容 |
|---|---|
| `summary/completion.csv`・`.json` | 全予定設定の完了・未実行・エラー・収束状況 |
| `summary/metrics_all.csv` | 実験1の全結果。診断不合格も残す |
| `summary/metrics_summary.csv` | 同じ反復の比較対象が診断を通った結果の平均・SD・件数 |
| `summary/cleaning_all.csv`・`cleaning_summary.csv` | 除去後の再推定。再推定失敗も記録 |
| `summary/policies_all.csv`・`policies_summary.csv` | 実験2の方策別の損失・誤除去・見逃し・確認件数 |
| `summary/paired_differences.csv` | 同一データ・同一費用での FoD − 比較法 の差 |
| `summary/macro_paired.csv` | 汚染ありの全設定が揃った反復ブロックを用いる平均差・区間 |
| `summary/downstream_all.csv`・`downstream_summary.csv` | 補助的なhard除去／soft WLS後の推定精度 |
| `summary/figures/` | PNG・PDFの図。収束不合格の結果を性能図に混ぜない |
| `jobs/ID/scores.csv` | 観測ごとのスコア、順位、中央の除去集合、真の汚染ラベル |
| `jobs/ID/diagnostics.json` | H-B・頻度論的推定・Bootstrapの診断 |
| `jobs/ID/posterior/`・`bootstrap/` | 事後サンプル、推論キャッシュ、Bootstrapの標本と診断 |

`FoD − 比較法` の損失差は、負ならFoDの費用が小さいことを表します。確認件数を揃えた比較は `rule=matched_FoD_workload`、スコアから費用に応じて行動を決める比較は `rule=risk_proxy` です。両者を併せて見れば、差が順位から来るのか、確認へ回す判断から来るのかを確認できます。

診断は原則 `R-hat <= 1.01`、bulk/tail ESSともに400以上です。頻度論側は制約付きの停留性、外側の数値境界への到達を確認します。Bootstrapは不合格標本を捨てて残りだけで確率を作らず、全標本を保存し、いずれかが不合格ならその設定を共通の有効比較から外します。除外件数・条件の偏りは必ず `completion.csv` と併せて確認してください。

## 5. 集計だけやり直す・並列実行する

```bash
python scripts/run_additional_experiments.py plot --study both \
  --output-dir outputs/additional_full
```

`plot` は集計も行います。集計だけなら `collect` です。これらは保存済みの設定を読み、推論を実行しません。未実行設定があっても集計し、完了率を明示します。

外部プロセスで分割する場合は、各プロセスで `--workers 4 --worker-index 0` のように指定し、indexを0〜3で変えます。各プロセスでは `--finalize` を付けず、全部終わってから上記 `plot` を一度実行します。CPU・メモリ消費を考えてプロセス数を決めてください。`--limit 1` は各指定実験の先頭1設定だけを試すオプションで、本実験の反復数を減らす指定ではありません。特定の設定だけは `--job-id ID` で実行できます。

同じ設定で診断不合格の完了結果が出た場合、同じコマンドを再実行してもその結果を再利用します。サンプル数・初期値数を増やす修正は設定ファイルを変更し、新しい出力先で実行してください。既定のH-B修復は4000 drawsで不合格なら8000 drawsを試し、両方の診断を残します。

既存の**同じネイティブ推論**のキャッシュがある場合は、`--paper-cache /absolute/path/to/native_cache_root` で指定できます。データ、推論対象、事前分布、サンプラー設定などの一致を元のコードで検証できた推定だけを再利用します。旧CSVや図から推定を再構成することはありません。

## 6. 解釈上の要点

- FoDもBootstrapも、ここでの確率は「各サンプルの監査集合に入る割合」です。**真の汚染ラベルの事後確率とは限りません。** 費用に応じた方策はこの確率を代理として用い、性能をシミュレーションの真のラベルで別途評価します。
- 確認者は既定では真のラベルを正しく識別します。`costs.review_error_rates` に例えば `[0, 0.05, 0.1]` を指定すると、同じ方策を保ったまま、確認者の対称誤判定を費用に加える感度分析ができます。
- 頻度論的推定は同じDPD型リスクを最小化し、汚染割合は解析的にprofileします。大域最適解の保証はありません。初期値・到達した目的関数値・境界情報を保存します。
- **密度推定の尺度崩壊に注意**：頻度論側は尺度の下限0.05を設けています。回帰H-Bは元のコードにも0.05の下限制約がありますが、密度H-Bにはありません。密度で下限に達する解が多いときは、それをFoDの優位性の根拠にせず、`optimizer.scale_min` を変えた感度分析と併せて報告してください。係数・位置の上限50、尺度の上限10への到達は診断不合格です。
- 比較を有利にする条件の選別はコードで行いません。主解析のγ・費用は設定ファイルで事前に指定し、全条件の結果と不成立件数を残します。詳細な生成設定・評価定義は [ADDITIONAL_EXPERIMENTS_PROTOCOL.md](ADDITIONAL_EXPERIMENTS_PROTOCOL.md) に記載しています。
