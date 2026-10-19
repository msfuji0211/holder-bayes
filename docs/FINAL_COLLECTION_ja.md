# 保存済み結果から最終集計を作る

`scripts/run_final_collection.py` と `scripts/run_final_collection.sh` は、保存済み結果から `mcmc_final_collection` 相当のフォルダを生成します。新規MCMC、データの再生成、指標の再評価は行いません。論文実験を最初から実行する入口は `hbayes paper run` です。

## 準備

この配布フォルダで実行します。集計・描画だけなら推論ライブラリは不要です。

```bash
python -m pip install -e ".[test]"
```

Python 3.12以上を使ってください。論文の記録環境はPython 3.12.8です。実験全体を再実行する場合は `.[experiments]` をインストールします。

## 以前の実験フォルダから生成する

次は以前のフォルダ名を使った例です。パスを実際の保存先に置き換えてください。相対パスはコマンドを実行した作業ディレクトリを基準に解釈します。

```bash
bash scripts/run_final_collection.sh \
  --results-root /path/to/h_bayes/outputs/temperature_gamma_nh_v3 \
  --synthetic-root /path/to/h_bayes/outputs/mcmc_synthetic_evaluation_rebuild_v1 \
  --real-root /path/to/h_bayes/outputs/mcmc_evaluation_rebuild_v1 \
  --output-root /path/to/h_bayes/outputs/mcmc_final_collection
```

入力だけ確認する場合は、末尾に `--dry-run` を付けます。`--results-root` は元の全実験結果です。syntheticでは `--synthetic-root` に存在する指標を、benchmark/cleaningでは `--real-root` に存在する指標を優先し、そこにない設定は元の結果から補います。優先した指標が診断不通過なら停止します。

全設定が一つのフォルダに揃っている場合は、追加の2引数を省略できます。

```bash
bash scripts/run_final_collection.sh \
  --results-root /path/to/complete_saved_results \
  --output-root outputs/mcmc_final_collection
```

Pythonから直接実行する場合は、`bash scripts/run_final_collection.sh` を `python scripts/run_final_collection.py` に置き換えてください。シェルスクリプトはbashで起動します。zshからも上の `bash ...` を実行できます。`PYTHON=/path/to/python` を先頭に付けるとPythonを指定できます。

## 新しい公開版の出力から生成する

`runs/` がある場合は自動的にその中を読みます。元の出力とは別の場所を指定してください。

```bash
hbayes paper collect \
  --results-root results/paper \
  --output-root results/collection
```

`hbayes paper run --phase collect` は新規実験の同じ出力フォルダ内で行う集計です。`hbayes paper collect` は、指定した保存済み結果を別フォルダに統合する入口です。

## 必要な入力と生成物

- 指標：全1,170設定の `metrics.json`。内訳はdensity 450、regression 540、benchmark 90、cleaning 90です。syntheticは各30反復、実データは各5分割を保持します。
- 描画：少なくともdensityのγ=0.5とregressionのγ=1.0について、汚染率0.2・反復0の `samples.npz`。事後標本に加えて、保存済み観測・真値・外れ値ラベル・FoD配列が必要です。CSVだけからFigure 3は生成できません。
- 出力：各設定の指標、20個のnative集計CSV、`summaries/` の20個の公開用CSV、`tables/` の表、`figures/main/` の主図3枚、`figures/appendix/` の追加図8枚、`final_collection.json`。

主図名は `fig1_posterior_regression_simulation.png`、`fig2_accuracy_regression_simulation.png`、`fig3_likelihood_selection_combined.png` です。Figure 1は回帰の切片を含み、Figure 3の事後区間は `95% CrI for K` と表示します。

結果ZIPなどに代表標本が含まれていない場合は、元の実験フォルダを入力にしてください。指標だけ集計する場合は `--no-plots` を付けます。

```bash
bash scripts/run_final_collection.sh \
  --results-root /path/to/saved_metrics \
  --output-root outputs/mcmc_final_collection \
  --no-plots
```

診断は保存されたR-hat・ESS・発散数などの確認で、標本からの再計算ではありません。メトロポリス法にはNUTSの発散数を要求しません。必要な設定の欠落や数値診断の不通過があれば、反復を除外せず停止します。

元の結果には書き込みません。新しい空フォルダを出力先に指定してください。同じ入力なら同じ出力先で再実行できます。入力を変更した場合は別の出力先を使います。別フォルダの標本を補う場合は、その横の `metrics.json` が採用した指標と一致することを要求し、異なるfitの指標と標本の混在を防ぎます。

過去の参照CSV、`reference/`、`provenance/`、固定のチェックサム一覧に依存しません。処理時に生成するハッシュは、今回指定した入力の変更検出に使います。

## JRSSB用のPNG・PDF両出力（2026-10-19）

既存の最終集計コマンドは、主図3枚と付録図8枚を、それぞれ600 dpiのPNGとベクトルPDFで保存します。パネル記号は左上の大文字A, B, C, ...に統一し、色バーには付けません。Figure 1は上段A–B、下段C–E、Figures 2–3は上段A–B、下段C–Dです。Figure 3の `95% CrI for K` 表記は維持します。

保存名のstemは従来と同じで、`.png` に加えて `.pdf` ができます。公開用の短いファイル名へ整理する処理もPDFを保持します。PDFは元のMatplotlib描画から直接生成し、画像を貼り付けたPDFにはしません。

```bash
bash scripts/run_final_collection.sh \
  --results-root /path/to/existing/results \
  --output-root results/collection_jrssb
```

必要に応じて既存の `--synthetic-root` と `--real-root` も指定してください。このコマンドは保存済み結果を集計・描画するもので、MCMCや指標の再評価を実行しません。既存の集計対象・採用反復・数値・FoD・監査予算は変更していません。

TeX用には `figures/main/` の主図PDFを論文の `fig/` にコピーし、同名の `.pdf` を参照してください。本文用の代替テキストと表の末尾配置はTeX側で設定します。
