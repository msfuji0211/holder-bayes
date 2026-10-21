# Hölder–Bayes

提案手法Hölder–BayesのPython実装です。密度推定・切片付き回帰を当てはめ、汚染割合と観測別の監査確率を推定します。論文実験のデータ生成・推定・集計・描画コードも含みます。保存済みの標本、結果CSV、参照画像、過去のチェックサム一覧は同梱も必須参照もしません。

## 手法を使う

Python 3.12の環境で、このフォルダからインストールします。

```bash
python -m pip install -e ".[inference]"
```

```python
from hbayes import fit_density, fit_regression

fit = fit_density(x, gamma=0.5, output_dir="results/fits/density")
print(fit.outlier_probability)
print(fit.summary())

reg = fit_regression(X, y, gamma=1.0, output_dir="results/fits/regression")
print(reg.coefficient_samples.mean(axis=(0, 1)))  # 切片、傾き
print(reg.predict(X_new))
```

`X` は切片列を含まない共変量行列です。切片の追加・訓練データによる標準化を内部で行い、係数・スケール・予測値は元の単位で返します。`examples/density.py` と `examples/regression.py` はデータ生成から実行する例です。どちらもMCMCを実行します。

保存ファイルは `posterior_samples.npz`、`summary.json`、`audit_scores.csv` です。`SamplerConfig` でchain数・warm-up・標本数・シードを指定できます。診断不通過も含めて結果と全標本を返すため、解釈前に診断を確認してください。観測別確率は、この観測集合でのbottom-K選択の事後頻度です。

## 論文実験を実行する

論文の環境はPython 3.12.8です。元の出力フォルダや保存標本は不要です。

```bash
python -m pip install -e ".[experiments]"
hbayes paper plan
hbayes paper run --download-data --dry-run
hbayes paper run --download-data
```

synthetic生成・実データ取得から、MCMC、診断、集計、描画へ進みます。実データを手元に持っている場合は `--download-data` を `--local-data-dir /path/to/datasets` に置き換えられます。診断不通過の設定は保存し、修正されるまで最終集計を停止します。

保存先は `results/paper/`、主図名は `posterior.png`（切片を含む）、`accuracy.png`、`audit_scores.png` です。CSVは `summaries/`、図は `figures/` にまとめます。観測・checkpointの対応を保つため、内部の実行IDは維持しています。

```bash
hbayes paper pack --archive paper-results.zip
```

このZIPには指標・集計・図を収録し、キャッシュは含めません。`--include-samples` を付けると各実行の保存標本も収録します。

## 検証・開発

```bash
python -m pip install -e ".[test]"
hbayes check
python -m unittest discover -s tests -v
```

`hbayes check` はソースと設定の配置を確認し、`hbayes doctor` は推論環境を表示します。過去の結果ファイルは読みません。テストデータも一時フォルダにコードから生成します。集計・11図の描画・表・ZIP作成の統合テストではMCMCを実行しません。推論環境で短い実推論も確認する場合は、次を実行できます。

```bash
HBAYES_TEST_MCMC=1 python -m unittest discover -s tests -p test_inference_smoke.py -v
```

表は今回生成したCSVから `results/paper/tables/` に書き出します。`hbayes paper tables` で推定なしに再出力できます。各条件で最良の表示値と同順位を太字にし、oracleは実用手法の強調対象から外します。

実行時に生成する入力・キャッシュのハッシュは、破損や異なる設定での誤再開を検出するために維持しています。過去の検証済み結果のチェックサムとは独立です。既存の検証済み結果は必要に応じて別の成果物として管理でき、GitHubへの同梱は不要です。

保存済み結果を別フォルダに統合し、`mcmc_final_collection` 相当のCSV・図を作る場合は `hbayes paper collect` または `scripts/run_final_collection.sh` を使えます。MCMCは実行しません。旧形式の入力も指定できます。手順は [docs/FINAL_COLLECTION_ja.md](docs/FINAL_COLLECTION_ja.md) にあります。

手法は [docs/API.md](docs/API.md)、実験手順は [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)、保存形式は [docs/OUTPUTS.md](docs/OUTPUTS.md) を参照してください。固定された過去の修正対象を必要とする推論修正スクリプトは公開版から外しました。新規実験の診断不通過は、今回生成されるキューに記録されます。

全1,170設定のMCMCはこの公開準備時には再実行していません。GitHubへのpushやPyPIへの登録も行っていません。

## ライセンスと引用

本ソフトウェアは [MIT License](LICENSE) の下で公開されています。公開前に正式な論文引用を追加してください。
