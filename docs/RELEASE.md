# Source release

Place the contents of `holder-bayes/` at the repository root. Check installation, the model examples, the dry-run experiment plan, and the tests in README.md. The public distribution/module names are `holder-bayes` and `hbayes`; paper artifacts use `results/paper/` and `paper-results.zip`.

Existing output archives and archived observation snapshots are optional. Publish implementation code, configurations, environment records, examples, tests, and usage instructions. Generated fits and caches are excluded by `.gitignore`; attach a result archive separately when useful.

The software is released under the MIT License (`LICENSE`). Cite the accompanying paper ([arXiv:2607.25665](https://arxiv.org/abs/2607.25665)); see README.md for BibTeX. The prepared source package has not been registered on PyPI or pushed to GitHub. Version `0.1.0` identifies this initial source package.

Mathematical targets and sampler kernels remain byte-identical to the reviewed implementation. The public source tree contains no samples, result CSVs, figures or historical checksum inventories. Run-specific contracts and checksums are generated under the selected output directory. Fixed historical recovery tools are excluded. The numerical MH and predictive kernels remain available as source.
