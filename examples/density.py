"""Fit a diagonal-Gaussian Hölder–Bayes model to a contaminated sample."""
import numpy as np
from hbayes import fit_density

rng=np.random.default_rng(42)
x=np.vstack([rng.normal(size=(180,2)),rng.normal(5,1,size=(20,2))])
fit=fit_density(x,gamma=.5,output_dir="results/fits/density")
print(fit.summary())
print("Audit probabilities:",fit.outlier_probability)
