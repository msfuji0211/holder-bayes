"""Fit robust regression; X contains covariates without an intercept column."""
import numpy as np
from hbayes import fit_regression

rng=np.random.default_rng(42)
X=rng.normal(size=(200,2))
y=1.+X@np.array([1.,-1.5])+rng.normal(size=200)
y[-40:]+=6.
fit=fit_regression(X,y,gamma=1.,output_dir="results/fits/regression")
print(fit.summary())
print("Intercept and slopes:",fit.coefficient_samples.mean(axis=(0,1)))
print("Predictions:",fit.predict(X[:5]))
