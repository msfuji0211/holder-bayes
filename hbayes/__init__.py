"""Hölder–Bayes inference and probabilistic outlier auditing."""
from .api import FitResult,SamplerConfig,audit_probabilities,fit_density,fit_regression

__version__="0.1.0"
__all__=["FitResult","SamplerConfig","audit_probabilities","fit_density","fit_regression"]
