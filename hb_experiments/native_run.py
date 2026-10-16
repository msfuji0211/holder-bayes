"""Run the paper workflow using generated synthetic data and raw real data."""
import sys
import numpy as np


def validate_dataset(name, X, y):
    shapes = {'cal_housing': (20640, 8), 'pumadyn32fh': (8192, 32), 'kin8nm': (8192, 8)}
    expected = shapes[name]
    X, y = np.asarray(X), np.asarray(y)
    if X.shape != expected or y.shape != (expected[0],) or not all(np.isfinite(v).all() for v in [X, y]):
        raise ValueError(f'{name} needs finite X{expected} and y({expected[0]},); received X{X.shape}, y{y.shape}. '
                         'Supply the paper dataset with --local-data-dir.')
    return X, y


def main():
    from experiments import run_temperature_workflow as workflow
    from experiments.exp1_regression import run_exp1_realdata_benchmark as benchmark
    args = workflow.parse(sys.argv[1:])
    original_loader = benchmark.load_benchmark_dataset
    def checked_loader(name, local_data_dir=None):
        return validate_dataset(name, *original_loader(name, local_data_dir))
    benchmark.load_benchmark_dataset = checked_loader
    try:
        for name in args.datasets:
            cached = args.cache_dir/'datasets'/(name+'.npz')
            if cached.is_file():
                with np.load(cached, allow_pickle=False) as data:
                    validate_dataset(name, data['X'], data['y'])
        workflow.main(sys.argv[1:])
    finally:
        benchmark.load_benchmark_dataset = original_loader


if __name__ == '__main__':
    main()
