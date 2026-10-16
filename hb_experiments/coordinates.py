"""NumPy-only plotting transforms, matching the inference data coordinates."""
import numpy as np
jnp = np

def standardize_regression_data(x_train, y_train, x_test):
    """Standardize features and response using training statistics."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std < 1e-9, 1.0, x_std)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-9:
        y_std = 1.0

    x_train_scaled = (x_train - x_mean) / x_std
    x_test_scaled = (x_test - x_mean) / x_std
    y_train_scaled = (y_train - y_mean) / y_std

    x_train_design = np.hstack([np.ones((x_train.shape[0], 1)), x_train_scaled])
    x_test_design = np.hstack([np.ones((x_test.shape[0], 1)), x_test_scaled])

    scaler = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    return (
        jnp.asarray(x_train_design, dtype=jnp.float32),
        jnp.asarray(y_train_scaled, dtype=jnp.float32),
        jnp.asarray(x_test_design, dtype=jnp.float32),
        scaler,
    )

def recover_posterior_params_original_scale(samples, scaler):
    """Convert posterior samples from standardized regression scale back to original scale."""
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return {"beta": np.empty((0, 0)), "sigma": np.empty((0,))}

    d_in = len(scaler["x_mean"]) + 1
    beta_scaled = samples[:, :d_in]
    log_scale = samples[:, d_in]

    slopes = scaler["y_std"] * beta_scaled[:, 1:] / scaler["x_std"][None, :]
    intercept = scaler["y_mean"] + scaler["y_std"] * beta_scaled[:, 0] - np.sum(
        slopes * scaler["x_mean"][None, :], axis=1
    )
    beta_orig = np.column_stack([intercept, slopes])
    sigma_orig = scaler["y_std"] * np.exp(log_scale)
    return {"beta": beta_orig, "sigma": sigma_orig}
