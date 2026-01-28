"""Reconstruction error metrics for PCA evaluation."""

import numpy as np
from typing import Union


def reconstruction_error(
    X: np.ndarray,
    pca_model,
    metric: str = 'mse',
) -> float:
    """Compute reconstruction error.

    Args:
        X: Original data array of shape (n_samples, n_features).
        pca_model: Fitted PCA model with transform and inverse_transform.
        metric: Error metric to use:
            - 'mse': Mean Squared Error
            - 'rmse': Root Mean Squared Error
            - 'mae': Mean Absolute Error

    Returns:
        Reconstruction error value.
    """
    X_transformed = pca_model.transform(X)
    X_reconstructed = pca_model.inverse_transform(X_transformed)

    diff = X - X_reconstructed

    if metric == 'mse':
        return np.mean(diff ** 2)
    elif metric == 'rmse':
        return np.sqrt(np.mean(diff ** 2))
    elif metric == 'mae':
        return np.mean(np.abs(diff))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def relative_reconstruction_error(
    X: np.ndarray,
    pca_model,
) -> float:
    """Compute relative reconstruction error.

    Args:
        X: Original data array of shape (n_samples, n_features).
        pca_model: Fitted PCA model.

    Returns:
        Relative error: ||X - X_hat|| / ||X||
    """
    X_transformed = pca_model.transform(X)
    X_reconstructed = pca_model.inverse_transform(X_transformed)

    error_norm = np.linalg.norm(X - X_reconstructed, 'fro')
    data_norm = np.linalg.norm(X, 'fro')

    return error_norm / (data_norm + 1e-10)


def per_sample_reconstruction_error(
    X: np.ndarray,
    pca_model,
) -> np.ndarray:
    """Compute reconstruction error for each sample.

    Args:
        X: Original data array of shape (n_samples, n_features).
        pca_model: Fitted PCA model.

    Returns:
        Array of per-sample errors, shape (n_samples,).
    """
    X_transformed = pca_model.transform(X)
    X_reconstructed = pca_model.inverse_transform(X_transformed)

    return np.mean((X - X_reconstructed) ** 2, axis=1)


def reconstruction_error_ratio(
    X: np.ndarray,
    pca_model_test,
    pca_model_baseline,
) -> float:
    """Compute ratio of reconstruction errors between two models.

    Args:
        X: Test data array.
        pca_model_test: Model being evaluated.
        pca_model_baseline: Baseline model for comparison.

    Returns:
        Ratio: error(test) / error(baseline). Values close to 1 are good.
    """
    error_test = reconstruction_error(X, pca_model_test)
    error_baseline = reconstruction_error(X, pca_model_baseline)

    return error_test / (error_baseline + 1e-10)
