"""Explained variance metrics for PCA evaluation."""

import numpy as np
from typing import Optional


def explained_variance_ratio(
    pca_model,
    X: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get explained variance ratio for each component.

    Args:
        pca_model: Fitted PCA model with explained_variance_ratio_ attribute.
        X: Optional data to compute variance from (if model doesn't store it).

    Returns:
        Array of variance ratios, shape (n_components,).
    """
    if hasattr(pca_model, 'explained_variance_ratio_') and pca_model.explained_variance_ratio_ is not None:
        return pca_model.explained_variance_ratio_

    if X is None:
        raise ValueError("Model doesn't have explained_variance_ratio_ and no data provided")

    # Compute from data
    X_centered = X - pca_model.mean_
    total_var = np.var(X_centered, axis=0).sum()

    X_transformed = pca_model.transform(X)
    component_vars = np.var(X_transformed, axis=0)

    return component_vars / (total_var + 1e-10)


def cumulative_explained_variance(
    pca_model,
    X: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get cumulative explained variance ratio.

    Args:
        pca_model: Fitted PCA model.
        X: Optional data for variance computation.

    Returns:
        Cumulative variance ratios, shape (n_components,).
    """
    ratios = explained_variance_ratio(pca_model, X)
    return np.cumsum(ratios)


def variance_retention(
    X: np.ndarray,
    pca_model,
) -> float:
    """Compute fraction of total variance retained after projection.

    Args:
        X: Data array.
        pca_model: Fitted PCA model.

    Returns:
        Variance retention ratio in [0, 1].
    """
    X_centered = X - pca_model.mean_
    total_var = np.var(X_centered, axis=0).sum()

    X_transformed = pca_model.transform(X)
    retained_var = np.var(X_transformed, axis=0).sum()

    return retained_var / (total_var + 1e-10)


def compare_explained_variance(
    pca_model_test,
    pca_model_baseline,
    X: np.ndarray,
) -> dict:
    """Compare explained variance between two PCA models.

    Args:
        pca_model_test: Model being evaluated.
        pca_model_baseline: Baseline model for comparison.
        X: Data for evaluation.

    Returns:
        Dictionary with variance comparison metrics.
    """
    var_test = explained_variance_ratio(pca_model_test, X)
    var_baseline = explained_variance_ratio(pca_model_baseline, X)

    cum_test = cumulative_explained_variance(pca_model_test, X)
    cum_baseline = cumulative_explained_variance(pca_model_baseline, X)

    return {
        'variance_ratio_test': var_test,
        'variance_ratio_baseline': var_baseline,
        'cumulative_test': cum_test,
        'cumulative_baseline': cum_baseline,
        'total_variance_ratio': cum_test[-1] / (cum_baseline[-1] + 1e-10),
        'mean_ratio_diff': np.mean(np.abs(var_test - var_baseline)),
    }
