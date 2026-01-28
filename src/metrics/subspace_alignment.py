"""Subspace alignment metrics for comparing PCA solutions."""

import numpy as np
from typing import Optional


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces.

    Principal angles measure the similarity between subspaces.
    Zero angles indicate identical directions, pi/2 indicates orthogonality.

    Args:
        U: First orthonormal basis, shape (n_features, k) or (k, n_features).
        V: Second orthonormal basis, shape (n_features, k) or (k, n_features).

    Returns:
        Array of k principal angles in radians [0, pi/2], sorted ascending.
    """
    # Ensure column vectors (n_features, k)
    if U.shape[0] < U.shape[1]:
        U = U.T
    if V.shape[0] < V.shape[1]:
        V = V.T

    # Orthonormalize (in case inputs are not perfectly orthonormal)
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)

    # SVD of U^T @ V gives cos(angles)
    _, s, _ = np.linalg.svd(U.T @ V, full_matrices=False)

    # Clip for numerical stability
    s = np.clip(s, -1.0, 1.0)

    # Convert to angles
    angles = np.arccos(s)

    return np.sort(angles)


def subspace_distance(
    U: np.ndarray,
    V: np.ndarray,
    metric: str = 'grassmann',
) -> float:
    """Compute distance between subspaces.

    Args:
        U: First orthonormal basis.
        V: Second orthonormal basis.
        metric: Distance metric to use:
            - 'grassmann': Geodesic distance on Grassmann manifold
            - 'chordal': Chordal (Frobenius) distance
            - 'projection': Projection distance (sin of largest angle)
            - 'mean_angle': Mean of principal angles

    Returns:
        Distance value (0 = identical subspaces).
    """
    angles = principal_angles(U, V)

    if metric == 'grassmann':
        # Geodesic distance: sqrt(sum(angles^2))
        return np.sqrt(np.sum(angles ** 2))
    elif metric == 'chordal':
        # Chordal distance: sqrt(sum(sin^2(angles)))
        return np.sqrt(np.sum(np.sin(angles) ** 2))
    elif metric == 'projection':
        # Projection distance: sin of largest angle
        return np.sin(angles[-1]) if len(angles) > 0 else 0.0
    elif metric == 'mean_angle':
        # Mean principal angle
        return np.mean(angles)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def alignment_score(U: np.ndarray, V: np.ndarray) -> float:
    """Compute alignment score between subspaces.

    Returns a value in [0, 1] where 1 indicates perfect alignment.

    Args:
        U: First orthonormal basis.
        V: Second orthonormal basis.

    Returns:
        Alignment score (1 = perfect alignment, 0 = orthogonal).
    """
    angles = principal_angles(U, V)

    # Average cosine of angles
    return np.mean(np.cos(angles))


def component_correlation(
    U: np.ndarray,
    V: np.ndarray,
    match_sign: bool = True,
) -> np.ndarray:
    """Compute correlation between corresponding components.

    Args:
        U: First set of components, shape (k, n_features).
        V: Second set of components, shape (k, n_features).
        match_sign: If True, take absolute value (sign ambiguity in PCA).

    Returns:
        Array of correlations for each component pair.
    """
    # Ensure row vectors
    if U.shape[0] > U.shape[1]:
        U = U.T
    if V.shape[0] > V.shape[1]:
        V = V.T

    k = min(U.shape[0], V.shape[0])
    correlations = np.zeros(k)

    for i in range(k):
        corr = np.dot(U[i], V[i]) / (np.linalg.norm(U[i]) * np.linalg.norm(V[i]) + 1e-10)
        if match_sign:
            corr = np.abs(corr)
        correlations[i] = corr

    return correlations


def subspace_overlap(U: np.ndarray, V: np.ndarray) -> float:
    """Compute overlap between two subspaces.

    Measures what fraction of variance in U is captured by V.

    Args:
        U: First orthonormal basis.
        V: Second orthonormal basis.

    Returns:
        Overlap value in [0, 1].
    """
    # Ensure column vectors
    if U.shape[0] < U.shape[1]:
        U = U.T
    if V.shape[0] < V.shape[1]:
        V = V.T

    # Project U onto V and measure preserved norm
    # ||P_V @ U||_F^2 / ||U||_F^2 where P_V = V @ V^T
    projection = V @ (V.T @ U)
    overlap = np.linalg.norm(projection, 'fro') ** 2 / (np.linalg.norm(U, 'fro') ** 2 + 1e-10)

    return overlap


def angle_to_degrees(angles: np.ndarray) -> np.ndarray:
    """Convert angles from radians to degrees."""
    return np.degrees(angles)
