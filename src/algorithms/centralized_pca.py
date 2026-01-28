"""Centralized PCA baseline implementation."""

from typing import List, Dict, Optional
import numpy as np
from .base import DistributedPCABase


class CentralizedPCA(DistributedPCABase):
    """Centralized PCA baseline for comparison.

    This is the standard PCA computed on all data pooled together,
    serving as the ground truth for evaluating distributed methods.
    """

    def __init__(self, n_components: int, random_state: Optional[int] = None):
        super().__init__(n_components, random_state)

    def fit(self, client_data: List[np.ndarray]) -> 'CentralizedPCA':
        """Fit PCA on pooled data from all clients.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            self: Fitted model.
        """
        # Pool all data
        all_data = np.vstack(client_data)

        # Compute global mean
        self.mean_ = np.mean(all_data, axis=0)

        # Center data
        centered = all_data - self.mean_

        # Compute covariance matrix
        n_samples = len(all_data)
        cov = (centered.T @ centered) / n_samples

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store top k components (as row vectors)
        self.components_ = eigenvectors[:, :self.n_components].T

        # Compute explained variance
        self._compute_explained_variance(eigenvalues)

        return self

    def local_computation(self, data: np.ndarray) -> Dict:
        """Not used for centralized PCA."""
        return {'data': data}

    def aggregate(self, local_results: List[Dict]) -> None:
        """Not used for centralized PCA."""
        pass

    def fit_single(self, data: np.ndarray) -> 'CentralizedPCA':
        """Fit PCA on a single dataset.

        Args:
            data: Data array of shape (n_samples, n_features).

        Returns:
            self: Fitted model.
        """
        return self.fit([data])
