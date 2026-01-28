"""P-COV: Pooled Covariance distributed PCA implementation.

This is an exact method that aggregates weighted covariance matrices from
each client to compute the global PCA solution.
"""

from typing import List, Dict, Optional
import numpy as np
from .base import DistributedPCABase


class PooledCovariancePCA(DistributedPCABase):
    """Pooled Covariance (P-COV) distributed PCA.

    This method computes the exact global PCA by:
    1. Each client computes local mean and covariance matrix
    2. Server aggregates means (weighted by sample count)
    3. Server aggregates covariances using the parallel covariance formula
    4. Server performs eigendecomposition on global covariance

    This produces the same result as centralized PCA on pooled data.
    Communication: 2 rounds, O(d^2) per client for covariance.
    """

    def __init__(self, n_components: int, random_state: Optional[int] = None):
        super().__init__(n_components, random_state)
        self._n_clients = 0
        self._communication_bytes = 0

    def fit(self, client_data: List[np.ndarray]) -> 'PooledCovariancePCA':
        """Fit P-COV on data from multiple clients.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            self: Fitted model.
        """
        self._n_clients = len(client_data)

        # Phase 1: Local computations
        local_results = [self.local_computation(data) for data in client_data]

        # Phase 2: Aggregation
        self.aggregate(local_results)

        return self

    def local_computation(self, data: np.ndarray) -> Dict:
        """Compute local mean and covariance at a client.

        Args:
            data: Local data array of shape (n_samples, n_features).

        Returns:
            Dictionary with local mean, covariance, and sample count.
        """
        n_samples = len(data)
        n_features = data.shape[1]

        # Local mean
        local_mean = np.mean(data, axis=0)

        # Local covariance (using n, not n-1 for consistency)
        centered = data - local_mean
        local_cov = (centered.T @ centered) / n_samples

        # Track communication cost
        self._communication_bytes += (n_features + n_features * n_features) * 8  # float64

        return {
            'mean': local_mean,
            'cov': local_cov,
            'n_samples': n_samples,
        }

    def aggregate(self, local_results: List[Dict]) -> None:
        """Aggregate local results using parallel covariance formula.

        The parallel covariance formula allows combining sample means
        and covariances from multiple sources:

        Combined mean: mu = sum(n_k * mu_k) / N
        Combined cov: C = sum(n_k * (C_k + (mu_k - mu)(mu_k - mu)^T)) / N

        Args:
            local_results: List of dictionaries from local_computation.
        """
        # Total sample count
        total_samples = sum(r['n_samples'] for r in local_results)
        n_features = local_results[0]['mean'].shape[0]

        # Weighted mean
        self.mean_ = np.zeros(n_features)
        for r in local_results:
            self.mean_ += r['n_samples'] * r['mean']
        self.mean_ /= total_samples

        # Aggregate covariance using parallel formula
        global_cov = np.zeros((n_features, n_features))
        for r in local_results:
            weight = r['n_samples'] / total_samples
            # Local covariance contribution
            global_cov += weight * r['cov']
            # Correction for different means
            mean_diff = r['mean'] - self.mean_
            global_cov += weight * np.outer(mean_diff, mean_diff)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(global_cov)

        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store top k components (as row vectors)
        self.components_ = eigenvectors[:, :self.n_components].T

        # Compute explained variance
        self._compute_explained_variance(eigenvalues)

    def get_communication_cost(self) -> Dict[str, int]:
        """Get communication cost metrics for P-COV.

        Returns:
            Dictionary with communication statistics.
        """
        n_features = self.components_.shape[1] if self.components_ is not None else 0

        return {
            'rounds': 2,  # Send means, then send covariances
            'bytes_per_client': (n_features + n_features * n_features) * 8,
            'total_bytes': self._communication_bytes,
            'n_clients': self._n_clients,
        }
