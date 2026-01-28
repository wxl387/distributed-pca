"""AP-COV: Approximate Covariance distributed PCA implementation.

This is an approximate method that simply averages local covariance
matrices without the correction term for different means.
"""

from typing import List, Dict, Optional
import numpy as np
from .base import DistributedPCABase


class ApproximateCovPCA(DistributedPCABase):
    """Approximate Covariance (AP-COV) distributed PCA.

    This method approximates global PCA by:
    1. Each client computes local covariance matrix (centered locally)
    2. Server averages the covariance matrices (simple average or weighted)
    3. Server performs eigendecomposition

    This is simpler than P-COV but doesn't account for different local
    means, making it approximate. Works well for IID data but degrades
    with heterogeneous distributions.

    Communication: 1 round, O(d^2) per client.
    """

    def __init__(
        self,
        n_components: int,
        weighted: bool = True,
        random_state: Optional[int] = None,
    ):
        """Initialize AP-COV.

        Args:
            n_components: Number of principal components.
            weighted: If True, weight covariances by sample count.
            random_state: Random seed for reproducibility.
        """
        super().__init__(n_components, random_state)
        self.weighted = weighted
        self._n_clients = 0
        self._communication_bytes = 0

    def fit(self, client_data: List[np.ndarray]) -> 'ApproximateCovPCA':
        """Fit AP-COV on data from multiple clients.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            self: Fitted model.
        """
        self._n_clients = len(client_data)

        # Phase 1: Local computation
        local_results = [self.local_computation(data) for data in client_data]

        # Phase 2: Aggregation
        self.aggregate(local_results)

        return self

    def local_computation(self, data: np.ndarray) -> Dict:
        """Compute local covariance at a client.

        Note: Uses local centering, not global centering.

        Args:
            data: Local data array of shape (n_samples, n_features).

        Returns:
            Dictionary with local covariance and sample count.
        """
        n_samples = len(data)
        n_features = data.shape[1]

        # Local mean and centering
        local_mean = np.mean(data, axis=0)
        centered = data - local_mean

        # Local covariance
        local_cov = (centered.T @ centered) / n_samples

        # Track communication
        self._communication_bytes += (n_features * n_features + n_features) * 8

        return {
            'cov': local_cov,
            'mean': local_mean,
            'n_samples': n_samples,
        }

    def aggregate(self, local_results: List[Dict]) -> None:
        """Aggregate local covariances by averaging.

        Note: This doesn't use the correction term for different means,
        making it an approximate method.

        Args:
            local_results: List of dictionaries from local_computation.
        """
        n_features = local_results[0]['cov'].shape[0]
        total_samples = sum(r['n_samples'] for r in local_results)

        # Compute global mean for transform
        self.mean_ = np.zeros(n_features)
        for r in local_results:
            self.mean_ += r['n_samples'] * r['mean']
        self.mean_ /= total_samples

        # Average covariances (without correction for different means)
        global_cov = np.zeros((n_features, n_features))

        if self.weighted:
            # Weighted average by sample count
            for r in local_results:
                weight = r['n_samples'] / total_samples
                global_cov += weight * r['cov']
        else:
            # Simple average
            for r in local_results:
                global_cov += r['cov'] / len(local_results)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(global_cov)

        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store top k components
        self.components_ = eigenvectors[:, :self.n_components].T

        # Compute explained variance
        self._compute_explained_variance(eigenvalues)

    def get_communication_cost(self) -> Dict[str, int]:
        """Get communication cost metrics for AP-COV.

        Returns:
            Dictionary with communication statistics.
        """
        n_features = self.components_.shape[1] if self.components_ is not None else 0

        return {
            'rounds': 1,
            'bytes_per_client': (n_features * n_features + n_features) * 8,
            'total_bytes': self._communication_bytes,
            'n_clients': self._n_clients,
        }
