"""AP-STACK: Approximate Stack distributed PCA implementation.

This is an approximate method that stacks local eigenvectors and applies
PCA to find the global subspace approximation.
"""

from typing import List, Dict, Optional
import numpy as np
from .base import DistributedPCABase


class ApproximateStackPCA(DistributedPCABase):
    """Approximate Stack (AP-STACK) distributed PCA.

    This method approximates global PCA by:
    1. Each client computes local PCA to get top-r eigenvectors U_k
    2. Server stacks all local eigenvectors: M = [U_1 | U_2 | ... | U_K]
    3. Server applies PCA to M to get global eigenvectors

    This is an approximate method that works well when data is IID,
    but degrades with heterogeneous data distributions.

    Communication: 1 round, O(d*r) per client.
    """

    def __init__(
        self,
        n_components: int,
        local_components: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize AP-STACK.

        Args:
            n_components: Number of global principal components.
            local_components: Number of local components per client.
                             If None, uses n_components.
            random_state: Random seed for reproducibility.
        """
        super().__init__(n_components, random_state)
        self.local_components = local_components or n_components
        self._n_clients = 0
        self._communication_bytes = 0

    def fit(self, client_data: List[np.ndarray]) -> 'ApproximateStackPCA':
        """Fit AP-STACK on data from multiple clients.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            self: Fitted model.
        """
        self._n_clients = len(client_data)

        # Compute global mean (for transform)
        total_samples = sum(len(d) for d in client_data)
        local_means = [np.mean(d, axis=0) * len(d) for d in client_data]
        self.mean_ = sum(local_means) / total_samples

        # Phase 1: Local PCA at each client
        local_results = [self.local_computation(data) for data in client_data]

        # Phase 2: Stack and aggregate
        self.aggregate(local_results)

        return self

    def local_computation(self, data: np.ndarray) -> Dict:
        """Compute local PCA at a client.

        Args:
            data: Local data array of shape (n_samples, n_features).

        Returns:
            Dictionary with local eigenvectors and singular values.
        """
        n_samples = len(data)
        n_features = data.shape[1]

        # Center locally
        local_mean = np.mean(data, axis=0)
        centered = data - local_mean

        # Local PCA via SVD
        # For efficiency, use truncated SVD if n_samples < n_features
        if n_samples < n_features:
            # Use data matrix directly
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            local_components = Vt[:self.local_components].T  # (d, r)
            local_singular_values = s[:self.local_components]
        else:
            # Use covariance matrix
            cov = (centered.T @ centered) / n_samples
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            local_components = eigenvectors[:, idx[:self.local_components]]
            local_singular_values = np.sqrt(np.maximum(eigenvalues[idx[:self.local_components]], 0))

        # Track communication
        self._communication_bytes += n_features * self.local_components * 8

        return {
            'components': local_components,  # (d, r)
            'singular_values': local_singular_values,
            'n_samples': n_samples,
            'local_mean': local_mean,
        }

    def aggregate(self, local_results: List[Dict]) -> None:
        """Stack local eigenvectors and compute global PCA.

        Args:
            local_results: List of dictionaries from local_computation.
        """
        # Stack local eigenvectors (weighted by singular values)
        stacked = []
        for r in local_results:
            # Weight by singular values to preserve importance
            weighted = r['components'] * r['singular_values']
            stacked.append(weighted)

        # Concatenate: (d, K*r)
        M = np.hstack(stacked)

        # Apply PCA to stacked matrix
        # SVD of M gives principal directions
        U, s, Vt = np.linalg.svd(M, full_matrices=False)

        # Take top n_components
        self.components_ = U[:, :self.n_components].T  # (k, d)

        # Approximate eigenvalues
        eigenvalues = (s[:self.n_components] ** 2) / len(local_results)
        self._compute_explained_variance(eigenvalues)

    def get_communication_cost(self) -> Dict[str, int]:
        """Get communication cost metrics for AP-STACK.

        Returns:
            Dictionary with communication statistics.
        """
        n_features = self.components_.shape[1] if self.components_ is not None else 0

        return {
            'rounds': 1,
            'bytes_per_client': n_features * self.local_components * 8,
            'total_bytes': self._communication_bytes,
            'n_clients': self._n_clients,
        }
