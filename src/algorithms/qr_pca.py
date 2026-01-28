"""QR-PCA: QR factorization based distributed PCA implementation.

This is an exact method that uses QR factorization for aggregating
local PCA results to obtain the global solution.
"""

from typing import List, Dict, Optional
import numpy as np
from .base import DistributedPCABase


class QRPCA(DistributedPCABase):
    """QR-PCA distributed PCA.

    This method computes exact global PCA by:
    1. Each client computes local SVD: X_k = U_k @ S_k @ V_k^T
    2. Server receives weighted local right singular vectors: sqrt(n_k/N) * V_k @ S_k
    3. Server stacks and performs QR factorization
    4. Server computes SVD on R to get global components

    This produces the exact global PCA solution.
    Communication: 1 round, O(d*r) per client.
    """

    def __init__(
        self,
        n_components: int,
        local_components: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize QR-PCA.

        Args:
            n_components: Number of global principal components.
            local_components: Number of local components per client.
                             If None, uses 2*n_components for better accuracy.
            random_state: Random seed for reproducibility.
        """
        super().__init__(n_components, random_state)
        self.local_components = local_components or (2 * n_components)
        self._n_clients = 0
        self._communication_bytes = 0

    def fit(self, client_data: List[np.ndarray]) -> 'QRPCA':
        """Fit QR-PCA on data from multiple clients.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            self: Fitted model.
        """
        self._n_clients = len(client_data)
        total_samples = sum(len(d) for d in client_data)

        # Compute global mean
        local_means = [np.mean(d, axis=0) * len(d) for d in client_data]
        self.mean_ = sum(local_means) / total_samples

        # Center all data with global mean
        centered_data = [d - self.mean_ for d in client_data]

        # Phase 1: Local computation
        local_results = [
            self.local_computation(data, total_samples)
            for data in centered_data
        ]

        # Phase 2: Aggregation
        self.aggregate(local_results, total_samples)

        return self

    def local_computation(
        self,
        data: np.ndarray,
        total_samples: int,
    ) -> Dict:
        """Compute local SVD at a client.

        Args:
            data: Centered local data array.
            total_samples: Total samples across all clients (for weighting).

        Returns:
            Dictionary with weighted local components.
        """
        n_samples = len(data)
        n_features = data.shape[1]

        # Determine number of components to compute
        k = min(self.local_components, n_samples - 1, n_features)

        # Local SVD
        U, s, Vt = np.linalg.svd(data, full_matrices=False)

        # Take top k components
        V = Vt[:k].T  # (d, k)
        singular_values = s[:k]

        # Weight by sqrt(n_k/N) for proper aggregation
        weight = np.sqrt(n_samples / total_samples)
        weighted_components = weight * V @ np.diag(singular_values)

        # Track communication
        self._communication_bytes += n_features * k * 8

        return {
            'weighted_components': weighted_components,  # (d, k)
            'n_samples': n_samples,
        }

    def aggregate(self, local_results: List[Dict], total_samples: int) -> None:
        """Aggregate using QR factorization.

        Args:
            local_results: List of dictionaries from local_computation.
            total_samples: Total samples across all clients.
        """
        # Stack all weighted components
        stacked = np.hstack([r['weighted_components'] for r in local_results])

        # QR factorization
        Q, R = np.linalg.qr(stacked)

        # SVD on R
        U_R, s, Vt = np.linalg.svd(R, full_matrices=False)

        # Global components: Q @ U_R
        global_components = Q @ U_R

        # Take top n_components
        self.components_ = global_components[:, :self.n_components].T  # (k, d)

        # Eigenvalues
        eigenvalues = (s[:self.n_components] ** 2) / total_samples
        self._compute_explained_variance(eigenvalues)

    def get_communication_cost(self) -> Dict[str, int]:
        """Get communication cost metrics for QR-PCA.

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
