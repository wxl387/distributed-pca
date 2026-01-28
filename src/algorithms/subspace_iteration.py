"""SUB-IT: Subspace Iteration distributed PCA implementation.

This is an exact iterative method that converges to the global PCA solution
through repeated power iteration steps across clients.
"""

from typing import List, Dict, Optional
import numpy as np
from .base import DistributedPCABase


class SubspaceIterationPCA(DistributedPCABase):
    """Subspace Iteration (SUB-IT) distributed PCA.

    This method computes PCA through iterative refinement:
    1. Server initializes random orthonormal matrix V
    2. Each iteration:
       a. Server broadcasts V to all clients
       b. Each client computes Z_k = X_k^T @ (X_k @ V)
       c. Server aggregates Z = sum(Z_k)
       d. Server orthonormalizes: V_new = QR(Z)
    3. Repeat until convergence

    Communication: Multiple rounds until convergence, O(d*r) per client per round.
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        """Initialize SUB-IT.

        Args:
            n_components: Number of principal components.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance (change in subspace).
            random_state: Random seed for reproducibility.
        """
        super().__init__(n_components, random_state)
        self.max_iter = max_iter
        self.tol = tol
        self._n_iterations = 0
        self._n_clients = 0
        self._convergence_history = []

    def fit(self, client_data: List[np.ndarray]) -> 'SubspaceIterationPCA':
        """Fit SUB-IT on data from multiple clients.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            self: Fitted model.
        """
        self._n_clients = len(client_data)
        n_features = client_data[0].shape[1]
        total_samples = sum(len(d) for d in client_data)

        # Compute global mean first (one round of communication)
        local_means = [np.mean(d, axis=0) for d in client_data]
        sample_counts = [len(d) for d in client_data]
        self.mean_ = sum(n * m for n, m in zip(sample_counts, local_means)) / total_samples

        # Center data locally
        centered_data = [d - self.mean_ for d in client_data]

        # Initialize random orthonormal matrix
        rng = np.random.RandomState(self.random_state)
        V = rng.randn(n_features, self.n_components)
        V, _ = np.linalg.qr(V)

        self._convergence_history = []

        # Iterative refinement
        for iteration in range(self.max_iter):
            V_old = V.copy()

            # Distributed power iteration step
            Z = np.zeros((n_features, self.n_components))
            for data in centered_data:
                # Each client: Z_k = X_k^T @ (X_k @ V)
                proj = data @ V  # (n_k, r)
                Z += data.T @ proj  # (d, r)

            # Orthonormalize
            V, R = np.linalg.qr(Z)

            # Check convergence (change in subspace)
            # Use ||V @ V^T - V_old @ V_old^T||_F
            change = np.linalg.norm(V @ V.T - V_old @ V_old.T, 'fro')
            self._convergence_history.append(change)

            if change < self.tol:
                break

        self._n_iterations = iteration + 1

        # Store components (as row vectors)
        self.components_ = V.T

        # Compute eigenvalues from final iteration
        # Lambda = diag(V^T @ C @ V) where C is the covariance
        eigenvalues = np.diag(R) ** 2 / total_samples

        # Sort by eigenvalues (should already be sorted, but ensure)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        self.components_ = self.components_[idx]

        self._compute_explained_variance(eigenvalues)

        return self

    def local_computation(self, data: np.ndarray, V: np.ndarray) -> Dict:
        """Perform local computation at a client for one iteration.

        Args:
            data: Centered local data array.
            V: Current subspace estimate from server.

        Returns:
            Dictionary with local result Z_k.
        """
        proj = data @ V
        Z = data.T @ proj
        return {'Z': Z, 'n_samples': len(data)}

    def aggregate(self, local_results: List[Dict]) -> np.ndarray:
        """Aggregate local Z matrices and orthonormalize.

        Args:
            local_results: List of dictionaries with Z matrices.

        Returns:
            Updated V matrix after orthonormalization.
        """
        Z = sum(r['Z'] for r in local_results)
        V, _ = np.linalg.qr(Z)
        return V

    def get_communication_cost(self) -> Dict[str, int]:
        """Get communication cost metrics for SUB-IT.

        Returns:
            Dictionary with communication statistics.
        """
        n_features = self.components_.shape[1] if self.components_ is not None else 0

        return {
            'rounds': self._n_iterations + 1,  # +1 for initial mean computation
            'bytes_per_client_per_round': n_features * self.n_components * 8,
            'total_bytes': (self._n_iterations + 1) * self._n_clients * n_features * self.n_components * 8,
            'n_clients': self._n_clients,
            'n_iterations': self._n_iterations,
            'convergence_history': self._convergence_history,
        }
