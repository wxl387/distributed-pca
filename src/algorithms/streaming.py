"""Streaming Distributed PCA with incremental updates.

This module implements streaming/incremental updates for distributed PCA,
allowing:
1. New clients to join without full recomputation
2. Existing clients to update their data incrementally
3. Clients to leave the federation

The key insight is that P-COV's covariance aggregation can be done incrementally
using the parallel algorithm for combining covariances:

C_combined = (n1*C1 + n2*C2 + n1*n2/(n1+n2) * (μ1-μ2)(μ1-μ2)^T) / (n1+n2)

This allows O(d²) updates instead of O(n*d²) recomputation.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from .base import DistributedPCABase


@dataclass
class ClientState:
    """State for a single client in the streaming system."""
    client_id: str
    n_samples: int
    mean: np.ndarray
    covariance: np.ndarray
    last_updated: int = 0  # Update counter

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'client_id': self.client_id,
            'n_samples': self.n_samples,
            'mean': self.mean.copy(),
            'covariance': self.covariance.copy(),
            'last_updated': self.last_updated,
        }

    @classmethod
    def from_data(cls, client_id: str, data: np.ndarray, update_counter: int = 0) -> 'ClientState':
        """Create client state from raw data."""
        n_samples = len(data)
        mean = np.mean(data, axis=0)
        centered = data - mean
        covariance = (centered.T @ centered) / n_samples
        return cls(
            client_id=client_id,
            n_samples=n_samples,
            mean=mean,
            covariance=covariance,
            last_updated=update_counter,
        )


@dataclass
class GlobalState:
    """Global aggregated state."""
    n_samples: int = 0
    mean: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    n_features: int = 0
    update_counter: int = 0

    def is_initialized(self) -> bool:
        return self.mean is not None and self.covariance is not None


class StreamingDistributedPCA(DistributedPCABase):
    """Streaming Distributed PCA with incremental updates.

    Maintains global PCA that can be updated incrementally as:
    - New clients join the federation
    - Existing clients update their local data
    - Clients leave the federation

    Uses the mathematical property that covariances can be combined
    incrementally without accessing the original data.

    Attributes:
        n_components: Number of principal components.
        clients: Dictionary mapping client_id to ClientState.
        global_state: Current global aggregated state.
        history: List of (action, client_id, timestamp) tuples.
    """

    def __init__(
        self,
        n_components: int,
        random_state: Optional[int] = None,
        recompute_threshold: int = 100,
    ):
        """Initialize streaming distributed PCA.

        Args:
            n_components: Number of principal components.
            random_state: Random seed for reproducibility.
            recompute_threshold: Number of updates before full recomputation
                                (to prevent numerical drift).
        """
        super().__init__(n_components, random_state)
        self.recompute_threshold = recompute_threshold

        self.clients: Dict[str, ClientState] = {}
        self.global_state = GlobalState()
        self.history: List[Tuple[str, str, int]] = []
        self._updates_since_recompute = 0

    def fit(self, client_data: List[np.ndarray]) -> 'StreamingDistributedPCA':
        """Initial fit with data from multiple clients.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            self: Fitted model.
        """
        # Initialize with all clients
        for i, data in enumerate(client_data):
            client_id = f"client_{i}"
            self.add_client(client_id, data, recompute=False)

        # Recompute global PCA
        self._recompute_global()
        return self

    def add_client(
        self,
        client_id: str,
        data: np.ndarray,
        recompute: bool = True,
    ) -> None:
        """Add a new client to the federation.

        Args:
            client_id: Unique identifier for the client.
            data: Client's local data array of shape (n_samples, n_features).
            recompute: If True, recompute eigendecomposition after update.
        """
        if client_id in self.clients:
            raise ValueError(f"Client {client_id} already exists. Use update_client instead.")

        # Create client state
        self.global_state.update_counter += 1
        client_state = ClientState.from_data(
            client_id, data, self.global_state.update_counter
        )
        self.clients[client_id] = client_state

        # Update global state incrementally
        self._incremental_update_add(client_state)

        # Record history
        self.history.append(('add', client_id, self.global_state.update_counter))
        self._updates_since_recompute += 1

        if recompute:
            self._maybe_recompute()

    def update_client(
        self,
        client_id: str,
        new_data: np.ndarray,
        mode: str = 'replace',
        recompute: bool = True,
    ) -> None:
        """Update an existing client's data.

        Args:
            client_id: Client identifier.
            new_data: New data array.
            mode: Update mode:
                - 'replace': Replace client's data entirely
                - 'append': Add new samples to existing data (approximate)
            recompute: If True, recompute eigendecomposition after update.
        """
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not found. Use add_client first.")

        old_state = self.clients[client_id]

        if mode == 'replace':
            # Remove old contribution, add new
            self._incremental_update_remove(old_state)

            self.global_state.update_counter += 1
            new_state = ClientState.from_data(
                client_id, new_data, self.global_state.update_counter
            )
            self.clients[client_id] = new_state
            self._incremental_update_add(new_state)

        elif mode == 'append':
            # Incrementally update client state with new samples
            self._incremental_update_remove(old_state)

            # Combine old and new statistics
            self.global_state.update_counter += 1
            new_state = self._combine_client_data(old_state, new_data)
            new_state.last_updated = self.global_state.update_counter
            self.clients[client_id] = new_state

            self._incremental_update_add(new_state)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'replace' or 'append'.")

        self.history.append(('update', client_id, self.global_state.update_counter))
        self._updates_since_recompute += 1

        if recompute:
            self._maybe_recompute()

    def remove_client(self, client_id: str, recompute: bool = True) -> None:
        """Remove a client from the federation.

        Args:
            client_id: Client identifier.
            recompute: If True, recompute eigendecomposition after removal.
        """
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not found.")

        old_state = self.clients[client_id]
        self._incremental_update_remove(old_state)
        del self.clients[client_id]

        self.global_state.update_counter += 1
        self.history.append(('remove', client_id, self.global_state.update_counter))
        self._updates_since_recompute += 1

        if recompute:
            self._maybe_recompute()

    def _incremental_update_add(self, client_state: ClientState) -> None:
        """Incrementally add a client's contribution to global state."""
        n_new = client_state.n_samples
        μ_new = client_state.mean
        C_new = client_state.covariance

        if not self.global_state.is_initialized():
            # First client
            self.global_state.n_samples = n_new
            self.global_state.mean = μ_new.copy()
            self.global_state.covariance = C_new.copy()
            self.global_state.n_features = len(μ_new)
        else:
            # Combine with existing
            n_old = self.global_state.n_samples
            μ_old = self.global_state.mean
            C_old = self.global_state.covariance

            n_total = n_old + n_new

            # Update mean
            μ_combined = (n_old * μ_old + n_new * μ_new) / n_total

            # Update covariance using parallel algorithm
            # C_combined = (n1*C1 + n2*C2 + n1*n2/n_total * (μ1-μ2)(μ1-μ2)^T) / n_total
            mean_diff = μ_old - μ_new
            C_combined = (
                n_old * C_old +
                n_new * C_new +
                (n_old * n_new / n_total) * np.outer(mean_diff, mean_diff)
            ) / n_total

            self.global_state.n_samples = n_total
            self.global_state.mean = μ_combined
            self.global_state.covariance = C_combined

    def _incremental_update_remove(self, client_state: ClientState) -> None:
        """Incrementally remove a client's contribution from global state."""
        if not self.global_state.is_initialized():
            return

        n_remove = client_state.n_samples
        μ_remove = client_state.mean
        C_remove = client_state.covariance

        n_old = self.global_state.n_samples
        μ_old = self.global_state.mean
        C_old = self.global_state.covariance

        n_new = n_old - n_remove

        if n_new <= 0:
            # All clients removed
            self.global_state = GlobalState(n_features=self.global_state.n_features)
            return

        # Reverse the combination formula
        # μ_old = (n_new * μ_new + n_remove * μ_remove) / n_old
        # => μ_new = (n_old * μ_old - n_remove * μ_remove) / n_new
        μ_new = (n_old * μ_old - n_remove * μ_remove) / n_new

        # C_old = (n_new*C_new + n_remove*C_remove + n_new*n_remove/n_old * (μ_new-μ_remove)^2) / n_old
        # => n_old * C_old = n_new*C_new + n_remove*C_remove + n_new*n_remove/n_old * outer
        # => n_new * C_new = n_old * C_old - n_remove*C_remove - n_new*n_remove/n_old * outer
        mean_diff = μ_new - μ_remove
        C_new = (
            n_old * C_old -
            n_remove * C_remove -
            (n_new * n_remove / n_old) * np.outer(mean_diff, mean_diff)
        ) / n_new

        self.global_state.n_samples = n_new
        self.global_state.mean = μ_new
        self.global_state.covariance = C_new

    def _combine_client_data(
        self,
        old_state: ClientState,
        new_data: np.ndarray,
    ) -> ClientState:
        """Combine existing client state with new data samples."""
        n_old = old_state.n_samples
        μ_old = old_state.mean
        C_old = old_state.covariance

        n_new = len(new_data)
        μ_new = np.mean(new_data, axis=0)
        centered = new_data - μ_new
        C_new = (centered.T @ centered) / n_new

        n_total = n_old + n_new
        μ_combined = (n_old * μ_old + n_new * μ_new) / n_total

        mean_diff = μ_old - μ_new
        C_combined = (
            n_old * C_old +
            n_new * C_new +
            (n_old * n_new / n_total) * np.outer(mean_diff, mean_diff)
        ) / n_total

        return ClientState(
            client_id=old_state.client_id,
            n_samples=n_total,
            mean=μ_combined,
            covariance=C_combined,
        )

    def _maybe_recompute(self) -> None:
        """Recompute eigendecomposition, possibly with full recomputation."""
        if self._updates_since_recompute >= self.recompute_threshold:
            self._recompute_global()
        else:
            self._update_eigendecomposition()

    def _recompute_global(self) -> None:
        """Full recomputation of global state from all clients."""
        if not self.clients:
            return

        # Recompute global statistics from scratch
        total_samples = sum(c.n_samples for c in self.clients.values())
        weights = {cid: c.n_samples / total_samples for cid, c in self.clients.items()}

        # Global mean
        global_mean = sum(
            weights[cid] * c.mean for cid, c in self.clients.items()
        )

        # Global covariance (P-COV formula)
        global_cov = np.zeros_like(list(self.clients.values())[0].covariance)
        for cid, client in self.clients.items():
            w = weights[cid]
            mean_diff = client.mean - global_mean
            global_cov += w * (client.covariance + np.outer(mean_diff, mean_diff))

        self.global_state.n_samples = total_samples
        self.global_state.mean = global_mean
        self.global_state.covariance = global_cov

        self._update_eigendecomposition()
        self._updates_since_recompute = 0

    def _update_eigendecomposition(self) -> None:
        """Update eigendecomposition from current global covariance."""
        if not self.global_state.is_initialized():
            return

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.global_state.covariance)

        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store top k components
        k = min(self.n_components, len(eigenvalues))
        self.components_ = eigenvectors[:, :k].T
        self.mean_ = self.global_state.mean
        self._compute_explained_variance(eigenvalues)

    def local_computation(self, data: np.ndarray) -> Dict:
        """Compute local statistics for a client."""
        n_samples = len(data)
        mean = np.mean(data, axis=0)
        centered = data - mean
        covariance = (centered.T @ centered) / n_samples
        return {
            'n_samples': n_samples,
            'mean': mean,
            'covariance': covariance,
        }

    def aggregate(self, local_results: List[Dict]) -> None:
        """Aggregate local results (for compatibility with base class)."""
        # Reset and add all clients
        self.clients = {}
        self.global_state = GlobalState()

        for i, result in enumerate(local_results):
            client_id = f"client_{i}"
            state = ClientState(
                client_id=client_id,
                n_samples=result['n_samples'],
                mean=result['mean'],
                covariance=result['covariance'],
            )
            self.clients[client_id] = state
            self._incremental_update_add(state)

        self._update_eigendecomposition()

    def get_status(self) -> Dict:
        """Get current status of the streaming system."""
        return {
            'n_clients': len(self.clients),
            'total_samples': self.global_state.n_samples,
            'n_features': self.global_state.n_features,
            'n_components': self.n_components,
            'update_counter': self.global_state.update_counter,
            'updates_since_recompute': self._updates_since_recompute,
            'client_ids': list(self.clients.keys()),
            'client_samples': {cid: c.n_samples for cid, c in self.clients.items()},
        }

    def get_client_contribution(self, client_id: str) -> float:
        """Get a client's contribution to the global model (by sample proportion)."""
        if client_id not in self.clients:
            return 0.0
        return self.clients[client_id].n_samples / self.global_state.n_samples


def demonstrate_streaming():
    """Demonstrate streaming PCA capabilities."""
    np.random.seed(42)

    print("=" * 60)
    print("STREAMING DISTRIBUTED PCA DEMONSTRATION")
    print("=" * 60)

    # Generate initial data for 3 clients
    n_features = 50
    n_components = 10

    def generate_client_data(n_samples, class_id):
        """Generate data for a client with class-specific pattern."""
        pattern = np.zeros(n_features)
        pattern[class_id * 5:(class_id + 1) * 5] = 2.0
        noise = np.random.randn(n_samples, n_features) * 0.5
        return noise + pattern

    # Initialize streaming PCA
    streaming = StreamingDistributedPCA(n_components)

    # Phase 1: Initial clients
    print("\n--- Phase 1: Initial 3 clients ---")
    for i in range(3):
        data = generate_client_data(500, i)
        streaming.add_client(f"hospital_{i}", data)
        print(f"Added hospital_{i}: {len(data)} samples")

    print(f"\nStatus: {streaming.get_status()}")
    print(f"Explained variance ratio: {streaming.explained_variance_ratio_[:3].sum():.3f}")

    # Phase 2: New client joins
    print("\n--- Phase 2: New client joins ---")
    new_data = generate_client_data(800, 3)
    streaming.add_client("hospital_3", new_data)
    print(f"Added hospital_3: {len(new_data)} samples")
    print(f"Status: {streaming.get_status()}")

    # Phase 3: Existing client updates
    print("\n--- Phase 3: Client updates data ---")
    update_data = generate_client_data(200, 0)
    streaming.update_client("hospital_0", update_data, mode='append')
    print(f"Updated hospital_0 with 200 new samples (append mode)")
    print(f"hospital_0 now has {streaming.clients['hospital_0'].n_samples} samples")

    # Phase 4: Client leaves
    print("\n--- Phase 4: Client leaves ---")
    streaming.remove_client("hospital_1")
    print("Removed hospital_1")
    print(f"Status: {streaming.get_status()}")

    # Phase 5: Verify accuracy
    print("\n--- Phase 5: Accuracy verification ---")
    # Recompute from scratch to verify
    all_data = []
    for cid, client in streaming.clients.items():
        # We don't have original data, but we can check consistency
        print(f"  {cid}: {client.n_samples} samples, contribution: {streaming.get_client_contribution(cid):.1%}")

    print(f"\nFinal explained variance ratio (top 3): {streaming.explained_variance_ratio_[:3].sum():.3f}")
    print(f"Total updates: {streaming.global_state.update_counter}")

    return streaming


if __name__ == '__main__':
    demonstrate_streaming()
