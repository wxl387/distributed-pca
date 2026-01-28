"""Base class for distributed PCA algorithms."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np


class DistributedPCABase(ABC):
    """Abstract base class for distributed PCA algorithms.

    All distributed PCA methods should inherit from this class and implement
    the abstract methods for local computation and aggregation.

    Attributes:
        n_components: Number of principal components to compute.
        components_: Fitted principal components (n_components x n_features).
        mean_: Global mean vector (n_features,).
        explained_variance_: Variance explained by each component.
        explained_variance_ratio_: Ratio of variance explained by each component.
    """

    def __init__(self, n_components: int, random_state: Optional[int] = None):
        """Initialize distributed PCA.

        Args:
            n_components: Number of principal components to compute.
            random_state: Random seed for reproducibility.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self._total_variance: Optional[float] = None

    @abstractmethod
    def fit(self, client_data: List[np.ndarray]) -> 'DistributedPCABase':
        """Fit distributed PCA on data from multiple clients.

        Args:
            client_data: List of data arrays, one per client.
                        Each array has shape (n_samples_k, n_features).

        Returns:
            self: Fitted model.
        """
        pass

    @abstractmethod
    def local_computation(self, data: np.ndarray) -> Dict:
        """Perform local computation at a single client.

        This method computes local statistics or components that will be
        sent to the server for aggregation.

        Args:
            data: Local data array of shape (n_samples, n_features).

        Returns:
            Dictionary containing local results to send to server.
        """
        pass

    @abstractmethod
    def aggregate(self, local_results: List[Dict]) -> None:
        """Aggregate local results from all clients at the server.

        This method combines the local computations to produce global
        principal components.

        Args:
            local_results: List of dictionaries from local_computation.
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto principal components.

        Args:
            X: Data array of shape (n_samples, n_features).

        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct data from principal components.

        Args:
            X_transformed: Transformed data of shape (n_samples, n_components).

        Returns:
            Reconstructed data of shape (n_samples, n_features).
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return X_transformed @ self.components_ + self.mean_

    def fit_transform(self, client_data: List[np.ndarray]) -> List[np.ndarray]:
        """Fit and transform data from multiple clients.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            List of transformed data arrays, one per client.
        """
        self.fit(client_data)
        return [self.transform(data) for data in client_data]

    def get_communication_cost(self) -> Dict[str, int]:
        """Get communication cost metrics.

        Returns:
            Dictionary with:
                - 'rounds': Number of communication rounds
                - 'bytes_per_client': Bytes sent per client per round
                - 'total_bytes': Total bytes communicated
        """
        return {
            'rounds': 0,
            'bytes_per_client': 0,
            'total_bytes': 0,
        }

    def _compute_explained_variance(self, eigenvalues: np.ndarray) -> None:
        """Compute explained variance from eigenvalues.

        Args:
            eigenvalues: Array of eigenvalues (sorted descending).
        """
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self._total_variance = total_var
        if total_var > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
