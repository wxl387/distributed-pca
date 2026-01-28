"""Unit tests for distributed PCA algorithms."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import (
    CentralizedPCA,
    PooledCovariancePCA,
    SubspaceIterationPCA,
    ApproximateStackPCA,
    QRPCA,
    ApproximateCovPCA,
)
from src.metrics.subspace_alignment import principal_angles, alignment_score


def generate_test_data(n_clients=5, n_samples_per_client=100, n_features=50, seed=42):
    """Generate synthetic test data for multiple clients."""
    np.random.seed(seed)
    client_data = [
        np.random.randn(n_samples_per_client, n_features)
        for _ in range(n_clients)
    ]
    return client_data


class TestCentralizedPCA:
    """Tests for centralized PCA baseline."""

    def test_fit_and_transform(self):
        """Test basic fit and transform."""
        data = generate_test_data(n_clients=1)[0]
        pca = CentralizedPCA(n_components=10)
        pca.fit([data])

        assert pca.components_.shape == (10, 50)
        assert pca.mean_.shape == (50,)

        transformed = pca.transform(data)
        assert transformed.shape == (100, 10)

    def test_components_orthonormal(self):
        """Test that components are orthonormal."""
        data = generate_test_data(n_clients=1)[0]
        pca = CentralizedPCA(n_components=10)
        pca.fit([data])

        # Check orthonormality
        gram = pca.components_ @ pca.components_.T
        assert np.allclose(gram, np.eye(10), atol=1e-10)


class TestPooledCovariance:
    """Tests for P-COV method."""

    def test_matches_centralized_iid(self):
        """P-COV should match centralized PCA on IID data."""
        client_data = generate_test_data(n_clients=5, n_samples_per_client=100)
        n_components = 10

        centralized = CentralizedPCA(n_components)
        centralized.fit(client_data)

        pcov = PooledCovariancePCA(n_components)
        pcov.fit(client_data)

        # Check subspace alignment
        angles = principal_angles(centralized.components_.T, pcov.components_.T)
        assert np.allclose(angles, 0, atol=1e-10), f"Angles should be 0, got {angles}"

    def test_single_client(self):
        """With one client, should match local PCA."""
        client_data = generate_test_data(n_clients=1)
        n_components = 10

        centralized = CentralizedPCA(n_components)
        centralized.fit(client_data)

        pcov = PooledCovariancePCA(n_components)
        pcov.fit(client_data)

        angles = principal_angles(centralized.components_.T, pcov.components_.T)
        assert np.allclose(angles, 0, atol=1e-10)


class TestSubspaceIteration:
    """Tests for SUB-IT method."""

    def test_converges_to_centralized(self):
        """SUB-IT should converge to centralized PCA solution."""
        client_data = generate_test_data(n_clients=5, n_samples_per_client=100)
        n_components = 10

        centralized = CentralizedPCA(n_components)
        centralized.fit(client_data)

        subit = SubspaceIterationPCA(n_components, max_iter=100, tol=1e-8, random_state=42)
        subit.fit(client_data)

        # Check alignment (allowing small numerical errors)
        score = alignment_score(centralized.components_.T, subit.components_.T)
        assert score > 0.99, f"Alignment score should be >0.99, got {score}"


class TestApproximateMethods:
    """Tests for approximate methods."""

    def test_apstack_reasonable_approximation(self):
        """AP-STACK should give reasonable approximation on IID data."""
        client_data = generate_test_data(n_clients=5, n_samples_per_client=200)
        n_components = 10

        centralized = CentralizedPCA(n_components)
        centralized.fit(client_data)

        apstack = ApproximateStackPCA(n_components)
        apstack.fit(client_data)

        score = alignment_score(centralized.components_.T, apstack.components_.T)
        assert score > 0.8, f"AP-STACK alignment should be >0.8 on IID, got {score}"

    def test_qrpca_accurate(self):
        """QR-PCA should be accurate."""
        client_data = generate_test_data(n_clients=5, n_samples_per_client=200)
        n_components = 10

        centralized = CentralizedPCA(n_components)
        centralized.fit(client_data)

        qrpca = QRPCA(n_components)
        qrpca.fit(client_data)

        score = alignment_score(centralized.components_.T, qrpca.components_.T)
        assert score > 0.95, f"QR-PCA alignment should be >0.95, got {score}"

    def test_apcov_degrades_on_heterogeneous(self):
        """AP-COV should work on IID but may degrade on heterogeneous data."""
        np.random.seed(42)
        # Create heterogeneous data (different means per client)
        client_data = [
            np.random.randn(100, 50) + i * 2  # Different mean shifts
            for i in range(5)
        ]
        n_components = 10

        centralized = CentralizedPCA(n_components)
        centralized.fit(client_data)

        apcov = ApproximateCovPCA(n_components)
        apcov.fit(client_data)

        # Should still give some alignment but may not be perfect
        score = alignment_score(centralized.components_.T, apcov.components_.T)
        assert score > 0.5, f"AP-COV alignment should be >0.5, got {score}"


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_principal_angles_identical(self):
        """Identical subspaces should have zero angles."""
        U = np.eye(10, 5)
        angles = principal_angles(U, U)
        assert np.allclose(angles, 0)

    def test_principal_angles_orthogonal(self):
        """Orthogonal subspaces should have pi/2 angles."""
        U = np.eye(10, 2)[:, :2]
        V = np.eye(10, 2)[:, 2:4]
        V = np.hstack([V, np.zeros((10, 0))])[:, :2]
        # Create orthogonal basis
        V = np.zeros((10, 2))
        V[2, 0] = 1
        V[3, 1] = 1
        angles = principal_angles(U, V)
        assert np.allclose(angles, np.pi / 2, atol=1e-10)

    def test_alignment_score_range(self):
        """Alignment score should be in [0, 1]."""
        U = np.random.randn(10, 3)
        U, _ = np.linalg.qr(U)
        V = np.random.randn(10, 3)
        V, _ = np.linalg.qr(V)

        score = alignment_score(U, V)
        assert 0 <= score <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
