"""Demo experiment using synthetic data (no PyTorch required)."""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.partitioners import DataPartitioner
from src.algorithms import (
    CentralizedPCA,
    PooledCovariancePCA,
    SubspaceIterationPCA,
    ApproximateStackPCA,
    QRPCA,
    ApproximateCovPCA,
)
from src.metrics.subspace_alignment import principal_angles, alignment_score, angle_to_degrees
from src.metrics.reconstruction import reconstruction_error
from src.metrics.variance import explained_variance_ratio


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 100,
    n_classes: int = 10,
    seed: int = 42,
):
    """Generate synthetic classification data with clear PCA structure."""
    np.random.seed(seed)

    # Create class-specific means
    class_means = np.random.randn(n_classes, n_features) * 3

    # Generate samples
    samples_per_class = n_samples // n_classes
    data = []
    labels = []

    for c in range(n_classes):
        class_data = np.random.randn(samples_per_class, n_features) * 0.5 + class_means[c]
        data.append(class_data)
        labels.extend([c] * samples_per_class)

    data = np.vstack(data)
    labels = np.array(labels)

    # Shuffle
    idx = np.random.permutation(len(data))
    return data[idx], labels[idx]


def run_demo_experiment():
    """Run a demo experiment with synthetic data."""
    print("=" * 60)
    print("DISTRIBUTED PCA DEMO EXPERIMENT")
    print("=" * 60)

    # Configuration
    n_train = 5000
    n_test = 1000
    n_features = 100
    n_classes = 10
    n_components = 20
    num_clients = 10
    seed = 42

    print(f"\nConfiguration:")
    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {n_test}")
    print(f"  Features: {n_features}")
    print(f"  PCA components: {n_components}")
    print(f"  Number of clients: {num_clients}")

    # Generate data
    print("\nGenerating synthetic data...")
    X_train, y_train = generate_synthetic_data(n_train, n_features, n_classes, seed)
    X_test, y_test = generate_synthetic_data(n_test, n_features, n_classes, seed + 1)

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Test different partition strategies
    partition_configs = [
        ('IID', 'iid', {}),
        ('Dirichlet α=0.5', 'dirichlet', {'alpha': 0.5}),
        ('Dirichlet α=0.1', 'dirichlet', {'alpha': 0.1}),
        ('Shard-based', 'shard', {'shards_per_client': 2}),
    ]

    methods = {
        'Centralized': CentralizedPCA,
        'P-COV': PooledCovariancePCA,
        'SUB-IT': lambda k: SubspaceIterationPCA(k, max_iter=100, tol=1e-6, random_state=seed),
        'AP-STACK': ApproximateStackPCA,
        'QR-PCA': QRPCA,
        'AP-COV': ApproximateCovPCA,
    }

    for partition_name, partition_type, partition_kwargs in partition_configs:
        print(f"\n{'=' * 60}")
        print(f"Partition: {partition_name}")
        print("=" * 60)

        # Partition data
        partitioner = DataPartitioner(num_clients, seed)
        if partition_type == 'iid':
            partitions = partitioner.iid_partition(X_train, y_train)
        elif partition_type == 'dirichlet':
            partitions = partitioner.dirichlet_partition(X_train, y_train, **partition_kwargs)
        elif partition_type == 'shard':
            partitions = partitioner.shard_partition(X_train, y_train, **partition_kwargs)

        client_data = [data for data, _ in partitions]

        # Get partition stats
        stats = partitioner.get_partition_stats(partitions)
        print(f"  Samples per client: min={stats['min_samples']}, max={stats['max_samples']}, mean={stats['mean_samples']:.1f}")
        print(f"  Heterogeneity (EMD): {stats['mean_emd']:.3f}")

        # Fit centralized baseline
        centralized = CentralizedPCA(n_components, random_state=seed)
        centralized.fit(client_data)

        # Evaluate each method
        print(f"\n{'Method':<15} {'Mean Angle':<12} {'Max Angle':<12} {'Recon MSE':<14} {'Time (s)':<10}")
        print("-" * 70)

        for method_name, method_class in methods.items():
            start = time.time()

            if method_name == 'Centralized':
                model = centralized
            elif method_name == 'SUB-IT':
                model = SubspaceIterationPCA(n_components, max_iter=100, tol=1e-6, random_state=seed)
                model.fit(client_data)
            else:
                model = method_class(n_components, random_state=seed)
                model.fit(client_data)

            elapsed = time.time() - start

            # Compute metrics
            angles = principal_angles(centralized.components_.T, model.components_.T)
            angles_deg = angle_to_degrees(angles)
            recon_mse = reconstruction_error(X_test, model)

            print(f"{method_name:<15} {np.mean(angles_deg):<12.4f} {np.max(angles_deg):<12.4f} {recon_mse:<14.6f} {elapsed:<10.4f}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

    print("\n*** Key Observations ***")
    print("1. P-COV matches centralized PCA exactly (angles ≈ 0)")
    print("2. SUB-IT converges close to centralized solution")
    print("3. Approximate methods (AP-STACK, AP-COV) show larger angles")
    print("4. Non-IID partitioning increases angles for approximate methods")
    print("5. Exact methods remain robust regardless of data distribution")


if __name__ == '__main__':
    run_demo_experiment()
