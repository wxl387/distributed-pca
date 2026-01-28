"""Test adaptive distributed PCA on real MNIST dataset."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import load_mnist
from src.data.partitioners import DataPartitioner
from src.algorithms import (
    CentralizedPCA,
    PooledCovariancePCA,
    ApproximateCovPCA,
    ApproximateStackPCA,
    AdaptiveDistributedPCA,
)
from src.metrics.subspace_alignment import principal_angles, angle_to_degrees


def test_on_mnist():
    """Test adaptive method selection on real MNIST."""

    print("=" * 70)
    print("ADAPTIVE PCA TEST ON REAL MNIST")
    print("=" * 70)

    # Load MNIST
    print("\nLoading MNIST...")
    X_train, y_train = load_mnist(train=True, flatten=True, normalize=True)
    print(f"Loaded {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Configuration
    n_components = 50
    num_clients = 10
    seed = 42

    partition_configs = [
        ('IID', 'iid', {}),
        ('Dirichlet α=0.5', 'dirichlet', {'alpha': 0.5}),
        ('Dirichlet α=0.1', 'dirichlet', {'alpha': 0.1}),
        ('Shard-based', 'shard', {'shards_per_client': 2}),
    ]

    for partition_name, partition_type, partition_kwargs in partition_configs:
        print(f"\n{'=' * 70}")
        print(f"PARTITION: {partition_name}")
        print("=" * 70)

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
        print(f"Label-based EMD: {stats['mean_emd']:.3f}")

        # Fit centralized baseline
        centralized = CentralizedPCA(n_components, random_state=seed)
        centralized.fit(client_data)

        # Fit adaptive model
        adaptive = AdaptiveDistributedPCA(n_components, random_state=seed, verbose=True)
        adaptive.fit(client_data)

        print(f"\n{'Method':<15} {'Mean Angle':<12} {'Max Angle':<12}")
        print("-" * 40)

        # Compare methods
        methods = {
            'Centralized': centralized,
            'P-COV': PooledCovariancePCA(n_components, random_state=seed),
            'AP-COV': ApproximateCovPCA(n_components, random_state=seed),
            'AP-STACK': ApproximateStackPCA(n_components, random_state=seed),
            'Adaptive': adaptive,
        }

        for name, model in methods.items():
            if name != 'Centralized' and name != 'Adaptive':
                model.fit(client_data)

            angles = principal_angles(centralized.components_.T, model.components_.T)
            angles_deg = angle_to_degrees(angles)

            marker = f" ← SELECTED ({adaptive.selected_method_name_})" if name == 'Adaptive' else ""
            print(f"{name:<15} {np.mean(angles_deg):<12.4f} {np.max(angles_deg):<12.4f}{marker}")

    print("\n" + "=" * 70)
    print("ADAPTIVE METHOD SELECTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    test_on_mnist()
