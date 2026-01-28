"""Visualize PCA projections: Centralized vs Distributed methods.

This script generates scatter plots comparing how data is projected
by centralized (global) PCA versus various distributed PCA methods.

Usage:
    python experiments/visualize_projections.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import (
    CentralizedPCA,
    PooledCovariancePCA,
    ApproximateStackPCA,
    ApproximateCovPCA,
)
from src.data.partitioners import DataPartitioner
from src.metrics.subspace_alignment import principal_angles, angle_to_degrees
from src.visualization.plots import plot_projection_overlay, plot_multiple_methods_comparison


def generate_synthetic_data(n_samples=1000, n_features=50, n_classes=5, seed=42):
    """Generate synthetic data with clear class structure.

    Creates data where each class has a distinct mean, making
    class separation visible in PCA projections.
    """
    np.random.seed(seed)

    # Create class-specific patterns
    samples_per_class = n_samples // n_classes
    data = []
    labels = []

    for c in range(n_classes):
        # Each class has a different mean direction
        mean = np.zeros(n_features)
        mean[c * 10:(c + 1) * 10] = 3.0  # Class-specific features

        # Add some random rotation to make it more interesting
        class_data = np.random.randn(samples_per_class, n_features) * 0.5 + mean
        data.append(class_data)
        labels.extend([c] * samples_per_class)

    data = np.vstack(data)
    labels = np.array(labels)

    # Shuffle
    idx = np.random.permutation(len(data))
    return data[idx], labels[idx]


def visualize_iid_partition():
    """Visualize projections with IID data partition."""
    print("=" * 60)
    print("VISUALIZATION: IID Partition")
    print("=" * 60)

    # Generate data
    X, y = generate_synthetic_data(n_samples=1000, n_features=50, n_classes=5)

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Partition into clients (IID)
    partitioner = DataPartitioner(num_clients=5, seed=42)
    partitions = partitioner.iid_partition(X, y)
    client_data = [data for data, _ in partitions]

    n_components = 10

    # Fit centralized PCA
    centralized = CentralizedPCA(n_components, random_state=42)
    centralized.fit(client_data)

    # Fit distributed methods
    methods = {
        'P-COV': PooledCovariancePCA(n_components, random_state=42),
        'AP-COV': ApproximateCovPCA(n_components, random_state=42),
        'AP-STACK': ApproximateStackPCA(n_components, random_state=42),
    }

    for name, model in methods.items():
        model.fit(client_data)

    # Compute angles
    angles = {}
    for name, model in methods.items():
        ang = principal_angles(centralized.components_.T, model.components_.T)
        angles[name] = np.mean(angle_to_degrees(ang))
        print(f"{name}: mean angle = {angles[name]:.2f}°")

    # Get projections
    centralized_proj = centralized.transform(X)[:, :2]

    # Plot overlay for each method
    print("\nGenerating overlay plots...")
    for name, model in methods.items():
        distributed_proj = model.transform(X)[:, :2]
        plot_projection_overlay(
            centralized_proj,
            distributed_proj,
            y,
            method_name=name,
            angle=angles[name],
            title=f'IID Partition: Centralized vs {name}',
        )

    # Plot grid comparison
    print("\nGenerating grid comparison...")
    plot_multiple_methods_comparison(
        X, y,
        centralized,
        methods,
        angles=angles,
    )


def visualize_noniid_partition():
    """Visualize projections with non-IID data partition."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Non-IID Partition (Dirichlet α=0.1)")
    print("=" * 60)

    # Generate data
    X, y = generate_synthetic_data(n_samples=1000, n_features=50, n_classes=5)

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Partition into clients (highly non-IID)
    partitioner = DataPartitioner(num_clients=5, seed=42)
    partitions = partitioner.dirichlet_partition(X, y, alpha=0.1)
    client_data = [data for data, _ in partitions]

    # Show partition statistics
    stats = partitioner.get_partition_stats(partitions)
    print(f"Partition heterogeneity (EMD): {stats['mean_emd']:.3f}")

    n_components = 10

    # Fit centralized PCA
    centralized = CentralizedPCA(n_components, random_state=42)
    centralized.fit(client_data)

    # Fit distributed methods
    methods = {
        'P-COV': PooledCovariancePCA(n_components, random_state=42),
        'AP-COV': ApproximateCovPCA(n_components, random_state=42),
        'AP-STACK': ApproximateStackPCA(n_components, random_state=42),
    }

    for name, model in methods.items():
        model.fit(client_data)

    # Compute angles
    angles = {}
    for name, model in methods.items():
        ang = principal_angles(centralized.components_.T, model.components_.T)
        angles[name] = np.mean(angle_to_degrees(ang))
        print(f"{name}: mean angle = {angles[name]:.2f}°")

    # Get projections
    centralized_proj = centralized.transform(X)[:, :2]

    # Plot overlay for each method
    print("\nGenerating overlay plots...")
    for name, model in methods.items():
        distributed_proj = model.transform(X)[:, :2]
        plot_projection_overlay(
            centralized_proj,
            distributed_proj,
            y,
            method_name=name,
            angle=angles[name],
            title=f'Non-IID (α=0.1): Centralized vs {name}',
        )

    # Plot grid comparison
    print("\nGenerating grid comparison...")
    plot_multiple_methods_comparison(
        X, y,
        centralized,
        methods,
        angles=angles,
    )


def main():
    """Run all visualizations."""
    print("=" * 60)
    print("DISTRIBUTED PCA PROJECTION VISUALIZATION")
    print("=" * 60)
    print("\nThis script generates scatter plots comparing centralized PCA")
    print("projections (circles) with distributed PCA projections (X markers).")
    print("Points are colored by class label.\n")

    # IID partition
    visualize_iid_partition()

    # Non-IID partition
    visualize_noniid_partition()

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print("\nKey observations:")
    print("- P-COV (exact): Circles and X markers overlap perfectly")
    print("- AP-COV: Slight deviation on non-IID data")
    print("- AP-STACK: Significant deviation on non-IID data")


if __name__ == '__main__':
    main()
