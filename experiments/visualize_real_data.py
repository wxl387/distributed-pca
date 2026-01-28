"""Visualize PCA projections on real datasets: MNIST and CIFAR-10.

This script generates scatter plots comparing how data is projected
by centralized (global) PCA versus various distributed PCA methods.

Usage:
    python experiments/visualize_real_data.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Output directory for saved plots
OUTPUT_DIR = Path(__file__).parent.parent / 'results' / 'visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from src.data.datasets import load_mnist, load_cifar10, get_dataset_info
from src.data.partitioners import DataPartitioner
from src.algorithms import (
    CentralizedPCA,
    PooledCovariancePCA,
    ApproximateStackPCA,
    ApproximateCovPCA,
)
from src.metrics.subspace_alignment import principal_angles, angle_to_degrees
from src.visualization.plots import plot_projection_overlay, plot_multiple_methods_comparison


def visualize_dataset(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    partition_type: str = 'iid',
    alpha: float = 0.1,
    n_clients: int = 5,
    n_components: int = 50,
    n_samples_viz: int = 2000,
    seed: int = 42,
):
    """Visualize PCA projections for a dataset.

    Args:
        X: Data array (n_samples, n_features)
        y: Labels array (n_samples,)
        dataset_name: Name of dataset for titles
        partition_type: 'iid' or 'noniid'
        alpha: Dirichlet alpha for non-IID (lower = more heterogeneous)
        n_clients: Number of clients for partitioning
        n_components: Number of PCA components
        n_samples_viz: Number of samples to visualize (subsample for clarity)
        seed: Random seed
    """
    np.random.seed(seed)

    # Subsample for visualization (too many points makes plot cluttered)
    if len(X) > n_samples_viz:
        idx = np.random.choice(len(X), n_samples_viz, replace=False)
        X_viz = X[idx]
        y_viz = y[idx]
    else:
        X_viz = X
        y_viz = y

    print(f"\n{'=' * 60}")
    print(f"{dataset_name} - {partition_type.upper()} Partition")
    print(f"{'=' * 60}")
    print(f"Data shape: {X.shape}, Visualizing: {len(X_viz)} samples")

    # Partition data
    partitioner = DataPartitioner(num_clients=n_clients, seed=seed)

    if partition_type == 'iid':
        partitions = partitioner.iid_partition(X, y)
    else:
        partitions = partitioner.dirichlet_partition(X, y, alpha=alpha)
        stats = partitioner.get_partition_stats(partitions)
        print(f"Partition heterogeneity (EMD): {stats['mean_emd']:.3f}")

    client_data = [data for data, _ in partitions]

    # Show class distribution per client
    print("\nClass distribution per client:")
    for i, (_, labels) in enumerate(partitions):
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  Client {i}: {dist}")

    # Fit centralized PCA
    print("\nFitting PCA models...")
    centralized = CentralizedPCA(n_components, random_state=seed)
    centralized.fit(client_data)

    # Fit distributed methods
    methods = {
        'P-COV': PooledCovariancePCA(n_components, random_state=seed),
        'AP-COV': ApproximateCovPCA(n_components, random_state=seed),
        'AP-STACK': ApproximateStackPCA(n_components, random_state=seed),
    }

    for name, model in methods.items():
        model.fit(client_data)

    # Compute angles
    angles = {}
    print("\nSubspace angles vs centralized:")
    for name, model in methods.items():
        ang = principal_angles(centralized.components_.T, model.components_.T)
        angles[name] = np.mean(angle_to_degrees(ang))
        print(f"  {name}: mean angle = {angles[name]:.2f}°")

    # Get projections for visualization subset
    centralized_proj = centralized.transform(X_viz)[:, :2]

    # Generate overlay plots for each method
    print("\nGenerating overlay plots...")
    for name, model in methods.items():
        distributed_proj = model.transform(X_viz)[:, :2]
        title = f'{dataset_name} ({partition_type.upper()}): Centralized vs {name}'
        save_name = f'{dataset_name.lower()}_{partition_type}_{name.lower().replace("-", "_")}_overlay.png'
        save_path = OUTPUT_DIR / save_name
        plot_projection_overlay(
            centralized_proj,
            distributed_proj,
            y_viz,
            method_name=name,
            angle=angles[name],
            title=title,
            save_path=str(save_path),
        )
        plt.close()
        print(f"  Saved: {save_path.name}")

    # Generate grid comparison
    print("Generating grid comparison...")

    # Need to create a wrapper that transforms the viz subset
    class VizWrapper:
        def __init__(self, model, X_viz):
            self.model = model
            self.X_viz = X_viz
            self._proj = model.transform(X_viz)

        def transform(self, X):
            # Return cached projection for viz data
            return self._proj

    methods_viz = {name: VizWrapper(model, X_viz) for name, model in methods.items()}
    centralized_viz = VizWrapper(centralized, X_viz)

    grid_save_path = OUTPUT_DIR / f'{dataset_name.lower()}_{partition_type}_grid_comparison.png'
    plot_multiple_methods_comparison(
        X_viz, y_viz,
        centralized_viz,
        methods_viz,
        angles=angles,
        save_path=str(grid_save_path),
    )
    plt.close()
    print(f"  Saved: {grid_save_path.name}")

    return angles


def main():
    """Run visualizations on MNIST and CIFAR-10."""
    print("=" * 60)
    print("DISTRIBUTED PCA VISUALIZATION ON REAL DATASETS")
    print("=" * 60)

    # Get dataset info
    mnist_info = get_dataset_info('mnist')
    cifar_info = get_dataset_info('cifar10')

    print(f"\nMNIST classes: digits 0-9")
    print(f"CIFAR-10 classes: {cifar_info['class_names']}")

    results = {}

    # =========================================
    # MNIST
    # =========================================
    print("\n" + "=" * 60)
    print("Loading MNIST...")
    print("=" * 60)

    X_mnist, y_mnist = load_mnist(train=True, flatten=True, normalize=True)
    print(f"MNIST shape: {X_mnist.shape}")

    # MNIST IID
    results['mnist_iid'] = visualize_dataset(
        X_mnist, y_mnist,
        dataset_name='MNIST',
        partition_type='iid',
        n_clients=5,
        n_components=50,
        n_samples_viz=2000,
    )

    # MNIST Non-IID
    results['mnist_noniid'] = visualize_dataset(
        X_mnist, y_mnist,
        dataset_name='MNIST',
        partition_type='noniid',
        alpha=0.1,
        n_clients=5,
        n_components=50,
        n_samples_viz=2000,
    )

    # =========================================
    # CIFAR-10
    # =========================================
    print("\n" + "=" * 60)
    print("Loading CIFAR-10...")
    print("=" * 60)

    X_cifar, y_cifar = load_cifar10(train=True, flatten=True, normalize=True)
    print(f"CIFAR-10 shape: {X_cifar.shape}")

    # CIFAR-10 IID
    results['cifar_iid'] = visualize_dataset(
        X_cifar, y_cifar,
        dataset_name='CIFAR-10',
        partition_type='iid',
        n_clients=5,
        n_components=50,
        n_samples_viz=2000,
    )

    # CIFAR-10 Non-IID
    results['cifar_noniid'] = visualize_dataset(
        X_cifar, y_cifar,
        dataset_name='CIFAR-10',
        partition_type='noniid',
        alpha=0.1,
        n_clients=5,
        n_components=50,
        n_samples_viz=2000,
    )

    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("SUMMARY: Mean Subspace Angles (degrees)")
    print("=" * 60)
    print(f"\n{'Dataset':<20} {'P-COV':<10} {'AP-COV':<10} {'AP-STACK':<10}")
    print("-" * 50)
    for key, angles in results.items():
        print(f"{key:<20} {angles['P-COV']:<10.2f} {angles['AP-COV']:<10.2f} {angles['AP-STACK']:<10.2f}")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print("\nHow to interpret:")
    print("- Circles (○) = Centralized PCA projection (ground truth)")
    print("- X markers (×) = Distributed PCA projection")
    print("- Colors = Class labels (0-9 for both datasets)")
    print("\nClass meanings:")
    print("  MNIST: digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
    print(f"  CIFAR-10: {', '.join(cifar_info['class_names'])}")
    print("\nKey observations:")
    print("- P-COV (exact method): Perfect overlap expected")
    print("- AP-COV: Small deviation, especially on non-IID data")
    print("- AP-STACK: Larger deviation on non-IID data")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print(f"Files created: {len(list(OUTPUT_DIR.glob('*.png')))} PNG files")


if __name__ == '__main__':
    main()
