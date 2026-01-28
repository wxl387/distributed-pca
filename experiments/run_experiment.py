"""Main experiment runner for distributed PCA comparison."""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import load_mnist, load_cifar10
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
from src.metrics.reconstruction import reconstruction_error, relative_reconstruction_error
from src.metrics.variance import explained_variance_ratio, cumulative_explained_variance
from src.metrics.downstream import evaluate_classification


# Available methods
METHODS = {
    'centralized': CentralizedPCA,
    'p_cov': PooledCovariancePCA,
    'sub_it': SubspaceIterationPCA,
    'ap_stack': ApproximateStackPCA,
    'qr_pca': QRPCA,
    'ap_cov': ApproximateCovPCA,
}


def load_data(dataset: str, data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test data for a dataset."""
    if dataset.lower() == 'mnist':
        X_train, y_train = load_mnist(data_dir, train=True)
        X_test, y_test = load_mnist(data_dir, train=False)
    elif dataset.lower() == 'cifar10':
        X_train, y_train = load_cifar10(data_dir, train=True)
        X_test, y_test = load_cifar10(data_dir, train=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return X_train, y_train, X_test, y_test


def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    partition_type: str,
    num_clients: int,
    seed: int = 42,
    **kwargs,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data across clients."""
    partitioner = DataPartitioner(num_clients, seed)

    if partition_type == 'iid':
        return partitioner.iid_partition(X, y)
    elif partition_type.startswith('dirichlet'):
        alpha = kwargs.get('alpha', 0.5)
        return partitioner.dirichlet_partition(X, y, alpha=alpha)
    elif partition_type == 'shard':
        shards_per_client = kwargs.get('shards_per_client', 2)
        return partitioner.shard_partition(X, y, shards_per_client=shards_per_client)
    elif partition_type == 'quantity_skew':
        power = kwargs.get('power', 1.5)
        return partitioner.quantity_skew_partition(X, y, power=power)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")


def evaluate_method(
    method_name: str,
    client_data: List[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train_pooled: np.ndarray,
    y_train_pooled: np.ndarray,
    centralized_model: CentralizedPCA,
    n_components: int,
    seed: int = 42,
) -> Dict:
    """Evaluate a single distributed PCA method."""
    results = {'method': method_name}

    # Create and fit model
    start_time = time.time()
    if method_name == 'centralized':
        model = METHODS[method_name](n_components, random_state=seed)
        model.fit(client_data)
    elif method_name == 'sub_it':
        model = METHODS[method_name](n_components, max_iter=100, tol=1e-6, random_state=seed)
        model.fit(client_data)
    else:
        model = METHODS[method_name](n_components, random_state=seed)
        model.fit(client_data)
    fit_time = time.time() - start_time
    results['fit_time'] = fit_time

    # Subspace alignment with centralized PCA
    angles = principal_angles(model.components_.T, centralized_model.components_.T)
    angles_deg = angle_to_degrees(angles)
    results['principal_angles'] = angles_deg
    results['mean_angle'] = np.mean(angles_deg)
    results['max_angle'] = np.max(angles_deg)
    results['alignment_score'] = alignment_score(model.components_.T, centralized_model.components_.T)

    # Reconstruction error
    results['reconstruction_mse'] = reconstruction_error(X_test, model, metric='mse')
    results['reconstruction_relative'] = relative_reconstruction_error(X_test, model)

    # Explained variance
    var_ratio = explained_variance_ratio(model, X_test)
    results['explained_variance'] = var_ratio
    results['total_explained_variance'] = np.sum(var_ratio)

    # Classification accuracy (using KNN)
    clf_results = evaluate_classification(
        X_train_pooled, y_train_pooled,
        X_test, y_test,
        model, classifier='knn', n_neighbors=5
    )
    results['classification_accuracy'] = clf_results['accuracy']
    results['classification_f1'] = clf_results['f1']

    # Communication cost
    results['communication'] = model.get_communication_cost()

    return results


def run_experiment(
    dataset: str = 'mnist',
    partition_type: str = 'iid',
    num_clients: int = 10,
    n_components: int = 50,
    alpha: float = 0.5,
    seed: int = 42,
    methods: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """Run a complete experiment comparing distributed PCA methods.

    Args:
        dataset: 'mnist' or 'cifar10'.
        partition_type: 'iid', 'dirichlet', 'shard', 'quantity_skew'.
        num_clients: Number of federated clients.
        n_components: Number of PCA components.
        alpha: Dirichlet concentration parameter (for non-IID).
        seed: Random seed.
        methods: List of method names to evaluate. If None, all methods.
        verbose: Print progress.

    Returns:
        Dictionary with experiment results.
    """
    if methods is None:
        methods = list(METHODS.keys())

    if verbose:
        print(f"Loading {dataset} dataset...")
    X_train, y_train, X_test, y_test = load_data(dataset)

    if verbose:
        print(f"Partitioning data across {num_clients} clients ({partition_type})...")
    partitions = partition_data(
        X_train, y_train, partition_type, num_clients, seed, alpha=alpha
    )
    client_data = [data for data, _ in partitions]
    X_train_pooled = np.vstack(client_data)
    y_train_pooled = np.concatenate([labels for _, labels in partitions])

    # Fit centralized baseline first
    if verbose:
        print("Fitting centralized PCA baseline...")
    centralized = CentralizedPCA(n_components, random_state=seed)
    centralized.fit(client_data)

    # Evaluate each method
    results = {
        'config': {
            'dataset': dataset,
            'partition_type': partition_type,
            'num_clients': num_clients,
            'n_components': n_components,
            'alpha': alpha,
            'seed': seed,
        },
        'methods': {},
    }

    for method_name in methods:
        if verbose:
            print(f"Evaluating {method_name}...")
        method_results = evaluate_method(
            method_name,
            client_data,
            X_test, y_test,
            X_train_pooled, y_train_pooled,
            centralized,
            n_components,
            seed,
        )
        results['methods'][method_name] = method_results

    return results


def print_results_summary(results: Dict):
    """Print a formatted summary of experiment results."""
    config = results['config']
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"Dataset: {config['dataset']}")
    print(f"Partition: {config['partition_type']}")
    if config['partition_type'] == 'dirichlet':
        print(f"Dirichlet alpha: {config['alpha']}")
    print(f"Clients: {config['num_clients']}")
    print(f"Components: {config['n_components']}")
    print("-" * 60)

    # Print table header
    print(f"{'Method':<15} {'Mean Angle':<12} {'Recon MSE':<12} {'Accuracy':<10} {'Time (s)':<10}")
    print("-" * 60)

    for method_name, method_results in results['methods'].items():
        print(f"{method_name:<15} "
              f"{method_results['mean_angle']:<12.4f} "
              f"{method_results['reconstruction_mse']:<12.6f} "
              f"{method_results['classification_accuracy']:<10.4f} "
              f"{method_results['fit_time']:<10.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run distributed PCA experiments')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--partition', type=str, default='iid',
                       choices=['iid', 'dirichlet', 'shard', 'quantity_skew'],
                       help='Data partitioning strategy')
    parser.add_argument('--num-clients', type=int, default=10,
                       help='Number of federated clients')
    parser.add_argument('--n-components', type=int, default=50,
                       help='Number of PCA components')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet concentration parameter')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                       help='Methods to evaluate')

    args = parser.parse_args()

    results = run_experiment(
        dataset=args.dataset,
        partition_type=args.partition,
        num_clients=args.num_clients,
        n_components=args.n_components,
        alpha=args.alpha,
        seed=args.seed,
        methods=args.methods,
        verbose=True,
    )

    print_results_summary(results)


if __name__ == '__main__':
    main()
