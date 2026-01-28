"""Visualization utilities for distributed PCA experiments."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


def plot_explained_variance(
    results: Dict[str, np.ndarray],
    title: str = 'Explained Variance by Method',
    save_path: Optional[str] = None,
):
    """Plot explained variance ratio for multiple methods.

    Args:
        results: Dictionary mapping method names to explained variance arrays.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    plt.figure(figsize=(10, 6))

    for method_name, var_ratio in results.items():
        cumulative = np.cumsum(var_ratio)
        plt.plot(range(1, len(cumulative) + 1), cumulative, 'o-', label=method_name)

    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_subspace_angles(
    angles_dict: Dict[str, np.ndarray],
    title: str = 'Principal Angles vs Centralized PCA',
    save_path: Optional[str] = None,
):
    """Plot principal angles between distributed and centralized PCA.

    Args:
        angles_dict: Dictionary mapping method names to angle arrays (in degrees).
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    plt.figure(figsize=(10, 6))

    methods = list(angles_dict.keys())
    n_components = len(list(angles_dict.values())[0])

    x = np.arange(n_components)
    width = 0.8 / len(methods)

    for i, (method_name, angles) in enumerate(angles_dict.items()):
        plt.bar(x + i * width, angles, width, label=method_name, alpha=0.8)

    plt.xlabel('Component Index')
    plt.ylabel('Angle (degrees)')
    plt.title(title)
    plt.xticks(x + width * (len(methods) - 1) / 2, range(1, n_components + 1))
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_reconstruction_comparison(
    reconstruction_errors: Dict[str, float],
    title: str = 'Reconstruction Error Comparison',
    save_path: Optional[str] = None,
):
    """Plot reconstruction errors for multiple methods.

    Args:
        reconstruction_errors: Dictionary mapping method names to error values.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    plt.figure(figsize=(10, 6))

    methods = list(reconstruction_errors.keys())
    errors = list(reconstruction_errors.values())

    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = plt.bar(methods, errors, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, error in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{error:.4f}', ha='center', va='bottom')

    plt.xlabel('Method')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_method_comparison(
    results_df,
    metric: str = 'accuracy',
    groupby: str = 'partition',
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot method comparison across different conditions.

    Args:
        results_df: DataFrame with columns [method, partition, metric_values].
        metric: Metric to plot.
        groupby: Column to group by (e.g., 'partition', 'n_clients').
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    plt.figure(figsize=(12, 6))

    if hasattr(results_df, 'pivot'):
        # If pandas DataFrame
        pivot = results_df.pivot(index=groupby, columns='method', values=metric)
        pivot.plot(kind='bar', ax=plt.gca(), width=0.8)
    else:
        # If dict of dicts
        methods = list(results_df.keys())
        conditions = list(results_df[methods[0]].keys())
        x = np.arange(len(conditions))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            values = [results_df[method][cond] for cond in conditions]
            plt.bar(x + i * width, values, width, label=method)

        plt.xticks(x + width * (len(methods) - 1) / 2, conditions, rotation=45, ha='right')

    plt.xlabel(groupby.replace('_', ' ').title())
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title or f'{metric.replace("_", " ").title()} by Method')
    plt.legend(title='Method')
    plt.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_noniid_robustness(
    results: Dict[str, Dict[float, float]],
    metric_name: str = 'Subspace Alignment',
    save_path: Optional[str] = None,
):
    """Plot method performance vs non-IID severity.

    Args:
        results: Dict[method_name, Dict[alpha, metric_value]].
        metric_name: Name of the metric being plotted.
        save_path: If provided, save figure to this path.
    """
    plt.figure(figsize=(10, 6))

    for method_name, alpha_values in results.items():
        alphas = sorted(alpha_values.keys())
        values = [alpha_values[a] for a in alphas]
        plt.plot(alphas, values, 'o-', label=method_name, linewidth=2, markersize=8)

    plt.xlabel('Dirichlet Alpha (higher = more IID)')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Data Heterogeneity')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_scalability(
    results: Dict[str, Dict[int, float]],
    metric_name: str = 'Computation Time (s)',
    save_path: Optional[str] = None,
):
    """Plot method scalability vs number of clients.

    Args:
        results: Dict[method_name, Dict[n_clients, metric_value]].
        metric_name: Name of the metric being plotted.
        save_path: If provided, save figure to this path.
    """
    plt.figure(figsize=(10, 6))

    for method_name, client_values in results.items():
        n_clients = sorted(client_values.keys())
        values = [client_values[n] for n in n_clients]
        plt.plot(n_clients, values, 'o-', label=method_name, linewidth=2, markersize=8)

    plt.xlabel('Number of Clients')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Number of Clients')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_results_heatmap(
    results: Dict[str, Dict[str, float]],
    title: str = 'Method Performance Heatmap',
    save_path: Optional[str] = None,
):
    """Create heatmap of results across methods and conditions.

    Args:
        results: Dict[method, Dict[condition, value]].
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    methods = list(results.keys())
    conditions = list(results[methods[0]].keys())

    data = np.array([[results[m][c] for c in conditions] for m in methods])

    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=True, fmt='.3f', xticklabels=conditions,
                yticklabels=methods, cmap='RdYlGn')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
