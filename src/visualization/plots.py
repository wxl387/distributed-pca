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


def plot_projection_overlay(
    centralized_projection: np.ndarray,
    distributed_projection: np.ndarray,
    labels: np.ndarray,
    method_name: str = "Distributed",
    angle: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """Overlay scatter plot comparing centralized vs distributed PCA projections.

    Creates a scatter plot showing data points projected by both centralized
    (ground truth) and distributed PCA methods, allowing visual comparison
    of how well the distributed method approximates the true projection.

    Args:
        centralized_projection: (n_samples, 2) array - PC1 and PC2 from centralized PCA.
        distributed_projection: (n_samples, 2) array - PC1 and PC2 from distributed PCA.
        labels: (n_samples,) array - class labels for coloring points.
        method_name: Name of the distributed method for legend.
        angle: Optional subspace angle to display in title.
        title: Custom plot title. If None, auto-generated.
        save_path: If provided, save figure to this path.
        figsize: Figure size as (width, height).
    """
    plt.figure(figsize=figsize)

    # Get unique classes and create colormap
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_classes, 10)))

    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = colors[i % len(colors)]

        # Centralized: filled circles
        plt.scatter(
            centralized_projection[mask, 0],
            centralized_projection[mask, 1],
            c=[color],
            marker='o',
            s=50,
            alpha=0.6,
            label=f'Centralized (Class {label})',
            edgecolors='white',
            linewidths=0.5,
        )

        # Distributed: X markers
        plt.scatter(
            distributed_projection[mask, 0],
            distributed_projection[mask, 1],
            c=[color],
            marker='x',
            s=50,
            alpha=0.6,
            label=f'{method_name} (Class {label})',
            linewidths=1.5,
        )

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    # Generate title
    if title:
        plt.title(title)
    elif angle is not None:
        plt.title(f'Centralized vs {method_name} (mean angle: {angle:.2f}°)')
    else:
        plt.title(f'Centralized vs {method_name}')

    # Create simplified legend (just method markers, not all classes)
    handles = [
        plt.Line2D([0], [0], marker='o', color='gray', linestyle='', markersize=8,
                   label='Centralized', markerfacecolor='gray', alpha=0.6),
        plt.Line2D([0], [0], marker='x', color='gray', linestyle='', markersize=8,
                   label=method_name, markerfacecolor='gray', alpha=0.6),
    ]
    plt.legend(handles=handles, loc='upper right')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_multiple_methods_comparison(
    data: np.ndarray,
    labels: np.ndarray,
    centralized_pca,
    distributed_methods: Dict,
    angles: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10),
):
    """Grid of scatter plots comparing centralized vs multiple distributed methods.

    Creates a grid where each subplot shows the 2D projection comparison
    between centralized PCA and one distributed method.

    Args:
        data: (n_samples, n_features) array - original data.
        labels: (n_samples,) array - class labels for coloring.
        centralized_pca: Fitted centralized PCA model with transform method.
        distributed_methods: Dict[method_name, fitted_model] - distributed PCA models.
        angles: Optional Dict[method_name, angle] - subspace angles to display.
        save_path: If provided, save figure to this path.
        figsize: Figure size as (width, height).
    """
    n_methods = len(distributed_methods)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Get centralized projection
    centralized_proj = centralized_pca.transform(data)[:, :2]

    # Get unique classes and colormap
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_classes, 10)))

    for idx, (method_name, model) in enumerate(distributed_methods.items()):
        ax = axes[idx]

        # Get distributed projection
        distributed_proj = model.transform(data)[:, :2]

        # Plot each class
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = colors[i % len(colors)]

            # Centralized: circles
            ax.scatter(
                centralized_proj[mask, 0],
                centralized_proj[mask, 1],
                c=[color],
                marker='o',
                s=30,
                alpha=0.5,
                edgecolors='white',
                linewidths=0.3,
            )

            # Distributed: X markers
            ax.scatter(
                distributed_proj[mask, 0],
                distributed_proj[mask, 1],
                c=[color],
                marker='x',
                s=30,
                alpha=0.5,
                linewidths=1,
            )

        # Title with angle if available
        if angles and method_name in angles:
            ax.set_title(f'{method_name} (angle: {angles[method_name]:.2f}°)')
        else:
            ax.set_title(method_name)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    # Add overall legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='gray', linestyle='', markersize=8,
                   label='Centralized', markerfacecolor='gray', alpha=0.6),
        plt.Line2D([0], [0], marker='x', color='gray', linestyle='', markersize=8,
                   label='Distributed', markerfacecolor='gray', alpha=0.6),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Centralized vs Distributed PCA Projections', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
