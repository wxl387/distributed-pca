"""Adaptive Distributed PCA with automatic method selection.

This module implements adaptive method selection for distributed PCA based on
detected data heterogeneity across clients. The key insight from our experiments:

- P-COV: Always exact (0° angle) but O(d²) communication
- AP-COV: Nearly identical to P-COV on low-medium heterogeneity
- AP-STACK: Only suitable for IID data, degrades severely on non-IID

The adaptive approach:
1. Detect heterogeneity from client data statistics (without needing labels)
2. Select the most appropriate method based on heterogeneity level
3. Execute the selected method and return results
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from .base import DistributedPCABase
from .pooled_covariance import PooledCovariancePCA
from .approximate_cov import ApproximateCovPCA
from .approximate_stack import ApproximateStackPCA


class HeterogeneityDetector:
    """Detect data heterogeneity across federated clients.

    Uses statistics that can be computed locally and aggregated without
    sharing raw data. Provides multiple heterogeneity metrics.

    Metrics computed:
    - Mean divergence: How different are client means from global mean
    - Covariance divergence: How different are local covariance structures
    - Sample imbalance: How unequal are sample sizes across clients
    - Eigenspectrum divergence: How different are local eigenvalue distributions
    """

    def __init__(self, n_components: int = 20):
        """Initialize detector.

        Args:
            n_components: Number of components for eigenspectrum analysis.
        """
        self.n_components = n_components
        self.metrics_: Optional[Dict] = None
        self.heterogeneity_score_: Optional[float] = None

    def compute_client_statistics(self, data: np.ndarray) -> Dict:
        """Compute local statistics for a single client.

        Args:
            data: Local data array of shape (n_samples, n_features).

        Returns:
            Dictionary with local statistics.
        """
        n_samples = len(data)
        mean = np.mean(data, axis=0)

        # Centered data for covariance
        centered = data - mean
        cov = (centered.T @ centered) / n_samples

        # Top eigenvalues for spectrum comparison
        n_eig = min(self.n_components, min(data.shape) - 1)
        if n_eig > 0:
            try:
                eigenvalues = np.linalg.eigvalsh(cov)[::-1][:n_eig]
            except np.linalg.LinAlgError:
                eigenvalues = np.zeros(n_eig)
        else:
            eigenvalues = np.array([])

        # Trace of covariance (total variance)
        trace = np.trace(cov)

        return {
            'n_samples': n_samples,
            'mean': mean,
            'cov': cov,
            'eigenvalues': eigenvalues,
            'trace': trace,
        }

    def detect(self, client_data: List[np.ndarray]) -> Dict:
        """Detect heterogeneity across clients.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            Dictionary with heterogeneity metrics and overall score.
        """
        # Compute local statistics
        local_stats = [self.compute_client_statistics(data) for data in client_data]

        # Aggregate for global statistics
        total_samples = sum(s['n_samples'] for s in local_stats)
        weights = np.array([s['n_samples'] / total_samples for s in local_stats])

        # Global mean
        global_mean = sum(w * s['mean'] for w, s in zip(weights, local_stats))

        # Compute heterogeneity metrics
        metrics = {}

        # 1. Mean divergence: weighted average of distance from local to global mean
        mean_distances = []
        for stats in local_stats:
            dist = np.linalg.norm(stats['mean'] - global_mean)
            mean_distances.append(dist)
        metrics['mean_divergence'] = np.average(mean_distances, weights=weights)
        metrics['mean_divergence_max'] = np.max(mean_distances)

        # Normalize by average data standard deviation (more robust than global mean norm)
        # This handles centered data better
        avg_trace = np.average([s['trace'] for s in local_stats], weights=weights)
        n_features = len(global_mean)
        avg_std = np.sqrt(avg_trace / n_features + 1e-8)
        metrics['mean_divergence_normalized'] = metrics['mean_divergence'] / (avg_std * np.sqrt(n_features))

        # 2. Covariance divergence: Frobenius distance between local and average cov
        avg_cov = sum(w * s['cov'] for w, s in zip(weights, local_stats))
        cov_distances = []
        for stats in local_stats:
            dist = np.linalg.norm(stats['cov'] - avg_cov, 'fro')
            cov_distances.append(dist)
        metrics['cov_divergence'] = np.average(cov_distances, weights=weights)

        # Normalize by average covariance norm
        avg_cov_norm = np.linalg.norm(avg_cov, 'fro') + 1e-8
        metrics['cov_divergence_normalized'] = metrics['cov_divergence'] / avg_cov_norm

        # 3. Sample imbalance: coefficient of variation of sample sizes
        sample_sizes = np.array([s['n_samples'] for s in local_stats])
        metrics['sample_cv'] = np.std(sample_sizes) / np.mean(sample_sizes)
        metrics['sample_imbalance'] = np.max(sample_sizes) / (np.min(sample_sizes) + 1)

        # 4. Eigenspectrum divergence: compare eigenvalue distributions
        # Handle case where clients may have different numbers of eigenvalues
        min_eig_len = min(len(s['eigenvalues']) for s in local_stats)
        if min_eig_len > 0:
            # Normalize eigenvalues to sum to 1 (like a probability distribution)
            normalized_spectra = []
            for stats in local_stats:
                eigs = stats['eigenvalues'][:min_eig_len]  # Truncate to common length
                eigs_sum = np.sum(eigs) + 1e-8
                normalized_spectra.append(eigs / eigs_sum)

            # Stack into array (now all same length)
            spectra_array = np.array(normalized_spectra)
            avg_spectrum = np.mean(spectra_array, axis=0)

            # JS divergence from average
            spectrum_divs = []
            for spec in normalized_spectra:
                # KL divergence (with smoothing)
                eps = 1e-10
                spec_smooth = spec + eps
                avg_smooth = avg_spectrum + eps
                kl = np.sum(spec_smooth * np.log(spec_smooth / avg_smooth))
                spectrum_divs.append(kl)

            metrics['spectrum_divergence'] = np.mean(spectrum_divs)
        else:
            metrics['spectrum_divergence'] = 0.0

        # 5. Variance imbalance: compare total variances across clients
        traces = np.array([s['trace'] for s in local_stats])
        trace_divs = np.abs(traces - avg_trace) / (avg_trace + 1e-8)
        metrics['variance_divergence'] = np.average(trace_divs, weights=weights)

        # Compute overall heterogeneity score (0 = IID, 1 = highly heterogeneous)
        # Combine multiple metrics with empirical weights
        score = self._compute_heterogeneity_score(metrics)

        self.metrics_ = metrics
        self.heterogeneity_score_ = score

        return {
            'metrics': metrics,
            'heterogeneity_score': score,
            'level': self._score_to_level(score),
            'n_clients': len(client_data),
            'total_samples': total_samples,
        }

    def _compute_heterogeneity_score(self, metrics: Dict) -> float:
        """Compute overall heterogeneity score from individual metrics.

        Uses empirical calibration based on experiment results.
        Score ranges from 0 (IID) to 1 (highly heterogeneous).

        Calibration based on observed metrics:
        - IID: mean_div_norm ~0.01, cov_div_norm ~0.1
        - Dirichlet α=0.5: mean_div_norm ~0.1-0.3, cov_div_norm ~0.3-0.5
        - Dirichlet α=0.1: mean_div_norm ~0.5+, cov_div_norm ~0.5+
        """
        # Weights determined empirically based on correlation with method performance
        # Mean divergence is most predictive of AP-STACK degradation
        # Covariance divergence correlates with AP-COV degradation

        # Scale factors to normalize different metrics to ~[0, 1] range
        # Based on observed values in experiments
        mean_score = min(1.0, metrics['mean_divergence_normalized'] * 3)
        cov_score = min(1.0, metrics['cov_divergence_normalized'] * 1.5)
        spectrum_score = min(1.0, metrics['spectrum_divergence'] * 10)
        variance_score = min(1.0, metrics['variance_divergence'] * 5)

        # Weighted combination - covariance divergence is most reliable predictor
        score = (
            0.25 * mean_score +
            0.45 * cov_score +
            0.20 * spectrum_score +
            0.10 * variance_score
        )

        return min(1.0, score)

    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to categorical level."""
        if score < 0.15:
            return 'low'  # IID-like
        elif score < 0.35:
            return 'medium'  # Moderate heterogeneity
        else:
            return 'high'  # Severe heterogeneity


class AdaptiveDistributedPCA(DistributedPCABase):
    """Adaptive Distributed PCA that selects method based on data heterogeneity.

    Automatically detects the level of data heterogeneity across clients
    and selects the most appropriate distributed PCA method:

    - Low heterogeneity (IID-like): AP-COV (efficient, nearly exact)
    - Medium heterogeneity: AP-COV (robust to moderate heterogeneity)
    - High heterogeneity: P-COV (exact, but higher communication cost)

    Attributes:
        n_components: Number of principal components.
        selected_method_: The method class that was selected.
        selected_method_name_: Name of the selected method.
        heterogeneity_info_: Heterogeneity detection results.
        model_: The fitted PCA model.
    """

    # Method selection thresholds (calibrated from experiments)
    THRESHOLDS = {
        'use_pcov': 0.35,  # Above this, use P-COV (exact)
        'use_apcov': 0.15,  # Above this but below pcov threshold, use AP-COV
        # Below use_apcov, could use AP-STACK but we default to AP-COV for safety
    }

    def __init__(
        self,
        n_components: int,
        random_state: Optional[int] = None,
        force_method: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize adaptive distributed PCA.

        Args:
            n_components: Number of principal components to compute.
            random_state: Random seed for reproducibility.
            force_method: If specified, use this method instead of auto-selection.
                         Options: 'P-COV', 'AP-COV', 'AP-STACK'
            verbose: If True, print method selection details.
        """
        super().__init__(n_components, random_state)
        self.force_method = force_method
        self.verbose = verbose

        self.selected_method_: Optional[type] = None
        self.selected_method_name_: Optional[str] = None
        self.heterogeneity_info_: Optional[Dict] = None
        self.model_: Optional[DistributedPCABase] = None
        self.detector_ = HeterogeneityDetector(n_components=min(n_components, 50))

    def fit(self, client_data: List[np.ndarray]) -> 'AdaptiveDistributedPCA':
        """Fit distributed PCA with automatic method selection.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            self: Fitted model.
        """
        # Step 1: Detect heterogeneity
        self.heterogeneity_info_ = self.detector_.detect(client_data)

        # Step 2: Select method
        if self.force_method:
            method_name = self.force_method
        else:
            method_name = self._select_method(self.heterogeneity_info_['heterogeneity_score'])

        self.selected_method_name_ = method_name

        if self.verbose:
            print(f"Heterogeneity score: {self.heterogeneity_info_['heterogeneity_score']:.3f}")
            print(f"Heterogeneity level: {self.heterogeneity_info_['level']}")
            print(f"Selected method: {method_name}")

        # Step 3: Create and fit the selected model
        if method_name == 'P-COV':
            self.selected_method_ = PooledCovariancePCA
            self.model_ = PooledCovariancePCA(self.n_components, self.random_state)
        elif method_name == 'AP-COV':
            self.selected_method_ = ApproximateCovPCA
            self.model_ = ApproximateCovPCA(self.n_components, self.random_state)
        elif method_name == 'AP-STACK':
            self.selected_method_ = ApproximateStackPCA
            self.model_ = ApproximateStackPCA(self.n_components, self.random_state)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        self.model_.fit(client_data)

        # Copy fitted attributes
        self.components_ = self.model_.components_
        self.mean_ = self.model_.mean_
        self.explained_variance_ = self.model_.explained_variance_
        self.explained_variance_ratio_ = self.model_.explained_variance_ratio_

        return self

    def _select_method(self, score: float) -> str:
        """Select the best method based on heterogeneity score.

        Args:
            score: Heterogeneity score from 0 (IID) to 1 (highly heterogeneous).

        Returns:
            Name of the selected method.
        """
        if score >= self.THRESHOLDS['use_pcov']:
            # High heterogeneity: need exact method
            return 'P-COV'
        elif score >= self.THRESHOLDS['use_apcov']:
            # Medium heterogeneity: AP-COV is robust enough
            return 'AP-COV'
        else:
            # Low heterogeneity: AP-COV is safe choice (could use AP-STACK but risky)
            # We default to AP-COV for safety since it's nearly as fast as AP-STACK
            # and much more robust
            return 'AP-COV'

    def local_computation(self, data: np.ndarray) -> Dict:
        """Delegate to selected model's local computation."""
        if self.model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self.model_.local_computation(data)

    def aggregate(self, local_results: List[Dict]) -> None:
        """Delegate to selected model's aggregation."""
        if self.model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        self.model_.aggregate(local_results)

        # Update our attributes
        self.components_ = self.model_.components_
        self.mean_ = self.model_.mean_
        self.explained_variance_ = self.model_.explained_variance_
        self.explained_variance_ratio_ = self.model_.explained_variance_ratio_

    def get_communication_cost(self) -> Dict[str, int]:
        """Get communication cost of the selected method."""
        if self.model_ is None:
            return super().get_communication_cost()
        return self.model_.get_communication_cost()

    def get_selection_report(self) -> str:
        """Generate a human-readable report of the method selection.

        Returns:
            Formatted string with selection details.
        """
        if self.heterogeneity_info_ is None:
            return "No heterogeneity analysis performed yet."

        info = self.heterogeneity_info_
        metrics = info['metrics']

        report = [
            "=" * 60,
            "ADAPTIVE METHOD SELECTION REPORT",
            "=" * 60,
            f"",
            f"Data Overview:",
            f"  Clients: {info['n_clients']}",
            f"  Total samples: {info['total_samples']}",
            f"",
            f"Heterogeneity Metrics:",
            f"  Mean divergence (normalized): {metrics['mean_divergence_normalized']:.4f}",
            f"  Covariance divergence (normalized): {metrics['cov_divergence_normalized']:.4f}",
            f"  Spectrum divergence: {metrics['spectrum_divergence']:.4f}",
            f"  Variance divergence: {metrics['variance_divergence']:.4f}",
            f"  Sample imbalance ratio: {metrics['sample_imbalance']:.2f}",
            f"",
            f"Overall Assessment:",
            f"  Heterogeneity score: {info['heterogeneity_score']:.3f}",
            f"  Heterogeneity level: {info['level'].upper()}",
            f"",
            f"Method Selection:",
            f"  Selected: {self.selected_method_name_}",
            f"  Reason: {self._get_selection_reason()}",
            "=" * 60,
        ]

        return "\n".join(report)

    def _get_selection_reason(self) -> str:
        """Get human-readable reason for method selection."""
        if self.selected_method_name_ == 'P-COV':
            return "High heterogeneity detected; using exact method for accuracy"
        elif self.selected_method_name_ == 'AP-COV':
            return "Low-medium heterogeneity; AP-COV provides good accuracy-efficiency trade-off"
        elif self.selected_method_name_ == 'AP-STACK':
            return "Very low heterogeneity (IID-like); using efficient approximate method"
        else:
            return "Unknown"


def evaluate_adaptive_selection(
    client_data: List[np.ndarray],
    y_train: Optional[np.ndarray] = None,
    n_components: int = 20,
    random_state: int = 42,
) -> Dict:
    """Evaluate adaptive method selection against all methods.

    Runs the adaptive method and compares it to all other methods,
    measuring both accuracy (vs centralized) and efficiency.

    Args:
        client_data: List of data arrays, one per client.
        y_train: Optional labels for heterogeneity analysis (not used in selection).
        n_components: Number of PCA components.
        random_state: Random seed.

    Returns:
        Dictionary with comparison results.
    """
    from .centralized_pca import CentralizedPCA
    from ..metrics.subspace_alignment import principal_angles, angle_to_degrees

    # Fit centralized baseline
    centralized = CentralizedPCA(n_components, random_state=random_state)
    centralized.fit(client_data)

    # Fit all methods
    methods = {
        'Adaptive': AdaptiveDistributedPCA(n_components, random_state=random_state, verbose=True),
        'P-COV': PooledCovariancePCA(n_components, random_state=random_state),
        'AP-COV': ApproximateCovPCA(n_components, random_state=random_state),
        'AP-STACK': ApproximateStackPCA(n_components, random_state=random_state),
    }

    results = {}

    for name, model in methods.items():
        model.fit(client_data)

        # Compute angle to centralized
        angles = principal_angles(centralized.components_.T, model.components_.T)
        angles_deg = angle_to_degrees(angles)

        results[name] = {
            'mean_angle': float(np.mean(angles_deg)),
            'max_angle': float(np.max(angles_deg)),
        }

        if name == 'Adaptive':
            results[name]['selected_method'] = model.selected_method_name_
            results[name]['heterogeneity_score'] = model.heterogeneity_info_['heterogeneity_score']
            results[name]['heterogeneity_level'] = model.heterogeneity_info_['level']

    return results
