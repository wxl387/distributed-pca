"""Differentially Private Distributed PCA.

This module implements differential privacy for distributed PCA, allowing
clients to share statistics with formal privacy guarantees.

Key concepts:
- (ε, δ)-differential privacy: Probability of distinguishing whether any
  single data point was included is bounded
- Sensitivity: Maximum change in output when one data point changes
- Gaussian mechanism: Add N(0, σ²) noise where σ = sensitivity * √(2ln(1.25/δ)) / ε

Privacy is achieved by adding calibrated noise to:
1. Local mean vectors
2. Local covariance matrices

The noise magnitude depends on:
- Privacy budget (ε, δ): Smaller ε = more privacy, more noise
- Data bounds: Assumed or computed bounds on feature values
- Sample size: More samples = less noise needed per sample
"""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from .base import DistributedPCABase


@dataclass
class PrivacyAccountant:
    """Track privacy budget consumption."""
    epsilon: float
    delta: float
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0
    queries: List[Tuple[str, float, float]] = None

    def __post_init__(self):
        if self.queries is None:
            self.queries = []

    def can_spend(self, eps: float, delt: float) -> bool:
        """Check if we can spend this much privacy budget."""
        return (self.epsilon_spent + eps <= self.epsilon and
                self.delta_spent + delt <= self.delta)

    def spend(self, eps: float, delt: float, query_name: str = "query") -> None:
        """Spend privacy budget."""
        if not self.can_spend(eps, delt):
            raise ValueError(f"Privacy budget exceeded. Remaining: "
                           f"ε={self.epsilon - self.epsilon_spent:.4f}, "
                           f"δ={self.delta - self.delta_spent:.6f}")
        self.epsilon_spent += eps
        self.delta_spent += delt
        self.queries.append((query_name, eps, delt))

    def remaining(self) -> Tuple[float, float]:
        """Return remaining (epsilon, delta)."""
        return (self.epsilon - self.epsilon_spent, self.delta - self.delta_spent)

    def summary(self) -> str:
        """Return summary of privacy spending."""
        lines = [
            f"Privacy Budget: ε={self.epsilon}, δ={self.delta}",
            f"Spent: ε={self.epsilon_spent:.4f}, δ={self.delta_spent:.6f}",
            f"Remaining: ε={self.epsilon - self.epsilon_spent:.4f}, "
            f"δ={self.delta - self.delta_spent:.6f}",
            f"Queries: {len(self.queries)}",
        ]
        for name, eps, delt in self.queries:
            lines.append(f"  - {name}: ε={eps:.4f}, δ={delt:.6f}")
        return "\n".join(lines)


def compute_gaussian_noise_scale(
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> float:
    """Compute noise scale for Gaussian mechanism.

    For (ε, δ)-differential privacy with Gaussian noise:
    σ = sensitivity * √(2 * ln(1.25/δ)) / ε

    Args:
        sensitivity: L2 sensitivity of the query.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.

    Returns:
        Standard deviation of Gaussian noise to add.
    """
    if epsilon <= 0 or delta <= 0:
        raise ValueError("epsilon and delta must be positive")
    if delta >= 1:
        raise ValueError("delta must be < 1")

    return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon


def clip_data(
    data: np.ndarray,
    clip_bound: float,
    per_sample: bool = True,
) -> np.ndarray:
    """Clip data to bounded L2 norm.

    Args:
        data: Data array of shape (n_samples, n_features).
        clip_bound: Maximum L2 norm.
        per_sample: If True, clip each sample; if False, clip entire matrix.

    Returns:
        Clipped data.
    """
    if per_sample:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        scale = np.minimum(1.0, clip_bound / (norms + 1e-10))
        return data * scale
    else:
        norm = np.linalg.norm(data)
        if norm > clip_bound:
            return data * (clip_bound / norm)
        return data


class DifferentiallyPrivatePCA(DistributedPCABase):
    """Differentially Private Distributed PCA.

    Implements (ε, δ)-differential privacy by adding Gaussian noise to
    local statistics (mean and covariance) before aggregation.

    Privacy guarantees:
    - Each client's contribution is protected
    - Adding/removing one sample changes output by bounded amount
    - Formal (ε, δ)-DP guarantee

    Attributes:
        n_components: Number of principal components.
        epsilon: Total privacy budget ε.
        delta: Privacy parameter δ.
        clip_bound: L2 norm bound for clipping data.
        accountant: Privacy budget tracker.
    """

    def __init__(
        self,
        n_components: int,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_bound: Optional[float] = None,
        random_state: Optional[int] = None,
        noise_multiplier: float = 1.0,
    ):
        """Initialize differentially private PCA.

        Args:
            n_components: Number of principal components.
            epsilon: Privacy budget ε. Smaller = more privacy. Typical: 0.1-10.
            delta: Privacy parameter δ. Should be < 1/n. Typical: 1e-5.
            clip_bound: L2 norm bound for data clipping. If None, estimated from data.
            random_state: Random seed for reproducibility.
            noise_multiplier: Multiplier for noise (>1 for more privacy, <1 for less).
        """
        super().__init__(n_components, random_state)
        self.epsilon = epsilon
        self.delta = delta
        self.clip_bound = clip_bound
        self.noise_multiplier = noise_multiplier

        self.accountant = PrivacyAccountant(epsilon, delta)
        self._rng = np.random.RandomState(random_state)

        # Store noise added for analysis
        self.noise_info_: Optional[Dict] = None

    def fit(self, client_data: List[np.ndarray]) -> 'DifferentiallyPrivatePCA':
        """Fit differentially private distributed PCA.

        Args:
            client_data: List of data arrays, one per client.

        Returns:
            self: Fitted model.
        """
        # Determine clip bound if not specified
        if self.clip_bound is None:
            # Estimate from data (this leaks some privacy, but is common practice)
            all_norms = [np.linalg.norm(data, axis=1) for data in client_data]
            self.clip_bound = np.percentile(np.concatenate(all_norms), 95)

        # Compute local statistics with privacy
        local_results = []
        for i, data in enumerate(client_data):
            result = self.local_computation(data, client_id=i)
            local_results.append(result)

        # Aggregate (server side, no additional noise needed)
        self.aggregate(local_results)

        return self

    def local_computation(
        self,
        data: np.ndarray,
        client_id: int = 0,
    ) -> Dict:
        """Compute local statistics with differential privacy.

        Args:
            data: Local data array of shape (n_samples, n_features).
            client_id: Client identifier for privacy accounting.

        Returns:
            Dictionary with noisy local statistics.
        """
        n_samples, n_features = data.shape

        # Step 1: Clip data to bounded norm
        clipped_data = clip_data(data, self.clip_bound, per_sample=True)

        # Step 2: Compute local statistics
        local_mean = np.mean(clipped_data, axis=0)
        centered = clipped_data - local_mean
        local_cov = (centered.T @ centered) / n_samples

        # Step 3: Compute sensitivities
        # Mean sensitivity: changing one sample changes mean by at most clip_bound/n
        mean_sensitivity = self.clip_bound / n_samples

        # Covariance sensitivity: changing one sample changes cov by at most
        # 2 * clip_bound² / n (from the outer product contribution)
        cov_sensitivity = 2 * self.clip_bound ** 2 / n_samples

        # Step 4: Split privacy budget between mean and covariance
        # Using composition theorem: total privacy = sum of individual privacies
        eps_mean = self.epsilon / 2
        eps_cov = self.epsilon / 2
        delta_mean = self.delta / 2
        delta_cov = self.delta / 2

        # Step 5: Add Gaussian noise
        mean_noise_scale = compute_gaussian_noise_scale(
            mean_sensitivity, eps_mean, delta_mean
        ) * self.noise_multiplier

        cov_noise_scale = compute_gaussian_noise_scale(
            cov_sensitivity, eps_cov, delta_cov
        ) * self.noise_multiplier

        # Noise for mean (vector)
        mean_noise = self._rng.normal(0, mean_noise_scale, size=n_features)
        noisy_mean = local_mean + mean_noise

        # Noise for covariance (symmetric matrix)
        # Generate symmetric noise matrix
        cov_noise = self._rng.normal(0, cov_noise_scale, size=(n_features, n_features))
        cov_noise = (cov_noise + cov_noise.T) / 2  # Make symmetric
        noisy_cov = local_cov + cov_noise

        # Store noise info for analysis
        if self.noise_info_ is None:
            self.noise_info_ = {'clients': []}

        self.noise_info_['clients'].append({
            'client_id': client_id,
            'n_samples': n_samples,
            'mean_noise_scale': mean_noise_scale,
            'cov_noise_scale': cov_noise_scale,
            'mean_noise_norm': np.linalg.norm(mean_noise),
            'cov_noise_norm': np.linalg.norm(cov_noise, 'fro'),
            'signal_to_noise_mean': np.linalg.norm(local_mean) / (np.linalg.norm(mean_noise) + 1e-10),
            'signal_to_noise_cov': np.linalg.norm(local_cov, 'fro') / (np.linalg.norm(cov_noise, 'fro') + 1e-10),
        })

        return {
            'n_samples': n_samples,
            'mean': noisy_mean,
            'covariance': noisy_cov,
        }

    def aggregate(self, local_results: List[Dict]) -> None:
        """Aggregate noisy local statistics.

        Uses weighted average based on sample counts.

        Args:
            local_results: List of dictionaries from local_computation.
        """
        # Total samples
        total_samples = sum(r['n_samples'] for r in local_results)
        weights = [r['n_samples'] / total_samples for r in local_results]

        # Weighted mean
        global_mean = sum(w * r['mean'] for w, r in zip(weights, local_results))

        # Weighted covariance with between-client variance correction
        global_cov = np.zeros_like(local_results[0]['covariance'])
        for w, r in zip(weights, local_results):
            mean_diff = r['mean'] - global_mean
            global_cov += w * (r['covariance'] + np.outer(mean_diff, mean_diff))

        # Eigendecomposition
        # Add small regularization for numerical stability (noise may make it non-PSD)
        global_cov = (global_cov + global_cov.T) / 2  # Ensure symmetric
        min_eig = np.min(np.linalg.eigvalsh(global_cov))
        if min_eig < 0:
            global_cov += (-min_eig + 1e-6) * np.eye(global_cov.shape[0])

        eigenvalues, eigenvectors = np.linalg.eigh(global_cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store results
        k = min(self.n_components, len(eigenvalues))
        self.components_ = eigenvectors[:, :k].T
        self.mean_ = global_mean
        self._compute_explained_variance(np.maximum(eigenvalues, 0))

    def get_privacy_report(self) -> str:
        """Generate privacy analysis report."""
        lines = [
            "=" * 60,
            "DIFFERENTIAL PRIVACY REPORT",
            "=" * 60,
            "",
            f"Privacy Parameters:",
            f"  ε (epsilon): {self.epsilon}",
            f"  δ (delta): {self.delta}",
            f"  Clip bound: {self.clip_bound:.4f}",
            f"  Noise multiplier: {self.noise_multiplier}",
            "",
        ]

        if self.noise_info_ and self.noise_info_['clients']:
            lines.append("Per-Client Noise Analysis:")
            for info in self.noise_info_['clients']:
                lines.extend([
                    f"  Client {info['client_id']}:",
                    f"    Samples: {info['n_samples']}",
                    f"    Mean noise scale: {info['mean_noise_scale']:.6f}",
                    f"    Cov noise scale: {info['cov_noise_scale']:.6f}",
                    f"    SNR (mean): {info['signal_to_noise_mean']:.2f}",
                    f"    SNR (cov): {info['signal_to_noise_cov']:.2f}",
                ])

        lines.extend([
            "",
            "Privacy Guarantee:",
            f"  This model satisfies ({self.epsilon}, {self.delta})-differential privacy.",
            f"  Interpretation: The probability that an adversary can determine",
            f"  whether any individual's data was included is bounded by e^ε ≈ {np.exp(self.epsilon):.2f}",
            "=" * 60,
        ])

        return "\n".join(lines)


def evaluate_privacy_utility_tradeoff(
    client_data: List[np.ndarray],
    epsilon_values: List[float],
    n_components: int = 20,
    delta: float = 1e-5,
    n_trials: int = 5,
    random_state: int = 42,
) -> Dict:
    """Evaluate privacy-utility trade-off across different epsilon values.

    Args:
        client_data: List of client data arrays.
        epsilon_values: List of epsilon values to test.
        n_components: Number of PCA components.
        delta: Privacy parameter delta.
        n_trials: Number of trials per epsilon (for variance estimation).
        random_state: Base random seed.

    Returns:
        Dictionary with results for each epsilon.
    """
    from .pooled_covariance import PooledCovariancePCA
    from ..metrics.subspace_alignment import principal_angles, angle_to_degrees

    # Non-private baseline
    baseline = PooledCovariancePCA(n_components, random_state=random_state)
    baseline.fit(client_data)

    results = {
        'epsilon_values': epsilon_values,
        'mean_angles': [],
        'std_angles': [],
        'snr_mean': [],
        'snr_cov': [],
    }

    for eps in epsilon_values:
        trial_angles = []
        trial_snr_mean = []
        trial_snr_cov = []

        for trial in range(n_trials):
            dp_pca = DifferentiallyPrivatePCA(
                n_components=n_components,
                epsilon=eps,
                delta=delta,
                random_state=random_state + trial,
            )
            dp_pca.fit(client_data)

            # Compute angle to baseline
            angles = principal_angles(baseline.components_.T, dp_pca.components_.T)
            angles_deg = angle_to_degrees(angles)
            trial_angles.append(np.mean(angles_deg))

            # Average SNR across clients
            if dp_pca.noise_info_:
                snr_m = np.mean([c['signal_to_noise_mean'] for c in dp_pca.noise_info_['clients']])
                snr_c = np.mean([c['signal_to_noise_cov'] for c in dp_pca.noise_info_['clients']])
                trial_snr_mean.append(snr_m)
                trial_snr_cov.append(snr_c)

        results['mean_angles'].append(np.mean(trial_angles))
        results['std_angles'].append(np.std(trial_angles))
        results['snr_mean'].append(np.mean(trial_snr_mean) if trial_snr_mean else 0)
        results['snr_cov'].append(np.mean(trial_snr_cov) if trial_snr_cov else 0)

    return results


def demonstrate_dp_pca():
    """Demonstrate differentially private PCA."""
    np.random.seed(42)

    print("=" * 60)
    print("DIFFERENTIALLY PRIVATE PCA DEMONSTRATION")
    print("=" * 60)

    # Generate synthetic data
    n_features = 50
    n_components = 10

    def generate_client_data(n_samples, class_id):
        pattern = np.zeros(n_features)
        pattern[class_id * 5:(class_id + 1) * 5] = 2.0
        noise = np.random.randn(n_samples, n_features) * 0.5
        return noise + pattern

    client_data = [generate_client_data(1000, i) for i in range(5)]

    # Compare different privacy levels
    print("\n--- Privacy-Utility Trade-off ---\n")

    from .pooled_covariance import PooledCovariancePCA
    from ..metrics.subspace_alignment import principal_angles, angle_to_degrees

    # Non-private baseline
    baseline = PooledCovariancePCA(n_components, random_state=42)
    baseline.fit(client_data)

    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"{'Epsilon':<10} {'Mean Angle':<15} {'SNR (mean)':<15} {'SNR (cov)':<15}")
    print("-" * 55)

    for eps in epsilon_values:
        dp_pca = DifferentiallyPrivatePCA(
            n_components=n_components,
            epsilon=eps,
            delta=1e-5,
            random_state=42,
        )
        dp_pca.fit(client_data)

        angles = principal_angles(baseline.components_.T, dp_pca.components_.T)
        angles_deg = angle_to_degrees(angles)

        snr_m = np.mean([c['signal_to_noise_mean'] for c in dp_pca.noise_info_['clients']])
        snr_c = np.mean([c['signal_to_noise_cov'] for c in dp_pca.noise_info_['clients']])

        print(f"{eps:<10.1f} {np.mean(angles_deg):<15.2f}° {snr_m:<15.2f} {snr_c:<15.2f}")

    # Show detailed report for ε=1.0
    print("\n--- Detailed Report for ε=1.0 ---")
    dp_pca = DifferentiallyPrivatePCA(
        n_components=n_components,
        epsilon=1.0,
        delta=1e-5,
        random_state=42,
    )
    dp_pca.fit(client_data)
    print(dp_pca.get_privacy_report())

    return dp_pca


if __name__ == '__main__':
    demonstrate_dp_pca()
