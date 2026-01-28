"""Communication-Compressed Distributed PCA.

This module implements compression techniques to reduce the O(d²) communication
cost of sending covariance matrices in distributed PCA.

For high-dimensional data (e.g., CIFAR-10 with d=3072), a full covariance matrix
requires d² ≈ 9.4 million values. Compression can reduce this significantly.

Compression Strategies:
1. Low-rank: Send top-k eigenvectors instead of full covariance → O(d×k)
2. Sketching: Random projection to compress covariance → O(m²)
3. Quantization: Reduce precision (32-bit → 8-bit) → 4x reduction
4. Top-k Sparsification: Only send largest entries → O(k)

Trade-off: More compression = less communication but potentially less accuracy.
"""

from typing import List, Dict, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import numpy as np

from .base import DistributedPCABase


class CompressionMethod(Enum):
    """Available compression methods."""
    NONE = "none"
    LOW_RANK = "low_rank"
    SKETCH = "sketch"
    QUANTIZE = "quantize"
    TOP_K = "top_k"
    COMBINED = "combined"  # Low-rank + quantization


@dataclass
class CompressionStats:
    """Statistics about compression."""
    original_size: int  # Number of floats without compression
    compressed_size: int  # Number of values after compression
    compression_ratio: float  # original / compressed
    bits_per_value: int  # Bits used per value
    total_bits: int  # Total bits transmitted


def compute_original_size(n_features: int, include_mean: bool = True) -> int:
    """Compute original communication size (number of float64 values)."""
    # Covariance: d×d symmetric, but we send full matrix
    # Mean: d values
    # Sample count: 1 value
    cov_size = n_features * n_features
    mean_size = n_features if include_mean else 0
    return cov_size + mean_size + 1


class LowRankCompressor:
    """Compress covariance using low-rank approximation.

    Instead of sending full d×d covariance, send:
    - Top-k eigenvectors: d×k matrix
    - Top-k eigenvalues: k values
    - Residual trace (optional): 1 value

    Communication: O(d×k + k) instead of O(d²)
    """

    def __init__(self, rank: int, preserve_trace: bool = True):
        """Initialize low-rank compressor.

        Args:
            rank: Number of eigenvectors to keep.
            preserve_trace: If True, also send residual trace for better reconstruction.
        """
        self.rank = rank
        self.preserve_trace = preserve_trace

    def compress(self, covariance: np.ndarray) -> Dict:
        """Compress covariance matrix to low-rank representation."""
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep top-k
        k = min(self.rank, len(eigenvalues))
        top_eigenvalues = eigenvalues[:k]
        top_eigenvectors = eigenvectors[:, :k]

        result = {
            'eigenvalues': top_eigenvalues,
            'eigenvectors': top_eigenvectors,
            'rank': k,
        }

        if self.preserve_trace:
            # Residual trace for better reconstruction
            result['residual_trace'] = np.sum(eigenvalues[k:])

        return result

    def decompress(self, compressed: Dict, n_features: int) -> np.ndarray:
        """Reconstruct covariance from low-rank representation."""
        eigenvectors = compressed['eigenvectors']
        eigenvalues = compressed['eigenvalues']

        # Reconstruct: V @ diag(λ) @ V^T
        covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Add residual trace as uniform diagonal if available
        if 'residual_trace' in compressed and compressed['residual_trace'] > 0:
            residual_per_dim = compressed['residual_trace'] / (n_features - compressed['rank'])
            # Add to dimensions not captured by top-k
            # This is an approximation - spread residual uniformly
            covariance += residual_per_dim * np.eye(n_features)
            # Subtract from captured dimensions to not double-count
            for i in range(compressed['rank']):
                covariance[i, i] -= residual_per_dim

        return covariance

    def get_size(self, n_features: int) -> int:
        """Get compressed size in number of values."""
        k = min(self.rank, n_features)
        size = n_features * k  # eigenvectors
        size += k  # eigenvalues
        if self.preserve_trace:
            size += 1  # residual trace
        return size


class SketchCompressor:
    """Compress covariance using random sketching.

    Use random projection: S = R @ C @ R^T
    where R is m×d random matrix (m << d).

    Communication: O(m²) instead of O(d²)
    """

    def __init__(self, sketch_size: int, seed: Optional[int] = None):
        """Initialize sketch compressor.

        Args:
            sketch_size: Size of sketch (m). Communication is O(m²).
            seed: Random seed for reproducibility.
        """
        self.sketch_size = sketch_size
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._projection_matrix = None

    def _get_projection_matrix(self, n_features: int) -> np.ndarray:
        """Get or create random projection matrix."""
        if self._projection_matrix is None or self._projection_matrix.shape[1] != n_features:
            # Use sparse random projection for efficiency
            # Gaussian random matrix scaled by 1/sqrt(m)
            self._rng = np.random.RandomState(self.seed)  # Reset for reproducibility
            self._projection_matrix = self._rng.randn(self.sketch_size, n_features) / np.sqrt(self.sketch_size)
        return self._projection_matrix

    def compress(self, covariance: np.ndarray) -> Dict:
        """Compress covariance using random sketch."""
        n_features = covariance.shape[0]
        R = self._get_projection_matrix(n_features)

        # Sketch: S = R @ C @ R^T
        sketch = R @ covariance @ R.T

        return {
            'sketch': sketch,
            'n_features': n_features,
        }

    def decompress(self, compressed: Dict, n_features: int) -> np.ndarray:
        """Reconstruct covariance from sketch (approximate)."""
        R = self._get_projection_matrix(n_features)
        sketch = compressed['sketch']

        # Pseudo-inverse reconstruction: C ≈ R^T @ S @ R
        # This is approximate but preserves key statistics
        covariance = R.T @ sketch @ R

        # Ensure symmetric
        covariance = (covariance + covariance.T) / 2

        return covariance

    def get_size(self, n_features: int) -> int:
        """Get compressed size."""
        return self.sketch_size * self.sketch_size


class QuantizationCompressor:
    """Compress using reduced precision quantization.

    Convert float64 values to lower precision:
    - 32-bit: 2x compression
    - 16-bit: 4x compression
    - 8-bit: 8x compression
    """

    def __init__(self, bits: int = 16):
        """Initialize quantization compressor.

        Args:
            bits: Bits per value (8, 16, or 32).
        """
        if bits not in [8, 16, 32]:
            raise ValueError("bits must be 8, 16, or 32")
        self.bits = bits

    def compress(self, covariance: np.ndarray) -> Dict:
        """Quantize covariance matrix."""
        # Store min/max for dequantization
        min_val = covariance.min()
        max_val = covariance.max()

        # Normalize to [0, 1]
        if max_val > min_val:
            normalized = (covariance - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(covariance)

        # Quantize
        if self.bits == 32:
            quantized = normalized.astype(np.float32)
        elif self.bits == 16:
            # Scale to uint16 range
            quantized = (normalized * 65535).astype(np.uint16)
        else:  # 8-bit
            quantized = (normalized * 255).astype(np.uint8)

        return {
            'quantized': quantized,
            'min_val': min_val,
            'max_val': max_val,
            'bits': self.bits,
        }

    def decompress(self, compressed: Dict, n_features: int) -> np.ndarray:
        """Dequantize covariance matrix."""
        quantized = compressed['quantized']
        min_val = compressed['min_val']
        max_val = compressed['max_val']
        bits = compressed['bits']

        # Dequantize
        if bits == 32:
            normalized = quantized.astype(np.float64)
        elif bits == 16:
            normalized = quantized.astype(np.float64) / 65535
        else:
            normalized = quantized.astype(np.float64) / 255

        # Denormalize
        if max_val > min_val:
            covariance = normalized * (max_val - min_val) + min_val
        else:
            covariance = np.full_like(normalized, min_val, dtype=np.float64)

        return covariance

    def get_size(self, n_features: int) -> Tuple[int, int]:
        """Get compressed size (values, bits per value)."""
        return n_features * n_features, self.bits


class TopKCompressor:
    """Compress by keeping only top-k largest magnitude entries.

    Communication: O(k) instead of O(d²)
    Best for sparse covariance structures.
    """

    def __init__(self, k: int):
        """Initialize top-k compressor.

        Args:
            k: Number of entries to keep.
        """
        self.k = k

    def compress(self, covariance: np.ndarray) -> Dict:
        """Keep only top-k entries by magnitude."""
        n = covariance.shape[0]

        # Get upper triangle indices (covariance is symmetric)
        triu_indices = np.triu_indices(n)
        values = covariance[triu_indices]

        # Find top-k by magnitude
        k = min(self.k, len(values))
        top_k_idx = np.argsort(np.abs(values))[-k:]

        # Store indices and values
        rows = triu_indices[0][top_k_idx]
        cols = triu_indices[1][top_k_idx]
        top_values = values[top_k_idx]

        # Also store diagonal for trace preservation
        diagonal = np.diag(covariance)

        return {
            'rows': rows,
            'cols': cols,
            'values': top_values,
            'diagonal': diagonal,
            'n_features': n,
        }

    def decompress(self, compressed: Dict, n_features: int) -> np.ndarray:
        """Reconstruct covariance from sparse representation."""
        covariance = np.zeros((n_features, n_features))

        # Set diagonal first
        np.fill_diagonal(covariance, compressed['diagonal'])

        # Add top-k entries (symmetric)
        rows = compressed['rows']
        cols = compressed['cols']
        values = compressed['values']

        for r, c, v in zip(rows, cols, values):
            if r != c:  # Don't double-set diagonal
                covariance[r, c] = v
                covariance[c, r] = v

        return covariance

    def get_size(self, n_features: int) -> int:
        """Get compressed size."""
        k = min(self.k, n_features * (n_features + 1) // 2)
        # k values + k row indices + k col indices + diagonal
        return k * 3 + n_features


class CompressedDistributedPCA(DistributedPCABase):
    """Distributed PCA with communication compression.

    Reduces communication cost from O(d²) to O(d×k), O(m²), or better
    depending on compression method.

    Attributes:
        n_components: Number of principal components.
        compression_method: Which compression to use.
        compression_params: Parameters for the compressor.
    """

    def __init__(
        self,
        n_components: int,
        compression_method: Union[str, CompressionMethod] = "low_rank",
        compression_rank: int = 100,
        sketch_size: int = 200,
        quantization_bits: int = 16,
        top_k: int = 10000,
        random_state: Optional[int] = None,
    ):
        """Initialize compressed distributed PCA.

        Args:
            n_components: Number of principal components.
            compression_method: Compression method to use.
            compression_rank: Rank for low-rank compression.
            sketch_size: Size for sketch compression.
            quantization_bits: Bits for quantization (8, 16, 32).
            top_k: Number of entries for top-k compression.
            random_state: Random seed.
        """
        super().__init__(n_components, random_state)

        if isinstance(compression_method, str):
            compression_method = CompressionMethod(compression_method)
        self.compression_method = compression_method

        # Create compressor
        if compression_method == CompressionMethod.LOW_RANK:
            self.compressor = LowRankCompressor(compression_rank)
        elif compression_method == CompressionMethod.SKETCH:
            self.compressor = SketchCompressor(sketch_size, random_state)
        elif compression_method == CompressionMethod.QUANTIZE:
            self.compressor = QuantizationCompressor(quantization_bits)
        elif compression_method == CompressionMethod.TOP_K:
            self.compressor = TopKCompressor(top_k)
        elif compression_method == CompressionMethod.COMBINED:
            # Low-rank + quantization
            self.compressor = LowRankCompressor(compression_rank)
            self.quantizer = QuantizationCompressor(quantization_bits)
        else:
            self.compressor = None

        self.compression_stats_: Optional[List[CompressionStats]] = None

    def fit(self, client_data: List[np.ndarray]) -> 'CompressedDistributedPCA':
        """Fit with compressed communication."""
        # Compute and compress local statistics
        local_results = []
        self.compression_stats_ = []

        for data in client_data:
            result, stats = self.local_computation_compressed(data)
            local_results.append(result)
            self.compression_stats_.append(stats)

        # Aggregate
        self.aggregate(local_results)

        return self

    def local_computation(self, data: np.ndarray) -> Dict:
        """Compute local statistics (uncompressed)."""
        n_samples = len(data)
        mean = np.mean(data, axis=0)
        centered = data - mean
        covariance = (centered.T @ centered) / n_samples

        return {
            'n_samples': n_samples,
            'mean': mean,
            'covariance': covariance,
        }

    def local_computation_compressed(self, data: np.ndarray) -> Tuple[Dict, CompressionStats]:
        """Compute local statistics with compression."""
        n_samples, n_features = data.shape
        mean = np.mean(data, axis=0)
        centered = data - mean
        covariance = (centered.T @ centered) / n_samples

        # Compute original size
        original_size = compute_original_size(n_features)

        # Compress covariance
        if self.compression_method == CompressionMethod.NONE or self.compressor is None:
            compressed_cov = covariance
            compressed_size = n_features * n_features
            bits_per_value = 64
        elif self.compression_method == CompressionMethod.QUANTIZE:
            compressed_cov = self.compressor.compress(covariance)
            compressed_size = n_features * n_features
            bits_per_value = self.compressor.bits
        elif self.compression_method == CompressionMethod.COMBINED:
            # Low-rank then quantize
            lr_compressed = self.compressor.compress(covariance)
            # Quantize eigenvectors and eigenvalues
            lr_compressed['eigenvectors'] = self.quantizer.compress(lr_compressed['eigenvectors'])
            compressed_cov = lr_compressed
            compressed_size = self.compressor.get_size(n_features)
            bits_per_value = self.quantizer.bits
        else:
            compressed_cov = self.compressor.compress(covariance)
            compressed_size = self.compressor.get_size(n_features)
            bits_per_value = 64

        # Compression stats
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size + n_features + 1,  # + mean + n_samples
            compression_ratio=original_size * 64 / (compressed_size * bits_per_value + n_features * 64 + 64),
            bits_per_value=bits_per_value,
            total_bits=(compressed_size * bits_per_value + n_features * 64 + 64),
        )

        result = {
            'n_samples': n_samples,
            'mean': mean,
            'compressed_covariance': compressed_cov,
            'n_features': n_features,
        }

        return result, stats

    def aggregate(self, local_results: List[Dict]) -> None:
        """Aggregate compressed local statistics."""
        n_features = local_results[0]['n_features']
        total_samples = sum(r['n_samples'] for r in local_results)
        weights = [r['n_samples'] / total_samples for r in local_results]

        # Weighted mean
        global_mean = sum(w * r['mean'] for w, r in zip(weights, local_results))

        # Decompress and aggregate covariances
        global_cov = np.zeros((n_features, n_features))

        for w, r in zip(weights, local_results):
            # Decompress covariance
            if self.compression_method == CompressionMethod.NONE or self.compressor is None:
                cov = r['compressed_covariance']
            elif self.compression_method == CompressionMethod.COMBINED:
                # Dequantize first
                compressed = r['compressed_covariance']
                compressed['eigenvectors'] = self.quantizer.decompress(
                    compressed['eigenvectors'], n_features
                )
                cov = self.compressor.decompress(compressed, n_features)
            else:
                cov = self.compressor.decompress(r['compressed_covariance'], n_features)

            # Add to global with between-client variance
            mean_diff = r['mean'] - global_mean
            global_cov += w * (cov + np.outer(mean_diff, mean_diff))

        # Ensure symmetric and PSD
        global_cov = (global_cov + global_cov.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(global_cov))
        if min_eig < 0:
            global_cov += (-min_eig + 1e-6) * np.eye(n_features)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(global_cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store results
        k = min(self.n_components, len(eigenvalues))
        self.components_ = eigenvectors[:, :k].T
        self.mean_ = global_mean
        self._compute_explained_variance(np.maximum(eigenvalues, 0))

    def get_compression_report(self) -> str:
        """Generate compression analysis report."""
        if not self.compression_stats_:
            return "No compression statistics available."

        lines = [
            "=" * 60,
            "COMMUNICATION COMPRESSION REPORT",
            "=" * 60,
            "",
            f"Compression Method: {self.compression_method.value}",
            "",
            "Per-Client Statistics:",
        ]

        total_original = 0
        total_compressed = 0

        for i, stats in enumerate(self.compression_stats_):
            lines.extend([
                f"  Client {i}:",
                f"    Original: {stats.original_size:,} values ({stats.original_size * 64 / 8 / 1024:.1f} KB)",
                f"    Compressed: {stats.compressed_size:,} values ({stats.total_bits / 8 / 1024:.1f} KB)",
                f"    Compression ratio: {stats.compression_ratio:.2f}x",
            ])
            total_original += stats.original_size * 64
            total_compressed += stats.total_bits

        total_ratio = total_original / total_compressed if total_compressed > 0 else 0

        lines.extend([
            "",
            "Total Communication:",
            f"  Without compression: {total_original / 8 / 1024:.1f} KB",
            f"  With compression: {total_compressed / 8 / 1024:.1f} KB",
            f"  Overall reduction: {total_ratio:.2f}x",
            f"  Bandwidth saved: {(1 - 1/total_ratio) * 100:.1f}%",
            "=" * 60,
        ])

        return "\n".join(lines)


def evaluate_compression_methods(
    client_data: List[np.ndarray],
    n_components: int = 20,
    random_state: int = 42,
) -> Dict:
    """Evaluate different compression methods."""
    from .pooled_covariance import PooledCovariancePCA
    from ..metrics.subspace_alignment import principal_angles, angle_to_degrees

    # Baseline (no compression)
    baseline = PooledCovariancePCA(n_components, random_state=random_state)
    baseline.fit(client_data)

    n_features = client_data[0].shape[1]

    # Methods to test
    methods = {
        'None': {'compression_method': 'none'},
        'Low-rank (k=50)': {'compression_method': 'low_rank', 'compression_rank': 50},
        'Low-rank (k=100)': {'compression_method': 'low_rank', 'compression_rank': 100},
        'Low-rank (k=200)': {'compression_method': 'low_rank', 'compression_rank': 200},
        'Sketch (m=100)': {'compression_method': 'sketch', 'sketch_size': 100},
        'Sketch (m=200)': {'compression_method': 'sketch', 'sketch_size': 200},
        'Quantize (16-bit)': {'compression_method': 'quantize', 'quantization_bits': 16},
        'Quantize (8-bit)': {'compression_method': 'quantize', 'quantization_bits': 8},
        'Top-k (k=10000)': {'compression_method': 'top_k', 'top_k': 10000},
    }

    results = {}

    for name, params in methods.items():
        pca = CompressedDistributedPCA(
            n_components=n_components,
            random_state=random_state,
            **params,
        )
        pca.fit(client_data)

        # Compute angle
        angles = principal_angles(baseline.components_.T, pca.components_.T)
        mean_angle = np.mean(angle_to_degrees(angles))

        # Compression stats
        if pca.compression_stats_:
            total_original = sum(s.original_size * 64 for s in pca.compression_stats_)
            total_compressed = sum(s.total_bits for s in pca.compression_stats_)
            ratio = total_original / total_compressed if total_compressed > 0 else 1
        else:
            ratio = 1.0

        results[name] = {
            'mean_angle': mean_angle,
            'compression_ratio': ratio,
            'bandwidth_kb': total_compressed / 8 / 1024 if pca.compression_stats_ else 0,
        }

    return results


def demonstrate_compression():
    """Demonstrate communication compression."""
    np.random.seed(42)

    print("=" * 60)
    print("COMMUNICATION COMPRESSION DEMONSTRATION")
    print("=" * 60)

    # Generate high-dimensional data (like CIFAR-10)
    n_features = 500  # Smaller for demo
    n_components = 20

    def generate_data(n_samples, class_id):
        pattern = np.zeros(n_features)
        pattern[class_id * 50:(class_id + 1) * 50] = 2.0
        return np.random.randn(n_samples, n_features) * 0.5 + pattern

    client_data = [generate_data(1000, i) for i in range(5)]

    print(f"\nData: {len(client_data)} clients, {n_features} features each")
    print(f"Full covariance size: {n_features}² = {n_features**2:,} values")
    print(f"                    = {n_features**2 * 8 / 1024:.1f} KB per client")

    # Evaluate methods
    print("\n--- Compression Methods Comparison ---\n")
    results = evaluate_compression_methods(client_data, n_components)

    print(f"{'Method':<25} {'Angle':<12} {'Ratio':<12} {'Size (KB)':<12}")
    print("-" * 55)

    for name, r in results.items():
        print(f"{name:<25} {r['mean_angle']:<12.2f}° {r['compression_ratio']:<12.1f}x {r['bandwidth_kb']:<12.1f}")

    # Show detailed report for low-rank
    print("\n--- Detailed Report: Low-rank (k=100) ---")
    pca = CompressedDistributedPCA(
        n_components=n_components,
        compression_method='low_rank',
        compression_rank=100,
        random_state=42,
    )
    pca.fit(client_data)
    print(pca.get_compression_report())

    return results


if __name__ == '__main__':
    demonstrate_compression()
