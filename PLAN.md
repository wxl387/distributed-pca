# Distributed PCA Research - Project Plan & Progress

## Overview

Research and implement distributed PCA mechanisms for federated/collaborative learning where collaborators cannot share raw data but need to integrate local PCAs into a global PCA for projecting data into a common coordinate space.

**Datasets:** CIFAR-10, MNIST (real datasets via OpenML)
**Goal:** Comprehensive comparison of 5+ methods with thorough non-IID evaluation

---

## Progress Tracking

- [x] Phase 1: Foundation
  - [x] Project structure setup
  - [x] PLAN.md created
  - [x] Data loading (MNIST, CIFAR-10)
  - [x] Data partitioners (IID, Dirichlet, shard, quantity skew)
  - [x] Centralized PCA baseline
  - [x] Basic metrics
- [x] Phase 2: Core Methods
  - [x] P-COV (Pooled Covariance)
  - [x] SUB-IT (Subspace Iteration)
  - [x] Subspace alignment metrics
- [x] Phase 3: Additional Methods
  - [x] AP-STACK
  - [x] QR-PCA
  - [x] AP-COV
  - [x] Downstream classification evaluation
- [x] Phase 4: Experiments
  - [x] Accuracy vs Centralized
  - [x] Non-IID Robustness
  - [x] Scalability
  - [x] Downstream Classification
- [x] Phase 5: Novel Contributions (ALL COMPLETE!)
  - [x] Adaptive Method Selection
  - [x] Streaming Updates
  - [x] Differential Privacy
  - [x] Communication Compression

---

## What We Did

### 1. Research Phase
- Conducted web research on distributed/federated PCA methods
- Identified 7 methods from literature: P-COV, SUB-IT, QR-PCA, AP-STACK, AP-COV, FAST-PCA, DSA
- Reviewed evaluation metrics: subspace alignment (most stringent), reconstruction error, explained variance, classification accuracy
- Analyzed non-IID data partitioning strategies from NIID-Bench

### 2. Implementation Phase
Built a complete distributed PCA research framework:

```
distributed_pca_2/
├── PLAN.md                           # This file
├── requirements.txt                  # Dependencies
├── src/
│   ├── algorithms/                   # 10 distributed PCA methods
│   │   ├── base.py                  # Abstract base class
│   │   ├── centralized_pca.py       # Baseline (ground truth)
│   │   ├── pooled_covariance.py     # P-COV (Exact) ★ RECOMMENDED
│   │   ├── subspace_iteration.py    # SUB-IT (Exact, iterative)
│   │   ├── qr_pca.py                # QR-PCA (Exact)
│   │   ├── approximate_stack.py     # AP-STACK (Approximate)
│   │   ├── approximate_cov.py       # AP-COV (Approximate)
│   │   ├── adaptive.py              # AdaptiveDistributedPCA
│   │   ├── streaming.py             # StreamingDistributedPCA
│   │   ├── differential_privacy.py  # DifferentiallyPrivatePCA
│   │   └── compression.py           # CompressedDistributedPCA ★ NEW
│   ├── data/
│   │   ├── datasets.py              # MNIST, CIFAR-10 loaders
│   │   └── partitioners.py          # IID, Dirichlet, shard, quantity skew
│   ├── metrics/
│   │   ├── subspace_alignment.py    # Principal angles (key metric)
│   │   ├── reconstruction.py        # Reconstruction error
│   │   ├── variance.py              # Explained variance
│   │   └── downstream.py            # Classification accuracy
│   └── visualization/
│       └── plots.py                  # Plotting utilities (incl. projection overlays)
├── experiments/
│   ├── run_experiment.py            # Full experiment runner
│   ├── demo_experiment.py           # Quick demo
│   ├── full_experiment.py           # Comprehensive experiments
│   ├── test_adaptive.py             # Adaptive selection test ★ NEW
│   ├── test_adaptive_real.py        # Adaptive test on MNIST ★ NEW
│   ├── test_streaming.py            # Streaming updates test
│   ├── test_differential_privacy.py # DP-PCA test
│   ├── test_compression.py          # Compression test
│   ├── visualize_projections.py     # Synthetic data visualization ★ NEW
│   └── visualize_real_data.py       # MNIST/CIFAR-10 visualization ★ NEW
└── tests/
    └── test_algorithms.py           # Unit tests
```

### 3. Experiment Phase
Ran comprehensive experiments comparing all methods across:
- **Datasets:** Real MNIST (784-dim, 60k train/10k test), Real CIFAR-10 (3072-dim, 50k train/10k test)
- **Clients:** 10 federated clients
- **Partitions:** IID, Dirichlet (α=0.1, 0.5), Shard-based
- **Metrics:** Mean angle vs centralized, reconstruction MSE, classification accuracy

---

## Methods Implemented

| Method | Type | Description | Communication |
|--------|------|-------------|---------------|
| **P-COV** | Exact | Aggregate weighted covariance matrices | 2 rounds, O(d²) |
| **SUB-IT** | Exact | Iterative subspace iteration | Multiple rounds |
| **QR-PCA** | Exact | QR factorization aggregation | 1 round, O(d×r) |
| **AP-STACK** | Approximate | Stack local eigenvectors + PCA | 1 round, O(d×r) |
| **AP-COV** | Approximate | Simple covariance averaging | 1 round, O(d²) |
| **Adaptive** | Auto | Detects heterogeneity, selects best method | Varies |
| **Streaming** | Exact | Incremental updates for dynamic clients | O(d²) per update |
| **DP-PCA** | Private | (ε,δ)-differential privacy with noise | 2 rounds, O(d²) |
| **Compressed** | Variable | Low-rank/quantization compression | 2 rounds, O(d×k) |

---

## Experiment Results (Real Datasets)

### MNIST (784-dim, 10 clients, 50 components)
| Partition | P-COV | SUB-IT | AP-STACK | QR-PCA | AP-COV |
|-----------|-------|--------|----------|--------|--------|
| IID | **0.01°** | 0.18° | 13.07° | 1.06° | 0.03° |
| Dirichlet α=0.5 | **0.01°** | 0.17° | 12.62° | 8.81° | 0.30° |
| Dirichlet α=0.1 | **0.01°** | 0.17° | 39.72° | 13.17° | 0.61° |
| Shard-based | **0.01°** | 0.17° | 8.78° | 1.80° | 1.04° |

### CIFAR-10 (3072-dim, 10 clients, 50 components)
| Partition | P-COV | SUB-IT | AP-STACK | QR-PCA | AP-COV |
|-----------|-------|--------|----------|--------|--------|
| IID | **0.00°** | 0.65° | 0.76° | 0.11° | 0.02° |
| Dirichlet α=0.5 | **0.00°** | 0.65° | 10.25° | 1.77° | 0.09° |
| Dirichlet α=0.1 | **0.00°** | 0.65° | 11.97° | 1.91° | 0.13° |
| Shard-based | **0.00°** | 0.65° | 1.93° | 0.30° | 0.18° |

### Classification Accuracy (KNN, k=5)
| Dataset | Partition | Centralized | P-COV | AP-STACK | AP-COV |
|---------|-----------|-------------|-------|----------|--------|
| MNIST | IID | 95.65% | **95.65%** | 95.64% | 95.65% |
| MNIST | Dirichlet α=0.1 | 95.65% | **95.65%** | 94.69% | 95.55% |
| CIFAR-10 | IID | 39.54% | **39.54%** | 39.51% | 39.54% |
| CIFAR-10 | Dirichlet α=0.1 | 39.54% | **39.54%** | 39.49% | 39.54% |

---

## Key Findings

### 1. P-COV is the Clear Winner ★
- **Exact**: ≈0° angle across ALL partition types (matches centralized perfectly)
- **Robust**: No degradation on non-IID data (stays at 0.01° regardless of heterogeneity)
- **Efficient**: Only 2 communication rounds
- **Simple**: Easy to implement and understand

### 2. AP-COV is Surprisingly Good
- Nearly as accurate as P-COV on real datasets (0.02-1.04° on MNIST, 0.02-0.18° on CIFAR-10)
- Much more robust than expected from literature
- Simpler implementation than P-COV (no between-client mean correction)

### 3. AP-STACK Degrades Severely on Non-IID
- **IID**: Reasonable (0.76-13° depending on dataset)
- **Dirichlet α=0.1**: Severe degradation (up to 40° on MNIST!)
- Not recommended for heterogeneous federated settings

### 4. QR-PCA Shows Moderate Degradation
- Good on IID data but degrades on non-IID
- MNIST Dirichlet α=0.1: 13.17° (significant deviation)
- Better than AP-STACK but worse than exact methods

### 5. Classification Accuracy
- **P-COV matches centralized accuracy exactly** (95.65% on MNIST, 39.54% on CIFAR-10)
- AP-STACK shows slight accuracy drop on heavily non-IID data (94.69% vs 95.65%)
- For downstream tasks, subspace angle translates to real accuracy differences

---

## Final Recommendation

**For your collaborative study scenario, use P-COV (Pooled Covariance)**:

```
Algorithm:
1. Each collaborator computes: local mean (μ_k), local covariance (C_k), sample count (n_k)
2. Send to central server (2 rounds of communication)
3. Server computes:
   - Global mean: μ = Σ(n_k × μ_k) / Σ(n_k)
   - Global covariance: C = Σ(n_k × (C_k + (μ_k - μ)(μ_k - μ)ᵀ)) / Σ(n_k)
4. Eigendecompose C for principal components
```

**Why P-COV?**
- Guarantees exact global PCA (identical to pooling all data)
- Works perfectly regardless of how heterogeneous data is distributed
- Only requires sending covariance matrices (no raw data shared)
- Mathematically proven to be exact

---

## How to Run

```bash
# Install dependencies
pip install numpy scipy scikit-learn matplotlib pandas

# Quick demo with synthetic data
python experiments/demo_experiment.py

# Full experiments (requires torch for real MNIST/CIFAR)
pip install torch torchvision
python experiments/run_experiment.py --dataset mnist --partition dirichlet --alpha 0.5
```

---

## Phase 5: Adaptive Method Selection ★ NEW

Implemented an intelligent method selection system that automatically detects data heterogeneity and chooses the optimal distributed PCA method.

### How It Works

1. **Heterogeneity Detection** (without requiring labels):
   - Mean divergence: Distance between client means and global mean
   - Covariance divergence: Frobenius distance between local and average covariance
   - Eigenspectrum divergence: KL divergence of normalized eigenvalue distributions
   - Variance divergence: Difference in total variance across clients

2. **Method Selection Strategy**:
   - **Low heterogeneity (score < 0.15)**: Use AP-COV (efficient, nearly exact)
   - **Medium heterogeneity (0.15-0.35)**: Use AP-COV (robust enough)
   - **High heterogeneity (score > 0.35)**: Use P-COV (exact, no degradation)

### Results on Real MNIST

| Partition | Het. Score | Selected | Adaptive Angle | Optimal Angle |
|-----------|-----------|----------|----------------|---------------|
| IID | 0.30 | AP-COV | 0.03° | 0.01° |
| Dirichlet α=0.5 | 0.60 | P-COV | 0.01° | 0.01° |
| Dirichlet α=0.1 | 0.67 | P-COV | 0.01° | 0.01° |
| Shard-based | 0.74 | P-COV | 0.01° | 0.01° |

### Usage

```python
from src.algorithms import AdaptiveDistributedPCA

# Automatically selects best method based on data heterogeneity
adaptive = AdaptiveDistributedPCA(n_components=50, verbose=True)
adaptive.fit(client_data)

# Check what was selected
print(adaptive.selected_method_name_)  # 'P-COV' or 'AP-COV'
print(adaptive.get_selection_report())  # Detailed report
```

### Key Benefits
- **No manual tuning**: Automatically adapts to data distribution
- **Privacy-preserving detection**: Heterogeneity computed from statistics, not raw data
- **Optimal trade-off**: Balances accuracy and communication efficiency

---

## Phase 5: Streaming Updates ★ NEW

Implemented streaming/incremental updates for distributed PCA, enabling dynamic client participation without full recomputation.

### Capabilities

1. **Add Client**: New clients can join the federation
2. **Remove Client**: Clients can leave the federation
3. **Update Client**:
   - `replace` mode: Replace client's data entirely
   - `append` mode: Add new samples to existing data

### How It Works

Uses the mathematical property that covariances can be combined incrementally:

```
C_combined = (n1*C1 + n2*C2 + n1*n2/(n1+n2) * (μ1-μ2)(μ1-μ2)^T) / (n1+n2)
```

This allows O(d²) updates instead of O(n*d²) recomputation.

### Accuracy Verification

| Operation | Angle vs Full Recomputation |
|-----------|---------------------------|
| Initial fit | 0.000001° |
| Add client | 0.000000° |
| Remove client | 0.000000° |
| Update client | 0.000001° |
| MNIST (real data) | 0.002° |

### Usage

```python
from src.algorithms import StreamingDistributedPCA

# Initialize
streaming = StreamingDistributedPCA(n_components=50)

# Add initial clients
streaming.add_client("hospital_a", data_a)
streaming.add_client("hospital_b", data_b)

# New client joins later
streaming.add_client("hospital_c", data_c)

# Client updates their data
streaming.update_client("hospital_a", new_data, mode='append')

# Client leaves
streaming.remove_client("hospital_b")

# Transform new data
projected = streaming.transform(test_data)

# Check status
print(streaming.get_status())
```

### Key Benefits
- **Dynamic Participation**: Clients can join/leave at any time
- **Incremental Updates**: O(d²) per update instead of O(n*d²) recomputation
- **Exact Results**: Mathematically equivalent to full recomputation
- **Numerical Stability**: Periodic full recomputation to prevent drift

---

## Phase 5: Differential Privacy ★ NEW

Implemented (ε, δ)-differential privacy for distributed PCA, providing formal privacy guarantees.

### How It Works

1. **Data Clipping**: Bound each sample's L2 norm to limit sensitivity
2. **Sensitivity Computation**:
   - Mean sensitivity: `clip_bound / n_samples`
   - Covariance sensitivity: `2 * clip_bound² / n_samples`
3. **Gaussian Mechanism**: Add calibrated noise to local statistics
   - Noise scale: `σ = sensitivity * √(2*ln(1.25/δ)) / ε`

### Privacy-Utility Trade-off on MNIST

| Epsilon | Privacy Level | Angle vs Baseline | Classification Accuracy |
|---------|--------------|-------------------|------------------------|
| 1.0 | High privacy | 74.2° | 92.69% |
| 5.0 | Medium | 60.0° | 94.87% |
| 10.0 | Low | 47.0° | 95.37% |
| ∞ | None | 0° | 95.65% |

### Key Findings

- **Sample size matters**: More samples per client → less relative noise
- **ε=5-10 is practical**: Good utility (~95% accuracy) with reasonable privacy
- **Covariance is challenging**: Higher sensitivity than mean (scales with clip_bound²)

### Usage

```python
from src.algorithms import DifferentiallyPrivatePCA

# Create DP-PCA with privacy budget
dp_pca = DifferentiallyPrivatePCA(
    n_components=50,
    epsilon=1.0,      # Privacy budget (smaller = more private)
    delta=1e-5,       # Should be < 1/n
    clip_bound=5.0,   # L2 norm bound for samples
)
dp_pca.fit(client_data)

# Get privacy report
print(dp_pca.get_privacy_report())

# Transform with privacy guarantee
projected = dp_pca.transform(test_data)
```

### Privacy Guarantee Interpretation

For ε=1.0, the probability that an adversary can determine whether any individual's data was included is bounded by e^ε ≈ 2.72.

---

## Phase 5: Communication Compression ★ NEW

Implemented multiple compression strategies to reduce O(d²) communication cost.

### Compression Methods

| Method | Compression | Angle | Best For |
|--------|-------------|-------|----------|
| Quantization (16-bit) | 4x | 0.02° | Nearly lossless |
| Quantization (8-bit) | 8x | 3° | Moderate compression |
| Low-rank (k=100) | 8x | 1.5° | Good balance |
| Low-rank (k=200) | 4x | 0.7° | High accuracy |
| Sketch (m=100) | 24x | 60°+ | Maximum compression |

### Real MNIST Results

| Method | Angle | Compression | Classification Accuracy |
|--------|-------|-------------|------------------------|
| None | 0° | 1x | 95.65% |
| Quantize (16-bit) | 0.02° | 4x | 95.65% |
| Quantize (8-bit) | 3° | 8x | 95.61% |
| Low-rank (k=100) | 1.5° | 8x | 95.69% |
| Low-rank (k=200) | 0.7° | 4x | 95.67% |

### High-Dimensional Data (CIFAR-10 Scale)

For d=3072 (CIFAR-10), full covariance = **72 MB per client**:
- Low-rank (k=200): 4.8 MB (93.5% savings)
- Quantization (16-bit): 18 MB (75% savings)

### Usage

```python
from src.algorithms import CompressedDistributedPCA

# Quantization (nearly lossless, 4x compression)
pca = CompressedDistributedPCA(
    n_components=50,
    compression_method='quantize',
    quantization_bits=16,
)

# Low-rank (better compression for high-dim data)
pca = CompressedDistributedPCA(
    n_components=50,
    compression_method='low_rank',
    compression_rank=200,
)

pca.fit(client_data)
print(pca.get_compression_report())
```

### Recommendations

- **For minimal accuracy loss**: Quantization (16-bit) - 4x compression, <0.1° angle
- **For high-dimensional data**: Low-rank (k=200) - balance of compression and accuracy
- **For severely limited bandwidth**: Sketch - 20x+ compression but significant accuracy loss

---

## Visualization ★ NEW

Added scatter plot visualizations to compare centralized PCA projections vs distributed PCA methods.

### Plot Types

1. **Overlay Plot**: Single scatter plot with circles (○) for centralized and X markers (×) for distributed
2. **Grid Comparison**: Side-by-side comparison of all methods

### How to Interpret

| Symbol | Meaning |
|--------|---------|
| ○ (circles) | Centralized PCA projection (ground truth) |
| × (X markers) | Distributed PCA projection |
| Colors | Class labels (10 classes) |

**Perfect overlap** between ○ and × indicates the distributed method matches centralized exactly.

### Results on Real Datasets

| Dataset | Partition | P-COV | AP-COV | AP-STACK |
|---------|-----------|-------|--------|----------|
| MNIST | IID | 0.01° | 0.02° | 13.03° |
| MNIST | Non-IID | 0.01° | 0.69° | 15.01° |
| CIFAR-10 | IID | 0.00° | 0.01° | 0.54° |
| CIFAR-10 | Non-IID | 0.00° | 0.13° | 4.58° |

### Class Labels

- **MNIST**: Digits 0-9
- **CIFAR-10**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Usage

```bash
# Synthetic data (quick demo)
python experiments/visualize_projections.py

# Real datasets (MNIST and CIFAR-10)
python experiments/visualize_real_data.py
```

Plots are saved to `results/visualizations/` directory.

### Key Observations

- **P-COV**: Perfect overlap (circles and X markers coincide exactly)
- **AP-COV**: Very slight deviation, barely visible even on non-IID data
- **AP-STACK**: Visible deviation on MNIST, especially with non-IID partitions

---

## Project Complete!

All 5 phases have been implemented:
1. ✅ Foundation (data loading, partitioning, baseline)
2. ✅ Core Methods (P-COV, SUB-IT)
3. ✅ Additional Methods (AP-STACK, QR-PCA, AP-COV)
4. ✅ Experiments (accuracy, non-IID robustness, scalability)
5. ✅ Novel Contributions (adaptive, streaming, DP, compression)
6. ✅ Visualization (projection scatter plots)

---

## Sources

- [Federated PCA for Biomedical Applications](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710634/)
- [FAST-PCA Algorithm](https://arxiv.org/pdf/2108.12373)
- [NIID-Bench: Non-IID Federated Learning Benchmark](https://github.com/Xtra-Computing/NIID-Bench)
- [Distributed Estimation of Principal Eigenspaces](https://pmc.ncbi.nlm.nih.gov/articles/PMC6836292/)
- [Personalized PCA for Federated Data](https://par.nsf.gov/servlets/purl/10506040)
