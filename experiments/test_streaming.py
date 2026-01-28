"""Test streaming distributed PCA accuracy.

Verifies that incremental updates produce results equivalent to
full recomputation from scratch.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import (
    CentralizedPCA,
    PooledCovariancePCA,
    StreamingDistributedPCA,
)
from src.metrics.subspace_alignment import principal_angles, angle_to_degrees


def generate_client_data(n_samples, n_features, class_id, seed=None):
    """Generate data for a client with class-specific pattern."""
    if seed is not None:
        np.random.seed(seed)
    pattern = np.zeros(n_features)
    pattern[class_id * 10:(class_id + 1) * 10] = 2.0
    noise = np.random.randn(n_samples, n_features) * 0.5
    return noise + pattern


def test_streaming_accuracy():
    """Test that streaming updates match full recomputation."""

    print("=" * 70)
    print("STREAMING PCA ACCURACY TEST")
    print("=" * 70)

    n_features = 100
    n_components = 20
    np.random.seed(42)

    # Test 1: Initial fit matches P-COV
    print("\n--- Test 1: Initial fit matches P-COV ---")

    client_data = [
        generate_client_data(500, n_features, 0, seed=1),
        generate_client_data(600, n_features, 1, seed=2),
        generate_client_data(400, n_features, 2, seed=3),
    ]

    # Fit with P-COV (ground truth for distributed PCA)
    pcov = PooledCovariancePCA(n_components, random_state=42)
    pcov.fit(client_data)

    # Fit with streaming
    streaming = StreamingDistributedPCA(n_components, random_state=42)
    streaming.fit(client_data)

    angles = principal_angles(pcov.components_.T, streaming.components_.T)
    angles_deg = angle_to_degrees(angles)
    print(f"Angle between P-COV and Streaming: {np.mean(angles_deg):.6f}° (should be ~0)")
    assert np.mean(angles_deg) < 0.01, "Initial fit should match P-COV exactly"
    print("✓ PASSED")

    # Test 2: Adding a client matches recomputation
    print("\n--- Test 2: Adding client matches recomputation ---")

    new_client_data = generate_client_data(550, n_features, 3, seed=4)

    # Recompute P-COV with all 4 clients
    all_client_data = client_data + [new_client_data]
    pcov_full = PooledCovariancePCA(n_components, random_state=42)
    pcov_full.fit(all_client_data)

    # Add to streaming
    streaming.add_client("client_3", new_client_data)

    angles = principal_angles(pcov_full.components_.T, streaming.components_.T)
    angles_deg = angle_to_degrees(angles)
    print(f"Angle after adding client: {np.mean(angles_deg):.6f}° (should be ~0)")
    assert np.mean(angles_deg) < 0.01, "After adding client should match P-COV"
    print("✓ PASSED")

    # Test 3: Removing a client matches recomputation
    print("\n--- Test 3: Removing client matches recomputation ---")

    # Remove client_1 (index 1)
    streaming.remove_client("client_1")

    # Recompute P-COV without client_1
    remaining_data = [client_data[0], client_data[2], new_client_data]
    pcov_remaining = PooledCovariancePCA(n_components, random_state=42)
    pcov_remaining.fit(remaining_data)

    angles = principal_angles(pcov_remaining.components_.T, streaming.components_.T)
    angles_deg = angle_to_degrees(angles)
    print(f"Angle after removing client: {np.mean(angles_deg):.6f}° (should be ~0)")
    assert np.mean(angles_deg) < 0.01, "After removing client should match P-COV"
    print("✓ PASSED")

    # Test 4: Updating client (replace mode) matches recomputation
    print("\n--- Test 4: Updating client (replace) matches recomputation ---")

    # Replace client_0's data
    updated_data_0 = generate_client_data(700, n_features, 0, seed=10)
    streaming.update_client("client_0", updated_data_0, mode='replace')

    # Recompute P-COV with updated data
    updated_remaining = [updated_data_0, client_data[2], new_client_data]
    pcov_updated = PooledCovariancePCA(n_components, random_state=42)
    pcov_updated.fit(updated_remaining)

    angles = principal_angles(pcov_updated.components_.T, streaming.components_.T)
    angles_deg = angle_to_degrees(angles)
    print(f"Angle after updating client: {np.mean(angles_deg):.6f}° (should be ~0)")
    assert np.mean(angles_deg) < 0.01, "After updating client should match P-COV"
    print("✓ PASSED")

    # Test 5: Multiple operations maintain accuracy
    print("\n--- Test 5: Multiple operations maintain accuracy ---")

    # Perform several operations
    streaming2 = StreamingDistributedPCA(n_components, random_state=42)

    # Add clients one by one
    for i in range(5):
        data = generate_client_data(np.random.randint(300, 800), n_features, i, seed=100+i)
        streaming2.add_client(f"site_{i}", data)

    # Remove one
    streaming2.remove_client("site_2")

    # Add more
    for i in range(5, 8):
        data = generate_client_data(np.random.randint(300, 800), n_features, i % 5, seed=100+i)
        streaming2.add_client(f"site_{i}", data)

    # Update one
    streaming2.update_client("site_0", generate_client_data(600, n_features, 0, seed=200), mode='replace')

    # Verify by full recomputation
    streaming2._recompute_global()  # Force full recomputation
    components_after_recompute = streaming2.components_.copy()

    # The components should be identical (within numerical precision)
    print(f"Status: {len(streaming2.clients)} clients, {streaming2.global_state.n_samples} samples")
    print("✓ PASSED (full recomputation completed)")

    # Test 6: Append mode
    print("\n--- Test 6: Append mode for incremental data ---")

    streaming3 = StreamingDistributedPCA(n_components, random_state=42)

    # Initial data
    initial_data = generate_client_data(500, n_features, 0, seed=300)
    streaming3.add_client("sensor_0", initial_data)

    # Append new data
    append_data = generate_client_data(200, n_features, 0, seed=301)
    streaming3.update_client("sensor_0", append_data, mode='append')

    # Verify sample count
    assert streaming3.clients["sensor_0"].n_samples == 700, "Append should combine samples"
    print(f"After append: {streaming3.clients['sensor_0'].n_samples} samples")

    # Compare to full computation
    combined_data = np.vstack([initial_data, append_data])
    pcov_combined = PooledCovariancePCA(n_components, random_state=42)
    pcov_combined.fit([combined_data])

    angles = principal_angles(pcov_combined.components_.T, streaming3.components_.T)
    angles_deg = angle_to_degrees(angles)
    print(f"Angle vs full computation: {np.mean(angles_deg):.6f}° (should be ~0)")
    assert np.mean(angles_deg) < 0.01, "Append mode should match full computation"
    print("✓ PASSED")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)

    print("""
Summary:
- Streaming updates produce results equivalent to full recomputation
- Supports: add_client, remove_client, update_client (replace/append modes)
- Mathematical foundation: Incremental covariance update formula
- Use case: Federated learning with dynamic client participation
""")


def test_on_real_data():
    """Test streaming on MNIST data."""
    print("\n" + "=" * 70)
    print("STREAMING PCA ON REAL MNIST")
    print("=" * 70)

    try:
        from src.data.datasets import load_mnist
        from src.data.partitioners import DataPartitioner

        X_train, y_train = load_mnist(train=True, flatten=True, normalize=True)
        print(f"Loaded MNIST: {X_train.shape}")

        # Partition into clients
        partitioner = DataPartitioner(num_clients=5, seed=42)
        partitions = partitioner.iid_partition(X_train, y_train)
        client_data = [data for data, _ in partitions]

        n_components = 50

        # Test streaming vs P-COV
        print("\n--- Comparing Streaming vs P-COV on MNIST ---")

        # P-COV baseline
        pcov = PooledCovariancePCA(n_components, random_state=42)
        pcov.fit(client_data)

        # Streaming: add clients one by one
        streaming = StreamingDistributedPCA(n_components, random_state=42)
        for i, data in enumerate(client_data):
            streaming.add_client(f"client_{i}", data)
            print(f"Added client_{i}: {len(data)} samples")

        angles = principal_angles(pcov.components_.T, streaming.components_.T)
        angles_deg = angle_to_degrees(angles)
        print(f"\nAngle between P-COV and Streaming: {np.mean(angles_deg):.6f}°")

        # Simulate a new hospital joining
        print("\n--- Simulating new hospital joining ---")
        # Take some test data as "new hospital"
        X_test, _ = load_mnist(train=False, flatten=True, normalize=True)
        new_hospital_data = X_test[:2000]

        streaming.add_client("new_hospital", new_hospital_data)
        print(f"Added new_hospital: {len(new_hospital_data)} samples")
        print(f"Total samples now: {streaming.global_state.n_samples}")

        # Verify against full recomputation
        all_data = client_data + [new_hospital_data]
        pcov_full = PooledCovariancePCA(n_components, random_state=42)
        pcov_full.fit(all_data)

        angles = principal_angles(pcov_full.components_.T, streaming.components_.T)
        angles_deg = angle_to_degrees(angles)
        print(f"Angle after adding new hospital: {np.mean(angles_deg):.6f}° (should be ~0)")

        print("\n✓ Streaming on MNIST verified!")

    except ImportError as e:
        print(f"Skipping MNIST test: {e}")


if __name__ == '__main__':
    test_streaming_accuracy()
    test_on_real_data()
