import numpy as np
import pytest

from pygridsynth.mixed_synthesis import (
    mixed_synthesis_parallel,
    mixed_synthesis_sequential,
)
from pygridsynth.mymath import random_su


def test_mixed_synthesis_sequential_basic():
    """Test basic functionality of mixed_synthesis_sequential"""
    num_qubits = 2
    eps = 1e-4
    M = 64
    unitary = random_su(num_qubits)

    result = mixed_synthesis_sequential(unitary, num_qubits, eps, M, seed=123)

    assert result is not None, "Result should not be None"
    circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt = result

    assert isinstance(circuit_list, list), "circuit_list should be a list"
    assert len(circuit_list) == M, f"circuit_list should have {M} elements"
    assert isinstance(eu_np_list, list), "eu_np_list should be a list"
    assert len(eu_np_list) == M, f"eu_np_list should have {M} elements"
    assert isinstance(probs_gptm, np.ndarray), "probs_gptm should be a numpy array"
    assert len(probs_gptm) == M, f"probs_gptm should have {M} elements"
    assert isinstance(u_choi, np.ndarray), "u_choi should be a numpy array"
    assert isinstance(u_choi_opt, np.ndarray), "u_choi_opt should be a numpy array"
    assert np.allclose(np.sum(probs_gptm), 1.0), "Probabilities should sum to 1"
    assert np.all(probs_gptm >= 0), "All probabilities should be non-negative"


def test_mixed_synthesis_parallel_basic():
    """Test basic functionality of mixed_synthesis_parallel"""
    num_qubits = 2
    eps = 1e-4
    M = 64
    unitary = random_su(num_qubits)

    result = mixed_synthesis_parallel(unitary, num_qubits, eps, M, seed=123)

    assert result is not None, "Result should not be None"
    circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt = result

    assert isinstance(circuit_list, list), "circuit_list should be a list"
    assert len(circuit_list) == M, f"circuit_list should have {M} elements"
    assert isinstance(eu_np_list, list), "eu_np_list should be a list"
    assert len(eu_np_list) == M, f"eu_np_list should have {M} elements"
    assert isinstance(probs_gptm, np.ndarray), "probs_gptm should be a numpy array"
    assert len(probs_gptm) == M, f"probs_gptm should have {M} elements"
    assert isinstance(u_choi, np.ndarray), "u_choi should be a numpy array"
    assert u_choi.shape == (16, 16), "Choi matrix for 2 qubits should be 16x16"
    assert isinstance(u_choi_opt, np.ndarray), "u_choi_opt should be a numpy array"
    assert u_choi_opt.shape == (
        16,
        16,
    ), "Optimal Choi matrix for 2 qubits should be 16x16"
    assert np.allclose(np.sum(probs_gptm), 1.0), "Probabilities should sum to 1"
    assert np.all(probs_gptm >= 0), "All probabilities should be non-negative"


def test_mixed_synthesis_sequential_deterministic():
    """Test that mixed_synthesis_sequential returns consistent results with same seed"""
    num_qubits = 1
    eps = 1e-3
    M = 16
    unitary = random_su(num_qubits)

    results = []
    for _ in range(3):
        result = mixed_synthesis_sequential(unitary, num_qubits, eps, M, seed=123)
        assert result is not None
        results.append(result)

    # Check that all results are identical
    for i in range(1, len(results)):
        circuit_list1, eu_np_list1, probs_gptm1, u_choi1, u_choi_opt1 = results[0]
        circuit_list2, eu_np_list2, probs_gptm2, u_choi2, u_choi_opt2 = results[i]

        assert len(circuit_list1) == len(circuit_list2)
        assert np.allclose(
            probs_gptm1, probs_gptm2
        ), "Probabilities should be identical"
        assert np.allclose(
            u_choi_opt1, u_choi_opt2
        ), "Choi matrices should be identical"


def test_mixed_synthesis_parallel_deterministic():
    """Test that mixed_synthesis_parallel returns consistent results with same seed"""
    num_qubits = 1
    eps = 1e-3
    M = 16
    unitary = random_su(num_qubits)

    results = []
    for _ in range(3):
        result = mixed_synthesis_parallel(unitary, num_qubits, eps, M, seed=123)
        assert result is not None
        results.append(result)

    # Check that all results are identical
    for i in range(1, len(results)):
        circuit_list1, eu_np_list1, probs_gptm1, u_choi1, u_choi_opt1 = results[0]
        circuit_list2, eu_np_list2, probs_gptm2, u_choi2, u_choi_opt2 = results[i]

        assert len(circuit_list1) == len(circuit_list2)
        assert np.allclose(
            probs_gptm1, probs_gptm2
        ), "Probabilities should be identical"
        assert np.allclose(
            u_choi_opt1, u_choi_opt2
        ), "Choi matrices should be identical"


def test_mixed_synthesis_sequential_one_qubit():
    """Test mixed_synthesis_sequential with one qubit"""
    num_qubits = 1
    eps = 1e-3
    M = 16
    unitary = random_su(num_qubits)

    result = mixed_synthesis_sequential(unitary, num_qubits, eps, M, seed=123)

    assert result is not None
    circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt = result

    assert len(circuit_list) == M
    assert len(eu_np_list) == M
    assert len(probs_gptm) == M
    assert u_choi_opt.shape == (4, 4), "Choi matrix for 1 qubit should be 4x4"


def test_mixed_synthesis_parallel_one_qubit():
    """Test mixed_synthesis_parallel with one qubit"""
    num_qubits = 1
    eps = 1e-3
    M = 16
    unitary = random_su(num_qubits)

    result = mixed_synthesis_parallel(unitary, num_qubits, eps, M, seed=123)

    assert result is not None
    circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt = result

    assert len(circuit_list) == M
    assert len(eu_np_list) == M
    assert len(probs_gptm) == M
    assert u_choi_opt.shape == (4, 4), "Choi matrix for 1 qubit should be 4x4"


def test_mixed_synthesis_sequential_with_numpy_array():
    """Test mixed_synthesis_sequential accepts numpy array input"""
    num_qubits = 1
    eps = 1e-3
    M = 16
    unitary_mpmath = random_su(num_qubits)
    unitary_np = np.array(unitary_mpmath.tolist(), dtype=complex)

    result = mixed_synthesis_sequential(unitary_np, num_qubits, eps, M, seed=123)

    assert result is not None
    circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt = result

    assert len(circuit_list) == M
    assert len(probs_gptm) == M


def test_mixed_synthesis_parallel_with_numpy_array():
    """Test mixed_synthesis_parallel accepts numpy array input"""
    num_qubits = 1
    eps = 1e-3
    M = 16
    unitary_mpmath = random_su(num_qubits)
    unitary_np = np.array(unitary_mpmath.tolist(), dtype=complex)

    result = mixed_synthesis_parallel(unitary_np, num_qubits, eps, M, seed=123)

    assert result is not None
    circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt = result

    assert len(circuit_list) == M
    assert len(probs_gptm) == M


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    print("All tests passed successfully!")
