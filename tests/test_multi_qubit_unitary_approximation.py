import mpmath
import pytest

from pygridsynth.config import GridsynthConfig
from pygridsynth.domega_unitary import DOmegaMatrix
from pygridsynth.multi_qubit_unitary_approximation import (
    approximate_multi_qubit_unitary,
)
from pygridsynth.mymath import all_close
from pygridsynth.quantum_circuit import QuantumCircuit


def test_approximate_one_qubit_unitary_identity():
    U = mpmath.matrix([[1, 0], [0, 1]])
    num_qubits = 1
    epsilon = "0.0001"

    circuit, approx_unitary = approximate_multi_qubit_unitary(U, num_qubits, epsilon)

    assert isinstance(circuit, QuantumCircuit)
    assert isinstance(approx_unitary, mpmath.matrix)
    assert len(circuit) == 0
    assert all_close(approx_unitary, U, tol=float(epsilon))


def test_approximate_two_qubit_unitary():
    U = mpmath.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    num_qubits = 2
    epsilon = "0.0001"

    circuit, approx_unitary = approximate_multi_qubit_unitary(U, num_qubits, epsilon)

    assert isinstance(circuit, QuantumCircuit)
    assert isinstance(approx_unitary, mpmath.matrix)
    assert len(circuit) > 0
    assert all_close(approx_unitary, U, tol=float(epsilon))


def test_approximate_three_qubit_unitary_identity():
    num_qubits = 3
    n = 2**num_qubits
    U = mpmath.eye(n)
    epsilon = "0.0001"

    circuit, approx_unitary = approximate_multi_qubit_unitary(
        U, num_qubits, epsilon, return_domega_matrix=True
    )

    assert isinstance(circuit, QuantumCircuit)
    assert isinstance(approx_unitary, DOmegaMatrix)
    assert all_close(approx_unitary.to_complex_matrix, U, tol=float(epsilon))


def test_approximate_multi_qubit_unitary_return_matrix_type():
    U = mpmath.matrix([[1, 0], [0, 1]])
    num_qubits = 1
    epsilon = "0.0001"

    circuit, result_matrix = approximate_multi_qubit_unitary(
        U, num_qubits, epsilon, return_domega_matrix=False
    )
    assert isinstance(circuit, QuantumCircuit)
    assert isinstance(result_matrix, mpmath.matrix)

    circuit, result_domega = approximate_multi_qubit_unitary(
        U, num_qubits, epsilon, return_domega_matrix=True
    )
    assert isinstance(circuit, QuantumCircuit)
    assert isinstance(result_domega, DOmegaMatrix)


def test_approximate_multi_qubit_unitary_config_handling():
    U = mpmath.matrix([[1, 0], [0, 1]])
    num_qubits = 1
    epsilon = "0.0001"

    custom_cfg = GridsynthConfig(dps=50, measure_time=True)
    with pytest.warns(
        UserWarning, match="When 'cfg' is provided, 'kwargs' are ignored."
    ):
        approximate_multi_qubit_unitary(
            U, num_qubits, epsilon, cfg=custom_cfg, verbose=100
        )

    approximate_multi_qubit_unitary(U, num_qubits, epsilon, verbose=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    print("All tests passed successfully!")
