# pygridsynth

`pygridsynth` is a Python library for approximating arbitrary Z-rotations using the Clifford+T gate set, based on a given angle `θ` and tolerance `ε`. It is particularly useful for addressing approximate gate synthesis problems in quantum computing and algorithm research.

## Features

- **Inspired by Established Work:** This library is based on P. Selinger's gridsynth program ([newsynth](https://www.mathstat.dal.ca/~selinger/newsynth/)), adapted for Python with additional functionality.
- **High Precision:** Utilizes the `mpmath` library to support high-precision calculations.
- **Customizable:** Allows adjustment of calculation precision (`dps`) and verbosity of output.
- **Graph Visualization:** Provides an option to visualize decomposition results as a graph.

## Installation

You can install `pygridsynth` via pip:

```bash
pip install pygridsynth
```

Or, to install from source:

```bash
pip install git+https://github.com/quantum-programming/pygridsynth.git
```

### Install executable

You can optionally install an executable script called `pygridsynth` with

```bash
pip install .
```

Or, to install in development (editable) mode

```bash
pip install -e .
```

Once you have installed this executable, you can use `pygridsynth` like this

```sh
shell> pygridsynth <theta> <epsilon> [options]
```

or `pygridsynth --help` for brief information on calling the script.

## Usage

`pygridsynth` can be used as a command-line tool even if you have not installed the
executable.

### Command-Line Example

```bash
pygridsynth <theta> <epsilon> [options]
```

### Arguments

- `theta` (required): The rotation angle to decompose, specified in radians (e.g., `0.5`).
- `epsilon` (required): The allowable error tolerance (e.g., `1e-10`).

### Options

- `--dps`: Sets the working precision of the calculation. If not specified, the working precision will be calculated from `epsilon`.
- `--dtimeout`, `-dt`: Maximum milliseconds allowed for a single Diophantine equation solving.
- `--ftimeout`, `-ft`: Maximum milliseconds allowed for a single integer factoring attempt.
- `--dloop`, `-dl`: Maximum number of failed integer factoring attempts allowed during Diophantine equation solving (default: `10`).
- `--floop`, `-fl`: Maximum number of failed integer factoring attempts allowed during the factoring process (default: `10`).
- `--seed`: Random seed for deterministic results.(default: `0`)
- `--verbose`, `-v`: Verbosity level (0=silent, 1=basic, 2=detailed, 3=debug).(default: `0`)
- `--time`, `-t`: Measures the execution time.
- `--showgraph`, `-g`: Displays the decomposition result as a graph.
- `--up-to-phase`, `-ph`: Approximates up to a phase.

### Example Execution

```bash
pygridsynth 0.5 1e-50 --verbose --time
```

This command will:
1. Compute the Clifford+T gate decomposition of a Z-rotation gate $R_Z(\theta) = e^{- i \frac{\theta}{2} Z}$ with $\theta = 0.5$ and $\epsilon = 10^{-50}$.
2. Display detailed output and measure the execution time.

## Using as a Library

You can also use `pygridsynth` directly in your scripts:

The recommended way is to use **`mpmath.mpf`** to ensure high-precision calculations:
<!-- mpf -->
```python
import mpmath

from pygridsynth.gridsynth import gridsynth_gates

mpmath.mp.dps = 128
theta = mpmath.mpf("0.5")
epsilon = mpmath.mpf("1e-10")
gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)

mpmath.mp.dps = 128
theta = mpmath.mp.pi / 8
epsilon = mpmath.mpf("1e-10")
gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)
```
You may also pass strings, which are automatically converted to `mpmath.mpf`.
<!-- str -->
```python
from pygridsynth.gridsynth import gridsynth_gates

theta = "0.5"
epsilon = "1e-10"
gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)
```

It is also possible to use **floats**, but beware that floating-point numbers may introduce precision errors.
Floats should only be used when **high precision is not required**:
<!-- float -->
```python
import numpy as np

from pygridsynth.gridsynth import gridsynth_gates

# Float can be used if high precision is not required
theta = 0.5
epsilon = 1e-10
gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)

theta = float(np.pi / 8)
epsilon = 1e-10
gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)
```

### Multi-Qubit Unitary Approximation

`pygridsynth` provides functionality for approximating multi-qubit unitary matrices using the Clifford+T gate set. This is useful for synthesizing quantum circuits that implement arbitrary multi-qubit unitaries.

**Basic usage:**
<!-- multi_qubit_basic -->
```python
import mpmath

from pygridsynth.multi_qubit_unitary_approximation import (
    approximate_multi_qubit_unitary,
)

# Define a target unitary matrix (example: 2-qubit identity)
num_qubits = 2
U = mpmath.eye(2**num_qubits)  # 4x4 identity matrix
epsilon = "1e-10"

# Approximate the unitary
circuit, U_approx = approximate_multi_qubit_unitary(U, num_qubits, epsilon)

print(f"Circuit length: {len(circuit)}")
print(f"Circuit: {str(circuit)}")
```

**Using with random unitary:**
<!-- multi_qubit_random -->
```python
from pygridsynth.multi_qubit_unitary_approximation import (
    approximate_multi_qubit_unitary,
)
from pygridsynth.mymath import random_su

# Generate a random SU(2^n) unitary
num_qubits = 2
U = random_su(num_qubits)
epsilon = "1e-10"

# Approximate with high precision
circuit, U_approx = approximate_multi_qubit_unitary(U, num_qubits, epsilon)
```

**Returning DOmegaMatrix:**
<!-- multi_qubit_domega -->
```python
from pygridsynth.multi_qubit_unitary_approximation import (
    approximate_multi_qubit_unitary,
)
from pygridsynth.mymath import random_su

# Generate a random SU(2^n) unitary
num_qubits = 2
U = random_su(num_qubits)
epsilon = "1e-10"

# Return DOmegaMatrix instead of mpmath.matrix for more efficient representation
circuit, U_domega = approximate_multi_qubit_unitary(
    U, num_qubits, epsilon, return_domega_matrix=True
)

# Convert to complex matrix if needed
U_complex = U_domega.to_complex_matrix
```

**Parameters:**

- `U`: Target unitary matrix (`mpmath.matrix`)
- `num_qubits`: Number of qubits
- `epsilon`: Error tolerance (can be `str`, `float`, or `mpmath.mpf`)
- `return_domega_matrix`: If `True`, returns `DOmegaMatrix`; if `False`, returns `mpmath.matrix` (default: `False`)
- `scale_epsilon`: Whether to scale epsilon based on the number of qubits (default: `True`)
- `cfg`: Optional `GridsynthConfig` object for advanced configuration
- `**kwargs`: Additional configuration options (ignored if `cfg` is provided)

**Returns:**

A tuple of `(circuit, U_approx)`:
- `circuit`: `QuantumCircuit` object representing the Clifford+T decomposition
- `U_approx`: Approximated unitary matrix (`mpmath.matrix` or `DOmegaMatrix` depending on `return_domega_matrix`)

### Mixed Unitary Synthesis

`pygridsynth` also provides functionality for mixed unitary synthesis, which approximates a target unitary by mixing multiple perturbed unitaries. This is useful for reducing the number of T-gates in quantum circuits.

The library provides two versions: `mixed_synthesis_parallel` (for parallel execution) and `mixed_synthesis_sequential` (for sequential execution).

**Basic usage with mpmath.matrix:**
<!-- mixed_synthesis_sequential -->
```python
from pygridsynth.mixed_synthesis import mixed_synthesis_sequential
from pygridsynth.mymath import diamond_norm_error_from_choi, random_su

# Generate a random SU(2^n) unitary matrix
num_qubits = 2
unitary = random_su(num_qubits)

# Parameters
eps = 1e-4  # Error tolerance
M = 64  # Number of Hermitian operators for perturbation
seed = 123  # Random seed for reproducibility

# Compute mixed synthesis (sequential version)
result = mixed_synthesis_sequential(unitary, num_qubits, eps, M, seed=seed)

if result is not None:
    circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt = result
    print(f"Number of circuits: {len(circuit_list)}")
    print(f"Mixing probabilities: {probs_gptm}")
    error = diamond_norm_error_from_choi(u_choi, u_choi_opt, eps, mixed_synthesis=True)
    print(f"error: {error}")
```

**Using parallel version:**
<!-- mixed_synthesis_parallel -->
```python
import mpmath

from pygridsynth.mixed_synthesis import mixed_synthesis_parallel

# Generate a random SU(2^n) unitary matrix
num_qubits = 2
unitary = mpmath.eye(2**num_qubits)

# Parameters
eps = 1e-4  # Error tolerance
M = 64  # Number of Hermitian operators for perturbation
seed = 123  # Random seed for reproducibility

# For faster computation with multiple cores
result = mixed_synthesis_parallel(unitary, num_qubits, eps, M, seed=seed)
```

**Parameters:**

- `unitary`: Target unitary matrix (`mpmath.matrix` or `numpy.ndarray`)
- `num_qubits`: Number of qubits
- `eps`: Error tolerance parameter
- `M`: Number of Hermitian operators for perturbation
- `seed`: Random seed for reproducibility (default: `123`)
- `dps`: Decimal precision (default: `-1` for auto-calculation)

**Returns:**

A tuple of `(circuit_list, eu_np_list, probs_gptm, u_choi_opt)` or `None` on failure:
- `circuit_list`: List of `QuantumCircuit` objects for perturbed unitaries
- `eu_np_list`: List of approximated unitary matrices (numpy arrays)
- `probs_gptm`: Array of mixing probabilities
- `u_choi_opt`: Optimal mixed Choi matrix

**Note:** The parallel version (`mixed_synthesis_parallel`) uses multiprocessing and may be faster for large `M` values, while the sequential version (`mixed_synthesis_sequential`) is more suitable for debugging or when parallel execution is not desired.


## Contributing

Bug reports and feature requests are welcome. Please submit them via the [GitHub repository](https://github.com/quantum-programming/pygridsynth) Issues section. Contributions must comply with the MIT License.

## License

This project is licensed under the MIT License.

## References

- Vadym Kliuchnikov, Kristin Lauter, Romy Minko, Adam Paetznick, Christophe Petit. "Shorter quantum circuits via single-qubit gate approximation." Quantum 7 (2023): 1208. DOI: 10.22331/q-2023-12-18-1208.
- Peter Selinger. "Efficient Clifford+T approximation of single-qubit operators." Quantum Info. Comput. 15, no. 1-2 (2015): 159-180.
- Neil J. Ross and Peter Selinger. "Optimal ancilla-free Clifford+T approximation of z-rotations." Quantum Info. Comput. 16, no. 11-12 (2016): 901-953.
- Vadym Kliuchnikov, Dmitri Maslov, and Michele Mosca. "Fast and efficient exact synthesis of single-qubit unitaries generated by Clifford and T gates." Quantum Info. Comput. 13, no. 7-8 (2013): 607-630.
- Ken Matsumoto and Kazuyuki Amano. "Representation of Quantum Circuits with Clifford and π/8 Gates." arXiv: Quantum Physics (2008). URL: https://api.semanticscholar.org/CorpusID:17327793.
- Brett Gordon Giles and Peter Selinger. "Remarks on Matsumoto and Amano's normal form for single-qubit Clifford+T operators." ArXiv abs/1312.6584 (2013). URL: https://api.semanticscholar.org/CorpusID:10077777.
- Vivek V. Shende, Igor L. Markov, and Stephen S. Bullock. "Minimal universal two-qubit controlled-NOT-based circuits." Phys. Rev. A 69, no. 6 (2004): 062321. DOI: 10.1103/PhysRevA.69.062321.
- Vivek V. Shende, Igor L. Markov, and Stephen S. Bullock. "Finding Small Two-Qubit Circuits." Proceedings of SPIE - The International Society for Optical Engineering (2004). DOI: 10.1117/12.542381.
- Vivek V. Shende, Igor L. Markov, and Stephen S. Bullock. "Smaller Two-Qubit Circuits for Quantum Communication and Computation." In Proceedings of the Conference on Design, Automation and Test in Europe - Volume 2, p. 20980. IEEE Computer Society, 2004.
- Korbinian Kottmann. "Two-qubit Synthesis." (2025). URL: https://pennylane.ai/compilation/two-qubit-synthesis.
- Anna M. Krol and Zaid Al-Ars. "Beyond quantum Shannon decomposition: Circuit construction for n-qubit gates based on block-ZXZ decomposition." Physical Review Applied 22, no. 3 (2024): 034019. DOI: 10.1103/PhysRevApplied.22.034019.
- Peter Selinger and Neil J. Ross. "Exact and approximate synthesis of quantum circuits." (2018). URL: https://www.mathstat.dal.ca/~selinger/newsynth/.
