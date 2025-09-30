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
- `--verbose`, `-v`: Enables detailed output.
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


## Contributing

Bug reports and feature requests are welcome. Please submit them via the [GitHub repository](https://github.com/quantum-programming/pygridsynth) Issues section. Contributions must comply with the MIT License.

## License

This project is licensed under the MIT License.

## References

- Brett Giles and Peter Selinger. Remarks on Matsumoto and Amano's normal form for single-qubit Clifford+T operators, 2019.
- Ken Matsumoto and Kazuyuki Amano. Representation of Quantum Circuits with Clifford and π/8 Gates, 2008.
- Neil J. Ross and Peter Selinger. Optimal ancilla-free Clifford+T approximation of z-rotations, 2016.
- Peter Selinger. Efficient Clifford+T approximation of single-qubit operators, 2014.
- Peter Selinger and Neil J. Ross. Exact and approximate synthesis of quantum circuits. https://www.mathstat.dal.ca/~selinger/newsynth/, 2018.
- Vadym Kliuchnikov, Dmitri Maslov, and Michele Mosca. Fast and efficient exact synthesis of single qubit unitaries generated by Clifford and T gates, 2013.
