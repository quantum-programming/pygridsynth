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

## Usage

`pygridsynth` can be used as a command-line tool.

### Command-Line Example

```bash
python -m pygridsynth <theta> <epsilon> [options]
```

### Arguments

- `theta` (required): The rotation angle to decompose, specified in radians (e.g., `0.5`).
- `epsilon` (required): The allowable error tolerance (e.g., `1e-10`).

### Options

- `--dps`: Sets the calculation precision (default: `128`).
- `--dtimeout`, `-dt`: Sets the timeout for solving diophantine equations in milliseconds (default: `200`).
- `--ftimeout`, `-ft`: Sets the timeout for factorization in milliseconds (default: `50`).
- `--verbose`, `-v`: Enables detailed output.
- `--time`, `-t`: Measures the execution time.
- `--showgraph`, `-g`: Displays the decomposition result as a graph.

### Example Execution

```bash
python -m pygridsynth 0.5 1e-50 --dps 256 --verbose --time
```

This command will:
1. Compute the Clifford+T gate decomposition of a Z-rotation gate with $\theta = 0.5$ and $\epsilon = 0.01$.
2. Set the calculation precision to 256 decimal places.
3. Display detailed output and measure the execution time.

## Using as a Library

You can also use `pygridsynth` directly in your scripts:

```python
from pygridsynth.gridsynth import gridsynth_gates
import mpmath

mpmath.mp.dps = 128
theta = mpmath.mpmathify("0.5")
epsilon = mpmath.mpmathify("1e-10")

gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)
```

## Contributing

Bug reports and feature requests are welcome. Please submit them via the [GitHub repository](https://github.com/quantum-programming/pygridsynth) Issues section. Contributions must comply with the GNU General Public License v3 or later.

## License

This project is licensed under the GNU General Public License v3 or later.

## References

- Brett Giles and Peter Selinger. Remarks on Matsumoto and Amano's normal form for single-qubit Clifford+T operators, 2019.
- Ken Matsumoto and Kazuyuki Amano. Representation of Quantum Circuits with Clifford and π/8 Gates, 2008.
- Neil J. Ross and Peter Selinger. Optimal ancilla-free Clifford+T approximation of z-rotations, 2016.
- Peter Selinger. Efficient Clifford+T approximation of single-qubit operators, 2014.
- Peter Selinger and Neil J. Ross. Exact and approximate synthesis of quantum circuits. https://www.mathstat.dal.ca/~selinger/newsynth/, 2018.
- Vadym Kliuchnikov, Dmitri Maslov, and Michele Mosca. Fast and efficient exact synthesis of single qubit unitaries generated by Clifford and T gates, 2013.