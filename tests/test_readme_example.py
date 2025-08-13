"""
Test for README.md example
"""

import os
import subprocess
import sys
from typing import Tuple

import mpmath
import pytest

from pygridsynth.gridsynth import gridsynth_gates

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Constants for README example
README_EXAMPLE_CODE = """import mpmath

from pygridsynth.gridsynth import gridsynth_gates

mpmath.mp.dps = 128
theta = mpmath.mpmathify("0.5")
epsilon = mpmath.mpmathify("1e-10")

gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)
"""

README_EXAMPLE_DPS = 128
README_EXAMPLE_THETA = "0.5"
README_EXAMPLE_EPSILON = "1e-10"


def _extract_readme_example_code() -> str:
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    readme_lines = readme_content.split("\n")

    # Find the specific code block
    start_line = None
    end_line = None

    for i, line in enumerate(readme_lines):
        if line.strip() == "import mpmath":
            start_line = i
        elif start_line is not None and line.strip() == "print(gates)":
            end_line = i
            break

    if start_line is None or end_line is None:
        raise ValueError("Could not find README example code block")

    return "\n".join(readme_lines[start_line : end_line + 1]) + "\n"


def _run_standalone_file(file_path: str) -> Tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..")

    result = subprocess.run(
        [sys.executable, file_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
        env=env,
    )

    return result.returncode, result.stdout, result.stderr


def _setup_readme_example_parameters():
    mpmath.mp.dps = README_EXAMPLE_DPS
    theta = mpmath.mpmathify(README_EXAMPLE_THETA)
    epsilon = mpmath.mpmathify(README_EXAMPLE_EPSILON)
    return gridsynth_gates(theta=theta, epsilon=epsilon)


def test_readme_example_execution():
    gates = _setup_readme_example_parameters()
    assert gates is not None
    print(f"✓ README example executed successfully, result: {gates}")


def test_readme_example_content_matches():
    readme_example = _extract_readme_example_code()
    assert readme_example == README_EXAMPLE_CODE
    print("✓ README example content matches exactly")


def test_readme_example_standalone_file():
    example_file_path = os.path.join(os.path.dirname(__file__), "readme_example.py")

    # Verify file content
    with open(example_file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    assert file_content == README_EXAMPLE_CODE
    print(f"✓ Created standalone example file: {example_file_path}")

    # Test execution
    returncode, stdout, stderr = _run_standalone_file(example_file_path)
    assert returncode == 0, f"Example file execution failed with error: {stderr}"
    assert stdout.strip(), "Example file should produce output"

    print("✓ Standalone example file executed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    print("All README example tests passed!")
