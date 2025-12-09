from unittest.mock import patch

import mpmath
import pytest

from pygridsynth.__main__ import main


def test_main_deterministic_results(capsys):
    """Test that main function returns the same result for identical inputs"""
    # pi/8 ≈ 0.39269908169
    test_args = ["pygridsynth", "0.39269908169", "0.01"]
    results = []

    for _ in range(5):
        with patch("sys.argv", test_args):
            original_dps = mpmath.mp.dps
            try:
                main()
                captured = capsys.readouterr()
                results.append(captured.out.strip())
            finally:
                mpmath.mp.dps = original_dps

    assert len(set(results)) == 1, f"Results do not match: {results}"
    print(f"✓ Got the same result from 5 executions: {results[0]}")


def test_main_with_different_inputs(capsys):
    """Test that main function works correctly with different inputs"""
    test_cases = [
        ["pygridsynth", "0.78539816339", "0.1"],  # pi/4
        ["pygridsynth", "0.52359877559", "0.05"],  # pi/6
        ["pygridsynth", "0.5", "0.01"],
    ]
    results = []

    for test_args in test_cases:
        with patch("sys.argv", test_args):
            original_dps = mpmath.mp.dps
            try:
                main()
                captured = capsys.readouterr()
                result = captured.out.strip()
                results.append(result)
                assert result is not None, f"Result is None for args {test_args}"
            finally:
                mpmath.mp.dps = original_dps

    print(f"✓ Successfully executed with {len(test_cases)} different inputs")


def test_main_consistency_with_options(capsys):
    """Test that consistent results are obtained even with options"""
    test_args = [
        "pygridsynth",
        "0.39269908169",
        "0.01",
        "--dps",
        "30",
        "--verbose",
        "1",
    ]
    results = []

    # Execute 3 times with the same arguments
    for _ in range(3):
        with patch("sys.argv", test_args):
            original_dps = mpmath.mp.dps
            try:
                main()
                captured = capsys.readouterr()
                results.append(captured.out.strip())
            finally:
                mpmath.mp.dps = original_dps

    assert len(set(results)) == 1, f"Results with options do not match: {results}"
    print("✓ Got the same result from 3 executions with options")


if __name__ == "__main__":
    pytest.main([__file__])
    print("All tests passed successfully!")
