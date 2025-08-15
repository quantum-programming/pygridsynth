"""
Test for examples/*.py execution
"""

import os

import pytest

from .test_utils.examples_utils import get_examples_directory, run_standalone_file


def test_all_example_files_execute():
    """Test that all example files execute successfully"""
    examples_dir = get_examples_directory()
    if not os.path.exists(examples_dir):
        return

    for file_name in os.listdir(examples_dir):
        if not file_name.endswith(".py"):
            continue

        path = os.path.join(examples_dir, file_name)
        returncode, stdout, stderr = run_standalone_file(path)
        assert returncode == 0, f"Example file {file_name} failed with error: {stderr}"
        print(f"✓ {file_name} executed successfully with output: {stdout.strip()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    print("All example execution tests passed!")
