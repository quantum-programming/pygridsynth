"""
Utility functions for testing examples and README consistency
"""

import os
import subprocess
import sys
from typing import Dict, Tuple


def get_examples_directory() -> str:
    """Get the path to the examples directory"""
    return os.path.join(os.path.dirname(__file__), "..", "..", "examples")


def read_example_file_content(filename: str) -> str:
    """Read the content of an example file"""
    file_path = os.path.join(get_examples_directory(), f"{filename}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Example file {filename}.py does not exist")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_readme_examples() -> Dict[str, str]:
    """Extract all Python code blocks from README.md with their identifiers"""
    readme_path = os.path.join(os.path.dirname(__file__), "..", "..", "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    examples = {}
    lines = readme_content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for comment with identifier
        if line.startswith("<!--") and line.endswith("-->"):
            # Extract identifier from comment
            identifier = line[4:-3].strip()

            # Look for the next Python code block
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```python"):
                j += 1

            if j < len(lines):
                # Found Python code block, extract the code
                code_start = j + 1
                code_end = code_start
                while code_end < len(lines) and not lines[code_end].strip().startswith(
                    "```"
                ):
                    code_end += 1

                if code_end < len(lines):
                    code = "\n".join(lines[code_start:code_end])
                    examples[identifier] = code + "\n"
                    i = code_end
                else:
                    i = j + 1
            else:
                i += 1
        else:
            i += 1

    return examples


def run_standalone_file(file_path: str) -> Tuple[int, str, str]:
    """Run a standalone Python file and return exit code, stdout, stderr"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "..")

    result = subprocess.run(
        [sys.executable, file_path],
        capture_output=True,
        text=True,
        env=env,
    )

    return result.returncode, result.stdout, result.stderr


def get_readme_path() -> str:
    """Get the path to the README.md file"""
    return os.path.join(os.path.dirname(__file__), "..", "..", "README.md")
