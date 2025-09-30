"""
Test for README.md and examples/*.py consistency
"""

import pytest

from .test_utils.examples_utils import (
    extract_readme_examples,
    get_readme_path,
    read_example_file_content,
)


def test_readme_has_identifiers_for_all_python_blocks():
    """Test that all Python code blocks in README have identifiers"""
    with open(get_readme_path(), "r", encoding="utf-8") as f:
        readme_content = f.read()

    lines = readme_content.split("\n")
    python_blocks = []
    identifiers = []

    # Find all Python code blocks and their preceding identifiers
    for i, line in enumerate(lines):
        if line.strip().startswith("```python"):
            # Look backwards for identifier comment
            identifier_found = False
            for j in range(i - 1, max(-1, i - 5), -1):  # Look up to 4 lines back
                prev_line = lines[j].strip()
                if prev_line.startswith("<!--") and prev_line.endswith("-->"):
                    identifier = prev_line[4:-3].strip()
                    identifiers.append(identifier)
                    identifier_found = True
                    break
                elif prev_line and not prev_line.startswith("<!--"):
                    # Found non-empty, non-comment line before code block
                    break

            if not identifier_found:
                python_blocks.append(i + 1)  # 1-indexed line number

    assert (
        len(python_blocks) == 0
    ), f"Python code blocks without identifiers found at lines: {python_blocks}"
    print(f"✓ All Python code blocks have identifiers: {identifiers}")


def test_readme_examples_match_files():
    """Test that README examples match their corresponding files"""
    readme_examples = extract_readme_examples()

    for identifier, readme_code in readme_examples.items():
        try:
            file_content = read_example_file_content(identifier)
            assert (
                readme_code == file_content
            ), f"README example '{identifier}' does not match examples/{identifier}.py"
            print(f"✓ README example '{identifier}' matches examples/{identifier}.py")
        except FileNotFoundError:
            pytest.fail(
                f"Example file examples/{identifier}.py not found"
                f"for README example '{identifier}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    print("All README consistency tests passed!")
