"""
Test that all marimo notebooks in the demos directory execute without errors.

This is a CI-oriented test: it runs each notebook through marimo's
export pipeline (which executes all cells) and checks for a clean exit code.
"""

import subprocess
import sys
from pathlib import Path

import pytest

DEMOS_DIR = Path(__file__).resolve().parent.parent / "demos"

# Collect all marimo notebooks in the demos directory.
# We skip import_fix.py because it's a helper, not a standalone notebook.
NOTEBOOKS = sorted(
    p for p in DEMOS_DIR.glob("*.py")
    if p.name != "import_fix.py"
)


def _is_marimo_notebook(path: Path) -> bool:
    """Check if a .py file looks like a marimo notebook."""
    try:
        content = path.read_text(encoding="utf-8")
        return "marimo.App" in content
    except Exception:
        return False


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes(notebook_path: Path):
    """Execute a marimo notebook via 'marimo export' and assert success."""
    if not _is_marimo_notebook(notebook_path):
        pytest.skip(f"{notebook_path.name} is not a marimo notebook")

    result = subprocess.run(
        [
            sys.executable, "-m", "marimo", "export", "html",
            str(notebook_path),
            "-o", "/dev/null",
            "--no-include-code",
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes per notebook – data downloads can be slow
        cwd=str(DEMOS_DIR.parent),  # run from project root so import_fix works
    )

    # Provide useful diagnostics on failure
    if result.returncode != 0:
        # Print stderr lines that look like Python tracebacks
        stderr = result.stderr.strip()
        stdout_tail = result.stdout.strip().splitlines()[-30:]
        msg = (
            f"Notebook {notebook_path.name} failed with exit code {result.returncode}.\n\n"
            f"--- STDERR ---\n{stderr}\n\n"
            f"--- STDOUT (last 30 lines) ---\n" + "\n".join(stdout_tail)
        )
        pytest.fail(msg)

    assert result.returncode == 0
