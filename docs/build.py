"""Build the Sphinx documentation from a clean generated state.

By default every tutorial and zoo notebook is executed (about 15 minutes), as
in CI: run this before pushing. ``--no-notebooks`` renders the notebooks
without running them, which checks docstrings, pages and cross-references in
under a minute: run this before a commit that changes documentation.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def clean_generated(source_dir: Path, output_dir: Path) -> None:
    """Remove generated autosummary sources and rendered documentation."""
    for generated_dir in (
        source_dir / "_autosummary",
        source_dir / "reference" / "_autosummary",
        output_dir.parent,
    ):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-notebooks", action="store_true",
                        help="render the tutorial and zoo notebooks without executing them")
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    source_dir = repository_root / "docs" / "source"
    output_dir = repository_root / "docs" / "_build" / "html"
    clean_generated(source_dir, output_dir)
    command = [sys.executable, "-m", "sphinx", "-W", "-b", "html"]
    if args.no_notebooks:
        command += ["-D", "nb_execution_mode=off"]
    subprocess.run(
        [*command, str(source_dir), str(output_dir)],
        cwd=repository_root,
        check=True,
    )


if __name__ == "__main__":
    main()
