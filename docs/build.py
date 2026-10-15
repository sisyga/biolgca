"""Build the Sphinx documentation from a clean generated state.

By default the tutorial notebooks are executed (about two minutes) and the zoo and
example notebooks are rendered with the outputs saved in them (refresh those with
docs/execute_notebooks.py). ``--no-notebooks`` renders every notebook without running
it, which checks docstrings, pages and cross-references in under a minute;
``--execute-all`` executes every notebook (about 25 minutes).
"""

import argparse
import os
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
                        help="render the tutorial, zoo and example notebooks without executing them")
    parser.add_argument("--execute-all", action="store_true",
                        help="execute every notebook, also those with saved outputs")
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    source_dir = repository_root / "docs" / "source"
    output_dir = repository_root / "docs" / "_build" / "html"
    clean_generated(source_dir, output_dir)
    command = [sys.executable, "-m", "sphinx", "-W", "-b", "html"]
    if args.no_notebooks:
        command += ["-D", "nb_execution_mode=off"]
    environment = dict(os.environ, BIOLGCA_DOCS_EXECUTE_ALL="1") if args.execute_all else None
    subprocess.run(
        [*command, str(source_dir), str(output_dir)],
        cwd=repository_root,
        check=True,
        env=environment,
    )


if __name__ == "__main__":
    main()
