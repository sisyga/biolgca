"""Build the Sphinx documentation from a clean generated state."""

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


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    source_dir = repository_root / "docs" / "source"
    output_dir = repository_root / "docs" / "_build" / "html"
    clean_generated(source_dir, output_dir)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "-b",
            "html",
            str(source_dir),
            str(output_dir),
        ],
        cwd=repository_root,
        check=True,
    )


if __name__ == "__main__":
    main()
