"""Build the Sphinx documentation from a clean generated state."""

from pathlib import Path
import shutil
import subprocess
import sys


def clean_generated(source_dir: Path, output_dir: Path) -> None:
    """Remove generated autosummary sources and rendered documentation."""
    shutil.rmtree(source_dir / "_autosummary", ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)


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
