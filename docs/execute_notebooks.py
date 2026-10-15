"""Execute the zoo and example notebooks and save their outputs.

The docs build renders these notebooks with the outputs saved in them (they run for minutes: a sweep of
many seeds, a spheroid that grows for a week); it executes only notebooks without outputs, the
tutorials. Run this after changing a saved notebook or the code it runs, and commit the result:

    uv run python docs/execute_notebooks.py                                  # every zoo and example notebook
    uv run python docs/execute_notebooks.py docs/source/zoo/jamming.ipynb    # one notebook

CI executes every notebook from a clean kernel, so a saved notebook that no longer runs fails there.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent / "source"
SAVED = ("zoo", "examples")


def execute(path: Path) -> float:
    import nbformat
    from nbclient import NotebookClient

    start = time.perf_counter()
    notebook = nbformat.read(path, as_version=4)
    # run where the docs build runs it; no timing metadata, so that a rerun changes only the outputs
    NotebookClient(notebook, timeout=1800, kernel_name="python3", record_timing=False,
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(notebook, path)
    return time.perf_counter() - start


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="*", type=Path,
                        help="notebooks to execute (default: every notebook in docs/source/zoo and examples)")
    parser.add_argument("--jobs", type=int, default=2, help="notebooks executed at the same time (default 2)")
    args = parser.parse_args(argv)
    notebooks = [path.resolve() for path in args.notebooks] or sorted(
        path for folder in SAVED for path in (SOURCE_DIR / folder).glob("*.ipynb"))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for path, seconds in zip(notebooks, pool.map(execute, notebooks)):
            print(f"{path.relative_to(SOURCE_DIR)}: {seconds:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
