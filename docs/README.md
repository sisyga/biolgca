# Building the BioLGCA documentation

The documentation source is in [`source/`](source/). It uses Sphinx and
MyST-NB. The tutorial notebooks in [`source/tutorials/`](source/tutorials/)
are committed without outputs and executed from clean kernels in every build.
The zoo and example notebooks ([`source/zoo/`](source/zoo/),
[`source/examples/`](source/examples/)) run for minutes; they are committed
with their outputs, which the build renders without running them.

From the repository root (`uv sync` installs the documentation tools as part of
the default `dev` group):

```bash
uv sync
uv run python docs/build.py                 # runs the tutorials (a few minutes)
uv run python docs/build.py --no-notebooks  # renders all notebooks without running them (under a minute)
uv run python docs/build.py --execute-all   # also runs the zoo and example notebooks
```

The build writes HTML to `docs/_build/html`. It removes stale autosummary and
HTML output first, treats Sphinx warnings as errors, and fails on notebook cell
exceptions.

After changing the code of a zoo or example notebook, or the code it runs,
regenerate its saved outputs and commit them:

```bash
uv run python docs/execute_notebooks.py [<notebook>]
```

Run the documentation contracts independently with:

```bash
uv run pytest -q tests/docs_build_test.py tests/docs_snippets_test.py tests/readme_test.py tests/examples_gallery_test.py
```

Historical notebooks live in [`../notebooks/`](../notebooks/) and are not part
of the documentation execution path.
