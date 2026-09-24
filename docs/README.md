# Building the BioLGCA documentation

The documentation source is in [`source/`](source/). It uses Sphinx and
MyST-NB; the six notebooks in [`source/tutorials/`](source/tutorials/) are
executed from clean kernels during every build.

From the repository root (`uv sync` installs the documentation tools as part of
the default `dev` group):

```bash
uv sync
uv run python docs/build.py
```

The build writes HTML to `docs/_build/html`. It removes stale autosummary and
HTML output first, treats Sphinx warnings as errors, and fails on notebook cell
exceptions. Keep maintained notebook sources free of saved outputs and
execution counts; the build supplies rendered outputs.

Run the documentation contracts independently with:

```bash
uv run pytest -q tests/docs_build_test.py tests/docs_snippets_test.py tests/readme_test.py tests/examples_gallery_test.py
```

Historical notebooks live in [`../notebooks/`](../notebooks/) and are not part
of the documentation execution path.
