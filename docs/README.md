# Building the BioLGCA documentation

The documentation source is in [`source/`](source/). It uses Sphinx and
MyST-NB; the six notebooks in [`source/tutorials/`](source/tutorials/) are
executed from clean kernels during every build.

From the repository root:

```bash
python -m pip install -e ".[docs]"
python docs/build.py
```

The build writes HTML to `docs/_build/html`. It removes stale autosummary and
HTML output first, treats Sphinx warnings as errors, and fails on notebook cell
exceptions. Keep maintained notebook sources free of saved outputs and
execution counts; the build supplies rendered outputs.

Run the documentation contracts independently with:

```bash
python -m pytest -q tests/tutorial_notebooks_test.py tests/example_gallery_docs_test.py
```

Historical notebooks live in [`../notebooks/`](../notebooks/) and are not part
of the documentation execution path.
