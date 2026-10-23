# Contributing to BioLGCA

Thank you for your interest in BioLGCA. Questions, bug reports, ideas for
models and code are all welcome. Everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).

## Asking questions and reporting problems

Open an issue on the [issue tracker](https://github.com/sisyga/biolgca/issues)
and pick the template that fits: a question about using BioLGCA, a bug
report or a feature request. For a bug, include the BioLGCA version
(`biolgca --version`), your Python version and operating system, and a short
script or model file that shows the problem, with its seed.

## Setting up a development environment

BioLGCA uses [uv](https://docs.astral.sh/uv/). From a fork or clone of the
repository:

```bash
git clone https://github.com/<you>/biolgca.git
cd biolgca
uv sync          # creates .venv/ from uv.lock, with the test and docs tools
```

Development happens on the `aidevelop` branch. `master` holds the released
version, from which Read the Docs builds the documentation, and `v1` holds the
code base before 2.0. Branch off `aidevelop` and open pull requests against it.

## Checks

Run from the repository root:

```bash
uv run pytest -q tests/<file>_test.py      # the tests of your change, while you work
uv run pytest -q                           # the full suite (a few minutes)
uv run pytest -q --run-slow                # also the slow paper reproductions of the zoo
uv run ruff check <changed files>          # lint
uv run python docs/build.py --no-notebooks # strict docs build without running notebooks
uv run python docs/build.py                # full docs build, runs the tutorials
```

CI runs the tests, the slow ones included, on Linux with Python 3.11 to 3.14,
on Windows and macOS with Python 3.13, and on Python 3.11 with the minimum
dependency versions. It also runs the suite once in random order, executes the
notebooks and builds the documentation.

After changing dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock`.

## Writing tests

- Every change comes with tests. A bug fix comes with a test that fails
  without the fix.
- Test what a user relies on, not how a function is written: conservation of
  cell numbers, symmetries, agreement with an analytic expectation, or with
  the frozen implementations of the code before 2.0 in `tests/legacy/`
  (do not change them).
- Tests use fixed seeds. Statistical tests compare with a tolerance derived
  from the standard error of the estimate, so that they pass for the right
  model and fail for a wrong one.
- Parametrize over geometries and model families where the behaviour should
  be the same.
- `ContractWarning` is an error in every test; test a warning with
  `pytest.warns`.
- Tests must not depend on their order: `tests/conftest.py` removes the rules
  a test registers. After changing global state, run
  `uv run --with pytest-randomly==5.0.0 pytest -q -p randomly`.

## Proposing a new interaction

A new rule is a function of the lattice state registered with
`@interaction` or `@reorientation_term` (see the
[custom interaction guide](docs/source/how_to/custom_interactions.rst)). You
can use and share such a rule from your own module without changing
BioLGCA. To propose it as a built-in:

1. Open a feature request describing the biological mechanism, the rule and
   a reference, if it comes from a paper.
2. Add it to `lgca/builtin_rules.py`, using only the operations of the state
   and `state.rng` for random numbers.
3. Test it with `lgca.testing.check_interaction`, which runs it on every
   lattice and model family it declares, and with a test of its expected
   effect (e.g. the growth rate or the drift it produces).
4. Document it in its docstring (NumPy style) and in
   `docs/source/concepts/interactions.rst`.

## Style and documentation

- PEP 8 with lines up to about 110 characters; `ruff check` must not report
  new findings. Match the surrounding code; comments say why, not what.
- NumPy-style docstrings for public functions and classes.
- Code in the README and in the how-to pages `custom_interactions`,
  `research_models`, `fields` and `exploring` is executed by
  `tests/readme_test.py` and `tests/docs_snippets_test.py`: keep it runnable
  (a new code block changes the block count in the test).
- Tutorials are committed without outputs; zoo and example notebooks keep
  their outputs, regenerated with `uv run python docs/execute_notebooks.py`.
- Describe user-facing changes in the pull request; the maintainer adds them
  to [CHANGELOG.md](CHANGELOG.md).

## Commits and pull requests

Commit messages start with `fix:`, `feat:`, `docs:`, `test:` or `chore:` and
a one-line summary, followed by a body that says what changed and why. A pull
request has a title of the form `[Fix|Feat|Docs] <one-line summary>` and a
body that describes the change, has a "Testing Done" section, and references
the issues it addresses.

If you used an AI coding assistant, say so in the pull request (and, if you
like, as a `Co-Authored-By` line in the commit). You are responsible for the
change as if you had written it yourself: read and understand all of it, and
make sure its tests check the behaviour, not just the code.

[AGENTS.md](AGENTS.md) records the design rules of the package in detail,
for human contributors and for coding agents alike.
