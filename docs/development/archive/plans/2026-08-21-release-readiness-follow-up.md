# Release-readiness Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the final local release-readiness gaps, verify the resulting tree from a clean state, and publish local `aidevelop` to `origin/aidevelop`.

**Architecture:** Keep the cleanup outside the runtime package. A small Python entry point owns removal of generated Sphinx directories and invokes strict Sphinx consistently on Windows, Linux, local development, and CI. Documentation and legacy notebooks are corrected in place, while an `Unreleased` changelog records the accumulated hardening work without asserting an unchosen release version.

**Tech Stack:** Python standard library, pytest, Sphinx, nbformat-compatible notebook JSON, GitHub Actions, setuptools/build.

**Spec:** `docs/superpowers/plans/2026-08-21-aidevelop-review-and-hardening.md` (Task 13 and the final audit follow-up)

## Global Constraints

- Preserve package version `0.1.0` until a release version is deliberately selected.
- Add no runtime dependency and no notebook-execution dependency.
- Keep generated autosummary and HTML output untracked.
- Use `conda run -n biolgca` for local verification.
- Push only after the merged `aidevelop` tree passes the release gate.

---

### Task 1: Reproducible strict documentation build

**Files:**
- Create: `docs/build.py`
- Create: `tests/docs_build_test.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/source/getting_started.rst`

**Interfaces:**
- Consumes: repository paths and the active Python interpreter.
- Produces: `clean_generated(source_dir, output_dir)` and a `python docs/build.py` command that cleans stale generated directories before invoking `python -m sphinx -W -b html`.

- [ ] Write a failing pytest that creates stale `_autosummary` and HTML files under temporary directories, calls `clean_generated`, and expects both directories to be absent.
- [ ] Run `conda run -n biolgca python -m pytest -q tests/docs_build_test.py` and confirm the module/function is missing.
- [ ] Implement the standard-library build helper and make the focused test pass.
- [ ] Replace all maintained developer/CI Sphinx commands with `python docs/build.py`.
- [ ] Seed a stale autosummary page and run `conda run -n biolgca python docs/build.py`; confirm cleanup occurs and strict Sphinx succeeds.
- [ ] Commit the focused change.

### Task 2: Correct maintained and legacy guidance

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/source/getting_started.rst`
- Modify: `docs/source/examples.rst`
- Modify: `docs/source/tutorial.rst`
- Modify: `BioLGCA.ipynb`
- Create: `CHANGELOG.md`

**Interfaces:**
- Consumes: current factory defaults, manual node-mutation contract, and the completed hardening milestones.
- Produces: correct examples, clearly labeled legacy notebooks, and concise unreleased change notes.

- [ ] Move `update_dynamic_fields()` after all manual node mutation in maintained and legacy examples.
- [ ] Correct the notebook claim that `recordN=True` is the default.
- [ ] Label repository notebooks as legacy exploratory material and direct new work to the tested ModelSpec gallery.
- [ ] Add an `Unreleased` changelog summarizing user-facing additions, fixes, and compatibility notes without changing the version.
- [ ] Parse all modified notebooks as JSON and run the curated teaching/example tests.
- [ ] Commit the documentation correction.

### Task 3: Release gate, integration, and publication

**Files:**
- Verify all changed files and generated artifacts; no additional production files expected.

**Interfaces:**
- Consumes: Tasks 1-2 and local `aidevelop`.
- Produces: a clean, verified `origin/aidevelop` at the same commit as local `aidevelop`.

- [ ] Run `conda run -n biolgca python -m pytest -q`.
- [ ] Run `conda run -n biolgca python docs/build.py` twice to prove stale-output independence.
- [ ] Run `conda run -n biolgca python -m build --no-isolation` and smoke-test the wheel/CLI outside the source tree.
- [ ] Review `git diff --check`, tracked files, and the complete branch diff.
- [ ] Merge the feature branch into local `aidevelop` and repeat tests/docs on the merged tree.
- [ ] Push `aidevelop` to `origin` without force and verify local, remote-tracking, and GitHub SHAs match.
