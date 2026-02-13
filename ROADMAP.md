# biolgca Roadmap — Kio's Development Plan 🦀

**Goal:** Make biolgca a production-ready, pip-installable research library comparable to CompuCell3D, HAL, Morpheus.

**Approach:** TDD — tests first, then implementation. Small PRs, one concern each.

## Phase 1: Foundation (packaging, CI, test health)
- [ ] **1.1** Create `pyproject.toml` with proper metadata, deps, entry points
- [ ] **1.2** GitHub Actions CI: run pytest on push/PR (Python 3.10, 3.11, 3.12)
- [ ] **1.3** Audit & fix existing tests — get full green suite on `aidevelop`
- [ ] **1.4** Review & merge open PR #85 (multi-species) if tests pass

## Phase 2: Architecture & Code Health (Kio's findings)
- [ ] **2.1** **Eliminate wildcard imports** — all geometry files do `from lgca.base import *`. Replace with explicit imports for clarity and to avoid namespace pollution
- [ ] **2.2** **Split plotting from model** — plotting methods live inside LGCA classes (700+ LOC in base alone). Extract to a separate module/mixin pattern so LGCA objects are data-only. This is partially done (SquarePlotMixin) but inconsistent
- [ ] **2.3** **`base_extensions.py` is 2233 LOC** — contains 4 base classes crammed into one file. Split into `ib_base.py`, `nove_base.py`, `nove_ib_base.py`
- [ ] **2.4** **Interaction functions are stringly-typed** — `set_interaction()` maps string names to functions via getattr. Add an enum or registry pattern, validate early with clear errors
- [ ] **2.5** **`**kwargs` swallows typos silently** — `__init__` accepts `**kwargs` which means misspelled parameters are silently ignored. Add explicit parameters or validate kwargs
- [ ] **2.6** **No `__all__` exports** — combined with wildcard imports, the public API is unclear. Define `__all__` in every module

## Phase 3: Code Quality
- [ ] **3.1** Add type hints to public API (`get_lgca`, base classes)
- [ ] **3.2** Docstrings for all public methods (NumPy style)
- [ ] **3.3** Sphinx docs: auto-generate API reference
- [ ] **3.4** Linting: ruff config in pyproject.toml

## Phase 4: Usability
- [ ] **4.1** Better error messages (Issues #12, #29)
- [ ] **4.2** Save/Load LGCA state (pickle + custom JSON serialization)
- [ ] **4.3** Parameter scans (Issue #2) — essential for research workflows
- [ ] **4.4** Callback in `timeevo()` (Issue #23) — enables custom analysis during simulation
- [ ] **4.5** **Reproducibility** — seed handling exists but should be documented and tested: same seed → same trajectory guaranteed
- [ ] **4.6** **Progress/logging** — tqdm is used in timeevo but there's no structured logging. Add Python logging for debugging simulations

## Phase 5: Features
- [ ] **5.1** Ensemble averages (Issue #25)
- [ ] **5.2** Custom interaction API (`set_custom_interaction()`)
- [ ] **5.3** Plot improvements (Issues #5, #11, #26, #27, #28)
- [ ] **5.4** More initial conditions (Issue #20)
- [ ] **5.5** **Configuration objects** — replace long `__init__` parameter lists with a config/builder pattern (like Morpheus XML but Pythonic)

## Unmerged Codex work to evaluate
- `codex/plan-multi-species-lgca-implementation` → PR #85, adds multi-species LGCA
- `codex/profile-timeevo-method-using-cprofile` → massive refactor, needs careful review

---
Last updated: 2026-02-13
