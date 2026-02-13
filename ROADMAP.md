# biolgca Roadmap — Kio's Development Plan 🦀

**Goal:** Make biolgca a production-ready, pip-installable research library comparable to CompuCell3D, HAL, Morpheus.

**Approach:** TDD — tests first, then implementation. Small PRs, one concern each.

## Phase 1: Foundation (packaging, CI, test health)
- [ ] **1.1** Create `pyproject.toml` with proper metadata, deps, entry points
- [ ] **1.2** GitHub Actions CI: run pytest on push/PR (Python 3.10, 3.11, 3.12)
- [ ] **1.3** Audit & fix existing tests — get full green suite on `aidevelop`
- [ ] **1.4** Review & merge open PR #85 (multi-species) if tests pass

## Phase 2: Code Quality
- [ ] **2.1** Add type hints to public API (`get_lgca`, base classes)
- [ ] **2.2** Docstrings for all public methods (NumPy style)
- [ ] **2.3** Sphinx docs: auto-generate API reference
- [ ] **2.4** Linting: ruff config in pyproject.toml

## Phase 3: Usability
- [ ] **3.1** Better error messages (Issues #12, #29)
- [ ] **3.2** Save/Load LGCA state (pickle + custom JSON)
- [ ] **3.3** Parameter scans (Issue #2)
- [ ] **3.4** Callback in `timeevo()` (Issue #23)

## Phase 4: Features
- [ ] **4.1** Ensemble averages (Issue #25)
- [ ] **4.2** Custom interaction API (`set_custom_interaction()`)
- [ ] **4.3** Plot improvements (Issues #5, #11, #26, #27, #28)
- [ ] **4.4** More initial conditions (Issue #20)

## Unmerged Codex work to evaluate
- `codex/plan-multi-species-lgca-implementation` → PR #85, adds multi-species LGCA
- `codex/profile-timeevo-method-using-cprofile` → massive refactor, needs careful review

---
Last updated: 2026-02-13
