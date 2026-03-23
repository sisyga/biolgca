# Test Coverage Audit — biolgca (`kio/development`)

Generated: 2026-03-23 | Branch: `kio/development` | Command: `python -m pytest --cov=lgca --cov-report=term-missing -q`

## Per-Module Coverage Table

| Module | Stmts | Miss | Cover |
|---|---|---|---|
| `lgca/__init__.py` | 95 | 15 | 84% |
| `lgca/base.py` | 368 | 103 | 72% |
| `lgca/base_extensions.py` | 1034 | 669 | 35% |
| `lgca/cubic_ext.py` | 347 | 261 | **25%** |
| `lgca/ib_interactions.py` | 159 | 9 | **94%** |
| `lgca/interactions.py` | 256 | 68 | 73% |
| `lgca/lgca_1d.py` | 325 | 198 | **39%** |
| `lgca/lgca_3dmoore.py` | 108 | 30 | 72% |
| `lgca/lgca_cubic.py` | 321 | 197 | **39%** |
| `lgca/lgca_hex.py` | 159 | 45 | 72% |
| `lgca/lgca_square.py` | 148 | 46 | 69% |
| `lgca/nove_ib_interactions.py` | 222 | 208 | **6%** |
| `lgca/nove_interactions.py` | 69 | 61 | **12%** |
| `lgca/plots.py` | 303 | 284 | **6%** |
| `lgca/square_ext.py` | 320 | 238 | **26%** |
| `lgca/square_plotting.py` | 369 | 347 | **6%** |
| **TOTAL** | **4603** | **2779** | **40%** |

## Top-3 Under-Tested Modules (by absolute uncovered lines)

### 1. `lgca/base_extensions.py` — 669 uncovered lines (35%)
Largest module; houses the bulk of identity-based and NoVE extension logic including
plotting, interaction dispatch, spatial analysis helpers, and identity tracking.
Missing: lines 197–1983+ (most of the extension class methods)

### 2. `lgca/square_plotting.py` — 347 uncovered lines (6%)
All square-lattice plotting functions are completely untested.
(Matplotlib dependency is mocked in autodoc but not in tests.)

### 3. `lgca/nove_ib_interactions.py` — 208 uncovered lines (6%)
NoVE + identity-based interactions: `go_or_grow`, `birth`, `birthdeath`,
`go_or_grow_kappa_chemo`, etc. Essentially zero test coverage.

## Stub test files (TODOs)

See:
- `tests/test_base_extensions_stub.py`
- `tests/test_nove_ib_interactions_stub.py`
- `tests/test_plots_stub.py`

## Notes

- scipy was missing from the test environment (installed now: `pip install scipy --break-system-packages`)
- Full run: **152 passed, 2 skipped, 0 failed** (after scipy fix)
- plots.py and square_plotting.py failures are expected without a display; mark with `@pytest.mark.skipif` or use `plt.switch_backend('Agg')` in conftest
