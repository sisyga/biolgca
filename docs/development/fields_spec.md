# Fields: reaction–advection–diffusion equations coupled to the cells

Status: design agreed (2026-09-26). Phases 1 and 2 implemented
(2026-09-26, see "Phase 1 as built" and "Phase 2 as built").

## Goal

Multiscale models in which cells on the LGCA lattice and continuous fields
(oxygen, nutrients, growth factors, chemoattractants, drugs) act on each
other:

- a field obeys a reaction–advection–diffusion equation on the lattice's
  nodes, with sources and sinks that may be the cells (secretion, uptake);
- the cells respond to the field through the existing cues: `chemotaxis`
  and the `field` and `gradient` cues of switching probabilities, and,
  new here, birth and death rates;
- a field is updated by an operator of the interaction pipeline, so its
  place among the cell operators is part of the model, like any other
  operator: `cells → field → cells` if listed that way;
- a quasi-steady solver covers the common case in which the field relaxes
  much faster than the cells change (reaction and diffusion on the scale of
  seconds to minutes, cell division on the scale of hours), as in
  evolutionary models of tumours under oxygen limitation.

"Basic support" means one scalar equation per operator, the terms below, and
the solvers below. Coupled systems solved together (Turing patterns with
implicit coupling), fields on a finer grid than the lattice, and
heterogeneous diffusion are later extensions (see "Later").

## The equation

For a field `c` on the nodes of the lattice:

```
∂c/∂t = D Δc − ∇·(v c) + P(x) − L(x) c
```

- `D`: diffusion coefficient (number).
- `v`: advection velocity, a constant vector or the name of a vector field
  of shape `dims + (d,)`.
- `P(x) ≥ 0`: production per node and time step: a constant, a field (e.g. a
  map of vessels), and secretion by cells.
- `L(x) ≥ 0`: loss rate: decay and uptake by cells, possibly saturating.

Every reaction is written as a production and a loss rate. This is the form
students meet first (source minus first-order sink), and treating `L c`
implicitly keeps `c ≥ 0` in every solver without clipping. Michaelis–Menten
uptake `μ n c / (K_m + c)` fits it as `L = μ n / (K_m + c)` (evaluated at the
previous iterate; see "Solvers").

### Units

Parameters are in lattice units per LGCA time step: `D` in nodes² per step,
rates per step, `v` in nodes per step. A field that relaxes fast relative to
the cells therefore has large `D` and loss rates; no separate time-scale
parameter is needed. The documentation gives the conversion from physical
units, `D_lattice = D τ / ε²` for node spacing `ε` and step duration `τ`,
with a worked example (oxygen in tissue: `D ≈ 2·10³ µm²/s`, `ε = 20 µm`,
`τ = 1 h` gives `D_lattice ≈ 2·10⁴`, far beyond what explicit time stepping
can afford, which is why the quasi-steady solver matters).

### The Laplacian on every lattice

`Δc(x) ≈ w Σ_i (c(x + c_i) − c(x))` with `w = 2d / Σ_i |c_i|²`, summed over
the velocity channels `c_i` of the lattice, i.e. over the same
neighbourhood the cells use. The stencil is exact for quadratic functions,
checked on all five lattices (2026-09-25, interior nodes of `|x|²`, and on
the 3D Moore lattice also with a mixed term `x y`):

| Lattice | d | neighbours b | w | Δ of `|x|²` (exact: 2d) |
|---|---|---|---|---|
| 1D | 1 | 2 | 1 | 2.000000 |
| square | 2 | 4 | 1 | 4.000000 |
| hexagonal | 2 | 6 | 2/3 | 4.000000 |
| cubic | 3 | 6 | 1 | 6.000000 |
| 3D Moore | 3 | 26 | 1/9 | 6.000000 |

The operator is assembled once per model as a sparse matrix from the
geometry's neighbour relations (`channel_weight`/`nb_sum` applied to node
indices), so no geometry needs its own PDE code. Advection uses first-order
upwinding along the same channel directions (`v · c_i` decides which
neighbour is upwind), which keeps `c ≥ 0`; its numerical diffusion is
documented.

### Boundaries

`boundary` of the operator:

- `"periodic"`: required when the lattice is periodic, invalid otherwise.
- `"no_flux"` (default on non-periodic lattices): zero normal gradient.
- `{"value": c_b}`: fixed value beyond the edge (Dirichlet), e.g. oxygen
  supplied by the surrounding tissue.
- per side on 1D, square and cubic lattices, e.g.
  `{"x-": {"value": 1.0}, "x+": {"value": 0.0}, "default": "no_flux"}` for a
  linear gradient across the domain, the classic chemotaxis set-up.

The field's boundary is independent of the cells' boundary (reflecting or
absorbing walls for cells say nothing about the chemistry), except that a
periodic lattice needs a periodic field. After each update the ghost nodes
of the stored field are filled consistently (wrapped, mirrored, or the
fixed value), so `state.gradient("oxygen")` and the `gradient` cue are
correct at the edges. Today `_attach_fields` pads static fields by edge
values, which is the no-flux convention; that stays for static fields.

## Model description

### Fields keep their place

`StateSpec.fields` keeps its meaning: named arrays on the lattice, now with
a number allowed as a uniform initial value. A field becomes dynamic when a
field operator updates it; fields that no operator updates stay static, as
today. The field lives where it lives now (an attribute of the LGCA object,
with ghost nodes), so every existing reader works unchanged: `chemotaxis`,
`contact_guidance`, the `field` and `gradient` cues, `state.field(...)`,
the pipeline graph (which already draws `field:` nodes and their readers).

### The operator

A fourth operator kind, `"field"`, next to `birth_death`,
`phenotype_switch` and `reorientation`. The built-in operator `pde` updates
one field. Its whole equation is in the operator, like the parameters of
every other operator:

```json
{"name": "pde", "parameters": {
    "field": "oxygen",
    "diffusion": 2e4,
    "decay": 0.0,
    "cells": [{"uptake": 0.5, "saturation": 0.05}],
    "boundary": {"value": 1.0},
    "solver": "steady"
}}
```

In Python, `PDESpec` (like `BirthDeathSpec`) with the same fields:

```python
from lgca.fields import PDESpec

oxygen = PDESpec(field="oxygen", diffusion=2e4,
                 cells=[{"uptake": 0.5, "saturation": 0.05}],
                 boundary={"value": 1.0}, solver="steady")
```

Parameters:

| Parameter | Meaning | Default |
|---|---|---|
| `field` | name of a field in `StateSpec.fields` | required |
| `diffusion` | `D` | 0 |
| `decay` | first-order loss rate everywhere | 0 |
| `production` | number or field name: `P` independent of cells | 0 |
| `cells` | list of cell terms (below) | `[]` |
| `reactions` | list of registered reactions (below) | `[]` |
| `advection` | vector of length `d` or name of a vector field | none |
| `boundary` | see "Boundaries" | `"no_flux"` or `"periodic"` |
| `solver` | `"explicit"`, `"implicit"`, `"steady"` | `"implicit"` |
| `solver_options` | tolerances, substeps, backend | automatic |

Cell terms, each a mapping:

- `{"production": rate}`: every cell adds `rate` per step (secretion).
- `{"uptake": rate}`: loss rate `rate · n` (linear uptake).
- `{"uptake": rate, "saturation": K, "n": 1}`: saturating uptake
  `rate · n_cells · cⁿ / (Kⁿ + cⁿ)`, a Hill function of the field (`n = 1`,
  the default, is Michaelis–Menten; `n ≥ 1`). As a loss rate:
  `L = rate · n_cells · cⁿ⁻¹ / (Kⁿ + cⁿ)`.
- optional `"species"` (index or list; default all), `"channels"`
  (`"all"`, `"rest"`, `"velocity"`; e.g. only resting cells consume), as in
  the other rules.
- in identity-based models `rate` may name a cell trait, so each cell
  secretes or consumes at its own rate (summed per node), and the rate can
  evolve through mutations: consumption as an evolving trait under oxygen
  limitation.

The field sees the cells as the previous operator left them, the rule that
already holds for cues. Propagation happens after all operators, so a `pde`
listed last sees the cells before they move and the next step's first
operators see the updated field.

### Reactions of your own

```python
from lgca.fields import reaction

@reaction
def activation(state, c, rate=1.0, other="inhibitor"):
    """Production by the activator, lost in proportion to the inhibitor."""
    u = state.field(other)
    return rate * c**2 / (1 + c**2), 0.1 * u   # production, loss rate
```

A reaction is a function of the lattice state and the field's current
values returning `(production, loss_rate)`, both non-negative with one
value per node. It is registered like `switch_cue` and used as
`"reactions": [{"name": "activation", "rate": 2.0}]`. Nonlinear terms are
iterated (see "Solvers"). Several fields that react with each other are
updated one after another in the order of their operators (operator
splitting), which is first-order accurate in time; the documentation says
so.

### Cells that respond to fields

Existing: `chemotaxis` (`"field": name`), `contact_guidance`, the `field`
and `gradient` cues in switching probabilities, `trait_switch` and mutation
events.

New: `birth_rate` and `death_rate` of `birth_death` accept the switching
probability forms of `lgca.switching` in addition to numbers and trait
names:

```json
{"name": "birth_death", "parameters": {
    "birth_rate": {"max": 0.1, "cues": [{"name": "field", "field": "oxygen", "kappa": 20, "theta": 0.2}]},
    "death_rate": {"max": 0.05, "cues": [{"name": "field", "field": "oxygen", "kappa": -30, "theta": 0.05}]}
}}
```

This is useful beyond fields (density-dependent death, for example) and
reuses the parser, trait-valued `kappa`/`theta`, and the Boltzmann form. It
changes no existing result: numbers and trait names mean what they meant.

New as well: a Hill form of probabilities (decided 2026-09-26). It is part
of `lgca.switching`, so every probability that accepts the tanh form accepts
it too: `phenotype_switch` rates, `trait_switch` and mutation events, the
switch of `go_or_rest` and `resting`, and `birth_rate`/`death_rate`. In a row
of `phenotype_switch` rates it is a probability like the tanh form, not a
Boltzmann weight, so the existing rule against mixing the two in a row
applies.

```json
{"max": 0.1, "hill": [{"name": "field", "field": "oxygen", "K": 0.2, "n": 1}]}
```

stands for `p = max · Π_k c_kⁿᵏ / (K_kⁿᵏ + c_kⁿᵏ)` over the listed cues:
the cue value `c` gives half the maximum at `c = K`, `n` sets the steepness
(`n = 1` is Michaelis–Menten), and a negative `n` gives the decreasing
response `Kⁿ / (Kⁿ + cⁿ)` with `n` replaced by `|n|`, e.g. death under
hypoxia. Several cues multiply, as independent limiting factors. `K` and `n`
may name cell traits, as `kappa` and `theta` can. A probability uses one
form: `"cues"` (tanh), `"rate"` with `"cues"` (Boltzmann) or `"hill"`.
Cue values must be non-negative for the Hill form; the parser checks this
at run time with a message naming the cue.

### Recording and plotting

- `FieldRecorder(fields=["oxygen"], every=...)`; `result.data["oxygen"]` is
  the field history (interior nodes), `result.data.steps("oxygen")` the
  steps; the recording budget counts it; the CLI writes it to
  `measurements.npz`.
- `lgca.plot_scalarfield` exists for square, hexagonal and cubic lattices
  and accepts recorded frames; new are `animate_scalarfield` (square,
  hexagonal) and a 1D kymograph of a field history.
- `result.lgca.oxygen` holds the final field, as for static fields today.

## Solvers

One update per call of the operator, i.e. per LGCA step if listed once.

All numerical work is done by SciPy: `scipy.integrate.solve_ivp` for
explicit time stepping and `scipy.sparse.linalg` for the linear systems of
the implicit and steady solvers. BioLGCA assembles the matrix from the
lattice and chooses the SciPy routine.

Why not `solve_ivp`'s implicit methods (BDF, Radau) for everything? Measured
2026-09-26: one LGCA step (time 1) of a field with diffusion and uptake by
a disc of cells, `rtol 10⁻⁴`, the sparse Jacobian passed to `solve_ivp`;
error relative to a reference of 200 implicit substeps:

| Method, one LGCA step | 100², D=1 | 200², D=1 | 200², D=50 | error |
|---|---|---|---|---|
| backward Euler, one sparse LU | 32 ms | 273 ms | 257 ms | 2–3 % |
| `solve_ivp` BDF | 177 ms | 1758 ms | 2534 ms | 10⁻⁴ |
| `solve_ivp` Radau | 1009 ms | 5733 ms | 6180 ms | 10⁻⁴ |
| `solve_ivp` RK45 | 6 ms | 39 ms | 275 ms (866 evaluations) | 10⁻⁴ |

The legacy interface `scipy.integrate.ode` (SciPy lists it under "Old API";
`solve_ivp` is the current one) was measured as well, on the same problem
with no-flux boundaries because its Jacobians must be full or banded, not
sparse: `vode` with `bdf` was as slow as `solve_ivp` BDF (98 ms at 100²,
157 ms with D=50), and `vode` with `adams` matched RK45 at 100² (1.5 against
2.0 ms) but was slower at 200² (69 against 8 ms, D=1). `solve_ivp` LSODA,
which switches to a dense Jacobian when it detects stiffness, took 433 s for
one step at 200² with D=50: a dense Jacobian of 40 000² entries (13 GB).

The sources change every LGCA step (the cells move), so an integrator
starts afresh every step: BDF and Radau factor the matrix 5–14 times per
step, and their strength, high order over long smooth intervals, does not
pay off in intervals of one step, where the error of freezing the cells
during the field update is of first order anyway. RK45 is cheap and
accurate while the field is slow; for fast fields the steady or implicit
solver is the right tool.

### `"explicit"`

`solve_ivp` over the step, method RK45 by default, with error control
(`rtol` 10⁻⁴). `solver_options={"method": ...}` chooses among RK23, RK45 and
DOP853 (explicit) and BDF and Radau, which get the sparse Jacobian of the
linear part, for a stiff transient that must be accurate. LSODA is refused
with an explanation: SciPy gives it no sparse Jacobian, and its dense one
exhausts memory on lattices of useful size. The
number of right-hand-side evaluations per step is recorded in the run
metadata, and a warning suggests `"implicit"` or `"steady"` when it exceeds
a threshold (default 200), a sign of a field too fast for explicit steps.
RK45 does not guarantee `c ≥ 0`; values below `−atol` raise an error that
suggests the implicit solver.

### `"implicit"`

Backward Euler over the step (`substeps`, default 1):
`(I − Δt (D Δ − A_v − L)) c_new = c_old + Δt P`, solved with SciPy's sparse
solvers (below). Unconditionally stable, positive (the matrix is an
M-matrix with upwinding), first order in time: 2–3 % error in one step of
the test above, less with substeps. For fields that are fast but not at
steady state, or to follow a transient.

### `"steady"`: the quasi-steady assumption

Solves `D Δc − ∇·(v c) + P − L c = 0` at every call: the field is at
equilibrium with the current cells. The field is also solved once when the
model is built, so the first operators of the first step see an equilibrated
field.

A steady state exists only if something removes the field: decay, uptake,
or a fixed-value boundary. Without any (periodic or no-flux boundaries,
no loss), the problem is singular and the operator raises an error that
says so when the model is built.

Linear algebra. The matrix changes every step when cells take up the field
(`L` depends on the cell numbers), so the cost of a solve per step decides
whether the feature is usable. Measured 2026-09-25 on this machine, square
periodic lattice, linear uptake by a disc of cells (`D = 1`, decay `10⁻³`,
uptake `0.05` per cell), relative error against a direct solve:

| Method (per LGCA step) | 100² | 200² | 400² | error |
|---|---|---|---|---|
| new sparse LU (SuperLU) | 43 ms | 338 ms | 2050 ms | exact |
| CG, Jacobi preconditioner, previous field as start, rtol 10⁻⁶ | 15 ms | 148 ms | 497 ms | 10⁻⁷ |
| incomplete LU (SciPy `spilu`, drop 10⁻⁴) of an earlier step as preconditioner, BiCGSTAB | 9.5 ms | 198 ms | 888 ms | 10⁻⁷ |
| algebraic multigrid rebuilt every step (pyamg) | 34 ms | 94 ms | 342 ms | 10⁻⁸ |
| CG, multigrid of an earlier step as preconditioner, previous field as start | **7 ms** | **33 ms** | **128 ms** | 10⁻⁸ |
| reuse one LU (matrix independent of the cells) | 1.3 ms | 6.8 ms | 29 ms | exact |
| FFT (periodic, constant coefficients) | 0.3 ms | 0.9 ms | 2.9 ms | exact |
| *for comparison: one go-or-grow LGCA step* | *7 ms* | *24 ms* | *96 ms* | |

The reused multigrid needed 6–7 CG iterations in each of 20 consecutive
steps with 5 % of the colony's nodes changing per step. So the default
(`backend="auto"`) is:

1. matrix independent of the cells (only production by cells, or no
   cells): factor once, solve per step by back substitution;
2. otherwise, with `pyamg`: preconditioned CG (BiCGSTAB with
   advection, whose matrix is not symmetric), started from the previous
   field, with a multigrid hierarchy that is rebuilt only when the
   iteration count doubles relative to the last rebuild;
3. on Python 3.14, until pyamg publishes wheels for it (SciPy only): sparse
   LU for up to about 20 000 nodes, Jacobi-preconditioned CG/BiCGSTAB above
   (faster than incomplete LU in the table).

`backend` can be forced (`"direct"`, `"cg"`, `"amg"`); `rtol` defaults to
10⁻⁶, well below the noise of the cells.

Nonlinear terms (saturating uptake, registered reactions): Picard
iteration, i.e. evaluate `P` and `L` at the current iterate, solve the
linear problem, repeat until the relative change is below `rtol` (at most
`max_iterations`, default 20, with a warning if not converged). Starting from
the previous step's field should make a few iterations enough; to be
measured.

## Implementation outline

- `lgca/fields.py`: `PDESpec`, the `pde` operator (class-based, kind
  `"field"`, `outputs() = {"field:<name>"}`, `dependencies()` = fields read
  by its terms), assembly of the Laplacian and upwind advection from the
  geometry, boundary handling, the three solvers, `@reaction`.
- `pipeline.py`/`operator_base.py`: the `"field"` kind in schedules,
  timings and `list_plugins`. `execute_step` needs no change: it refreshes
  the cell-derived quantities only after operators whose outputs include
  `nodes`, which a field operator's do not.
- `model.py`: numbers as uniform initial values in `StateSpec.fields`;
  schema for the `pde` operator; `_attach_fields` pads with the boundary
  convention of the operator that owns the field.
- `builtin_rules.py`: probability forms for `birth_rate`/`death_rate`.
- `simulation.py`: `FieldRecorder`; `RunData` names.
- `switching.py`: the Hill form.
- `pyproject.toml`: required dependency
  `"pyamg>=5; python_version < '3.14'"`, added with the steady solver in
  phase 2 (the lowest pyamg release that works with the other minimum
  versions is determined then). pyamg 5.3.0 depends on NumPy and
  `scipy>=1.11` and has wheels for Python 3.10–3.13 on Linux, macOS and
  Windows, none yet for 3.14 (checked 2026-09-26): without the marker,
  installing on 3.14 would compile it from source and fail on computers
  without a C++ compiler. The SciPy minimum rises from 1.9.2 to 1.11. The
  CI jobs on 3.14 test the SciPy fallback; drop the marker once pyamg
  publishes 3.14 wheels.
- Docs: how-to "Fields and multiscale models"; tutorials 7 and 8;
  model-file reference; changelog; roadmap status note.

## Tests

- Laplacian exact on quadratics for every geometry (above); total field
  conserved to rounding with diffusion alone and periodic or no-flux
  boundaries, for every solver.
- Decay alone: discrete exponential of each scheme exactly.
- 1D steady state with decay and fixed values at both ends against the
  analytic `cosh` profile, with second-order convergence under refinement;
  2D point source against the FFT solution.
- `"steady"` equals the long-time limit of `"implicit"` and `"explicit"`.
- `c ≥ 0` under strong uptake for the implicit and steady solvers, an
  error for negative explicit values; the singular steady problem raises at
  build time.
- Hill form: values at `c = K`, limits, negative `n`, product of cues,
  trait-valued `K` and `n`; `birth_rate`/`death_rate` with each probability
  form give the expected mean rates (statistical tests from standard
  errors).
- Coupling: uptake removes `Σ L c` per step; species and channel selection;
  trait-valued rates in identity-based models with and without volume
  exclusion; a pipeline `cells → pde → cells` sees the intermediate state.
- Parametrized over geometry × family × species wherever behaviour should
  be the same.
- Model files: save, load and run round trips; CLI run records fields.
- Benchmark `benchmarks/fields.py`: steady solve against one LGCA step on
  100², 200², 400² (fresh process per measurement), speed-ups reported.

## Phases

1. Operator, Laplacian, boundaries, `explicit` and `implicit`, cell terms,
   `FieldRecorder`, field plots. Chemotaxis toward a secreted signal runs
   end to end.
2. `steady` with the backends above, Picard iteration, build-time solve,
   pyamg dependency, benchmark.
3. `birth_rate`/`death_rate` responding to cues, the Hill form; advection;
   `@reaction`.
4. Tutorials and how-to:
   - tutorial 7, oxygen-limited growth: a colony consuming oxygen supplied
     from the boundary forms a proliferating rim around a hypoxic core; an
     identity-based extension in which the consumption rate evolves;
   - tutorial 8, aggregation: cells secrete a chemokine and move up its
     gradient (Keller–Segel on a lattice), with and without decay, showing
     the onset of aggregation.

## Phase 1 as built (2026-09-26)

`lgca/fields.py`, `FieldRecorder` in `simulation.py`, plots in
`square_plotting.py` and `lgca_1d.py`, tests in `tests/fields_test.py`.
Choices made while implementing, beyond the text above:

- **No flux** drops the term of a neighbour beyond the edge (zero flux
  through that face). The matrix is then symmetric with zero column sums, so
  diffusion conserves the total to rounding on every lattice (tested). On 1D,
  square and cubic lattices this equals edge padding; on the hexagonal and
  Moore lattices, where a channel from an edge node can reach a ghost node
  whose edge copy is another node, it is the conservative choice. The stored
  ghost nodes of a no-flux field are edge-padded, as for static fields.
- **Inflow lattices** (reflecting in x, periodic in y): the field wraps in y;
  `boundary` sets the x sides.
- **Implicit backend** `"auto"`: a matrix independent of the cells (no
  uptake) is factored once (SuperLU); otherwise conjugate gradients with a
  Jacobi preconditioner, started from the previous field. Measured on this
  machine, square lattice, uptake by a disc of cells, rtol 10⁻⁶, median of
  5 steps, fresh process each:

  | one implicit step | 100², D=1 | 100², D=50 | 200², D=1 | 200², D=50 |
  |---|---|---|---|---|
  | SuperLU (`"direct"`) | 18 ms | 20 ms | 103 ms | 106 ms |
  | CG, Jacobi (`"cg"`, the default here) | 1.4 ms | 3.6 ms | 9.7 ms | 38 ms |

  Saturating uptake (`K = 0.1`) at 200², D=1: 11 Picard iterations,
  33 ms per step with CG against 1018 ms with SuperLU. Tiny negative values
  left by CG's rounding are set to zero.
- **Secretion only** (the matrix is factored once), square lattice, density
  0.5, median of 7 steps, fresh process each, against one random-walk step:

  | | 100² | 200² | 400² |
  |---|---|---|---|
  | random walk (`ReorientationSpec()`) | 1.7 ms | 7.7 ms | 29 ms |
  | `pde` implicit, D=1 | 0.9 ms | 4.3 ms | 22 ms |
  | `pde` explicit (RK45), D=1, 62 evaluations | 1.8 ms | 7.9 ms | 28 ms |
  | `pde` explicit (RK45), D=20, about 350 evaluations | 19 ms | 89 ms | 377 ms |

- **Defaults**: implicit `rtol` 10⁻⁶ for CG and for the Picard iteration of
  saturating uptake (`max_iterations` 20); explicit `rtol` 10⁻⁴, `atol`
  10⁻⁶, values in `[−atol, 0)` set to zero, below that an error.
  `solve_ivp` gets `t_eval=[1]`, so it keeps no intermediate states.
- **`solver="steady"`** raises "not available yet" until phase 2;
  `advection` and `reactions` are not parameters yet (phase 3), so they are
  rejected as unknown.
- **`FieldRecorder(fields, schedule=Schedule(every=...))`**, like the other
  recorders (the text above wrote `every=`). The CLI writes
  `field_<name>` and `field_<name>_steps` to `measurements.npz`, one pair per
  recorded field (user decision 2026-09-26). A field named like a recording
  (`"population"`) or recorded twice is refused before the run.
- **Statistics** per field in `metadata["fields"][name]`: `calls`, and for
  explicit steps `rhs_evaluations` and `max_rhs_evaluations`, for saturating
  uptake `max_iterations_used`.
- **Build**: `build_model` lets each field operator pad its field after the
  static fields are attached (`attach_field`); the initial values must be
  finite and non-negative. Field names must not collide with attributes of
  the model, an existing rule: `c` (the velocities) is taken.
- **Plots**: `animate_scalarfield(field_t, steps=...)` on square and
  hexagonal lattices, with one colour scale for all frames; on 1D lattices
  `plot_scalarfield` draws a history as a kymograph and a single profile as
  a line.

## Phase 2 as built (2026-09-26)

- **`solver="steady"`** solves `(decay + L − D Δ) c = P + boundary source`
  at every call and once in `build_model`, after the field is attached, so
  the first operators of the first step see it at equilibrium.
- **When no steady state exists**: at build time, no diffusion and no decay
  (a node without cells has no loss), or no decay, no uptake term and no
  fixed value; at run time, uptake is the only loss and no cell takes up the
  field. Each raises an error that says what removes the field.
- **Backends** as in "Solvers", with these details: `"amg"` is conjugate
  gradients preconditioned by a pyamg smoothed-aggregation V-cycle, started
  from the previous field. The hierarchy is rebuilt when a solve needs more
  than twice the iterations of the first solve after the last rebuild, or
  when an aged hierarchy has not converged after `4 × that + 10`
  iterations (then the solve is repeated). With birth and death changing the
  matrix every step, one hierarchy served 20 steps at up to 8 iterations.
  Without pyamg: SuperLU up to 20 000 nodes, Jacobi-preconditioned CG above.
  A constant matrix (no uptake) is factored once with `"auto"` and
  `"direct"`. Saturating uptake: Picard iteration as in the implicit
  solver (7 iterations per step in the benchmark).
- **Dependencies**: `pyamg>=5.1; python_version < '3.14'` (5.0 imports
  `pkg_resources` and fails without setuptools), `scipy>=1.11`, and
  `threadpoolctl>=3.0` (not planned; see next point). The minimum versions
  (Python 3.11, SciPy 1.11.1, NumPy 1.24, pyamg 5.1.0, threadpoolctl 3.0.0)
  pass the whole test suite.
- **BLAS threads**: NumPy and SciPy each load an OpenBLAS with one thread
  per core (16 here). The iterative solvers spend their time in dot products
  and norms of lattice size, for which waking the threads costs more than
  the arithmetic, and timings varied between runs. Measured (steady amg /
  steady cg / implicit cg, ms per step, 200²): 75 / 412 / 18 with 16
  threads, 21 / 61 / 3.6 with one. Every field update therefore runs with
  one BLAS thread (`threadpoolctl`, about 9 µs per update).
- **Benchmark** `benchmarks/fields.py`: a disc of cells (radius a quarter of
  the width) on a periodic square lattice takes up a field with D = 1,
  production and decay 10⁻³, uptake 0.05 per cell; the cells divide and die
  every step. Time of the `pde` operator per step, fresh process each,
  median of 3, against one go-or-grow step (`go_or_rest`,
  `go_or_grow.growth`, velocity random walk, with propagation):

  | ms per step | 100² | 200² | 400² |
  |---|---|---|---|
  | go-or-grow step | 4.5 | 20.4 | 76.3 |
  | steady, amg (default) | 6.4 | 20.6 | 78.5 |
  | steady, cg (Jacobi) | 14.4 | 60.8 | 319 |
  | steady, direct (SuperLU) | 38.9 | 302 | 1936 |
  | steady, amg, saturating uptake (K = 0.1) | 24.5 | 81.5 | 298 |
  | implicit, cg | 1.4 | 3.5 | 12.1 |

  The default steady solver costs about one go-or-grow step and is 6, 15
  and 25 times faster than SuperLU.

## Phase 3 as built (2026-09-26)

- **Hill form** in `lgca.switching`: a `Probability` with `hill=True` keeps
  a cue's `n` in `kappa` and its `K` in `theta` (as the Boltzmann form keeps
  `beta` in `kappa`). The response is computed as
  `log σ(n (log c − log K))`, which is `log(cⁿ / (Kⁿ + cⁿ))` for either sign
  of `n` and has no overflow; the drive of a Hill probability is the sum of
  these logarithms, so `nodes`, `cells` and `log_odds` (used by `resting`)
  work unchanged. `K` is required (no natural default), `n` defaults to 1;
  `K ≤ 0` and `n = 0` are refused when parsed, and for trait values when
  evaluated. The keys `K` and `n` are not passed to the cue function, so a
  registered cue with parameters of these names cannot be used in the Hill
  form.
- **`birth_rate`/`death_rate`**: a mapping (or a list per species with at
  least one mapping) is evaluated per node and species before the step; the
  classical code paths take per-node probabilities of shape
  `dims + (n_species,)` where they took one per species. Numbers, lists of
  numbers and trait names take the old paths, so existing results are
  unchanged (the suite's regression tests pass). Identity-based models
  evaluate the probability per cell, so trait-valued `kappa`, `theta`, `K`
  and `n` work there.
- **Graph**: `describe_model_graph` adds an edge from every field that a cue
  nested in an operator's parameters reads (`{"name": "field"|"gradient",
  "field": ...}`), e.g. `field:oxygen → birth_death`.

## Later

Systems of fields solved together (implicit coupling of reactions),
heterogeneous or field-dependent diffusion, fields on a finer grid than the
lattice, fixed values at interior nodes (vessels as Dirichlet nodes; for
now vessels are a `production` map), a helper for physical units, and the
linear stability analyses of 4.3 applied to cell–field models
(chemotactic aggregation).

## Decisions

Decided 2026-09-26:

1. The equation lives in the operator.
2. Lattice units per LGCA step; the documentation converts physical units.
3. Steady fields are solved once when the model is built.
4. pyamg is a required dependency (with the Python 3.14 marker, see
   "Implementation outline"), not an optional extra.
5. No flux is the default boundary on non-periodic lattices.
6. `birth_rate`/`death_rate` accept the probability forms of
   `lgca.switching`, and a Hill form is added to them (`n = 1` is
   Michaelis–Menten). It is available to every switching probability,
   `phenotype_switch` and `trait_switch` included; saturating uptake uses
   the same Hill function.
7. Two tutorials: oxygen-limited growth (7) and aggregation toward a
   chemokine the cells secrete (8).
8. The operator is called `pde`.
9. The solvers use SciPy: `solve_ivp` for explicit steps (not the legacy
   `ode`/`vode`, see "Solvers"), `scipy.sparse.linalg` for implicit steps
   and steady states.
