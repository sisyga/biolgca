# Model zoo: published LGCA models you can rerun

Status: agreed 2026-09-26 (roadmap 4.4, first part); entry 1 built (see
"Entry 1 as built"). The course pack follows the zoo and gets its own
section once the zoo is built.

## Goal

A catalogue of LGCA models from the group's papers (and three unpublished
models), each written with the current rules so that it can be read, run,
changed and explored:

- the model in the notation of its paper, mapped onto a `ModelSpec`;
- one result of the paper reproduced at a size that runs in about a minute,
  and at the paper's size with `full=True`;
- what differs from the paper, stated;
- sliders on the paper's parameters (`lgca.explore`);
- a test that checks the reproduced result statistically.

The zoo is for reproducing and extending published work; `lgca.examples`
stays the catalogue of single mechanisms for teaching.

## Structure

```
lgca/zoo/
    __init__.py        catalogue(), ZooEntry, load(name)
    _card.py           ZooEntry: ExampleInfo + citation fields
    allee_effect.py    one module per entry
    ...
docs/source/zoo/
    index.rst          table of entries
    allee_effect.ipynb one notebook per entry
tests/zoo_test.py      builds, runs and checks every entry
```

### One entry

A module defines:

- `CARD = ZooEntry(...)`: name, title, authors, year, journal, DOI, the
  figure or result reproduced, the biological question, mechanisms, lattice,
  model family, and `fidelity`: `"same rules"` or `"simplified: ..."`, or
  `"new model"` for the unpublished entries.
- `PARAMETERS`: the paper's symbols with their meaning, value and spec path,
  e.g. `{"kappa": ("switch intensity κ", 4.4, "dynamics.operators[go_or_rest].kappa")}`.
  The notebook shows it as a table; `lgca.explore` and `vary` take the
  paths.
- `build_spec(full=False, seed=None, **parameters)`: the model with the
  paper's values as defaults, written as an explicit list of generic rules
  in the order of the paper. Where a paper's rule is not a built-in, the
  module defines it with `@interaction` or `@reorientation_term`, visibly,
  so the entry doubles as an example of extending the library.
- Measurement functions used by the notebook and the test, e.g.
  `extinct(result)`.

The research models of `lgca.research_models` stay as they are (legacy
names); zoo entries do not call them but write the rules out, because the
point is to read the model. A test checks that an entry and the matching
research model agree in distribution where both exist.

### One notebook

Sections: the question; the model (rules and the parameter table); the
spec; the reproduced result (figure next to what the paper reports); what
differs from the paper; explore it (`lgca.explore` with the key
parameters); things to try; reference. Executed by the docs build (target
under 60 s each, like the tutorials), saved without outputs, with a Colab
badge.

### Catalogue and docs

`lgca.zoo.catalogue()` returns the cards; `lgca.zoo.load(name)` the module.
The docs get a "Model zoo" page with a table (question, mechanisms,
lattice, fidelity, reference) linked from the home page and the gallery.

## Entries

### 1. Allee effect (Böttger et al. 2015, PLoS Comput Biol 11: e1004366)

Go-or-grow on a square lattice, classical with volume exclusion: b = 4
velocity and 4 rest channels (K = 8), periodic, 100 × 100 nodes. Per step:
death (r_d = 0.01), proliferation of resting cells into free rest channels
(r_b = 0.2), switch moving → resting with
r_s(ϱ) = ½(1 + tanh(κ(ϱ − θ))), ϱ = n/K (resting → moving with 1 − r_s),
random walk of moving cells, propagation.

Reproduced: the supplementary movies S1/S2 (κ = 1.1, θ = 0.375: the same
initial disc of radius 10, one rest and one velocity cell per node, dies
out in one run and grows in another), and, since the paper has no single
figure for it, the extinction frequency against the initial radius over
seeds (the content of Fig 3a) at κ = 4.4, θ = 0.75, plus the per-capita
growth rate F(ϱ)/ϱ of the mean-field model (Fig 5) next to the measured
one. Built from `birth_death`, `go_or_rest` and `random_walk`.

### 2. Genotypic and phenotypic heterogeneity (Syga, Nava-Sedeño, Deutsch 2026, EPJ ST)

Identity-based without volume exclusion (evo-LGCA); the trait is the
proliferation rate α. Per step: death (δ = 0.01), division with
α(1 − n/K), every daughter mutates (p_μ = 1) with α′ ~ N(α, σ²), σ² = 10⁻⁴;
then all cells are redistributed over the channels, velocity channels with
weight 1 and the rest channel with e^γ, γ set so that D = 0.14. Square
lattice 500 × 10, reflecting; cells start at the left edge with α₀ = 0.2 at
density Ψ₀ = 1 − δ/α₀. K and the run length are in the supplement, which we
do not have; chosen so that the front crosses most of the lattice.

Reproduced: Fig 3: (a) kymograph of the density projected on x, (b) front
position (first x with density below 0.1 Ψ₀) against the Fisher-KPP
predictions v = 2√(D(ᾱ − δ)) with ᾱ = α₀, the population mean, and the mean
of the fastest 10% (best), (c) kymograph of the mean α, (d) mean α over
time with one standard deviation. Checked: the front accelerates, and the
mean α is highest at the front.

### 3. Evolution of phenotypic plasticity (Syga et al. 2024, PLoS Comput Biol 20: e1012003)

Identity-based, no volume exclusion, carrying capacity K; one rest channel.
Per step: death (δ); switch to the proliferative (rest) phenotype with
r_κ(ρ_N) = ½(1 + tanh(κ(ρ_N − θ))), ρ_N the cells of the node and its
neighbours over (b + 1)K; division of resting cells with α(1 − n/K),
daughters rest and inherit κ ~ N(κ_mother, Δκ²); random walk of migrating
cells. κ of the initial cells uniform in [−4, 4].

Reproduced: Fig 3 in 2D, i.e. S1–S3 Figs: hexagonal lattice (paper: side
250, K = 50 cells in the centre, 300 steps, α = 1, K = 50, Δκ = 0.2) for
the three regimes (θ, δ) = (0.5, 0), (0.2, 0.2), (0.9, 0.2): total density,
migrating and resting cells and the mean κ per node, and the κ
distribution in the core and at the rim. Checked: regime 2 has κ > 0 in
the core and κ < 0 at the rim; regime 3 has κ < 0 throughout.

### 4. Clonal evolution of the go-or-grow switch (new model, `go_or_grow_glioblastoma`)

Go-or-grow in which a daughter mutates with probability r_m: it founds a
clone, its birth rate is multiplied by a fitness factor and its κ takes a
normal step. Proposed question: where do fitter clones arise and win, at
the rim or in the core, and does κ evolve along with the birth rate? 2D
hexagonal colony from a small seed; clone map, mean κ and birth rate per
node, and a Muller plot of the clones (`lgca.plots.muller_plot`). Compare
with κ fixed (no switch evolution) to separate the two traits.

### 5. Clonal expansion under crowding (new model, `evo_steric`)

Near-neutral turnover (r_d = 0.98 r_b) with driver mutations that raise the
birth rate; cells avoid crowded neighbours (steric repulsion α) and prefer
to rest (γ). Proposed question: does pushing into free space change which
clones survive? Colony from a small seed with and without steric
repulsion: clone sectors at the rim, number of surviving clones and the
Muller plot over time.

### 6. Drivers and passengers (new model from the thesis, `birthdeath_cancerdfe`)

Logistic growth; daughters acquire rare drivers (probability p_d, mean
effect s_d, exponential) and frequent passengers (p_p, s_p) that raise and
lower the birth rate. Proposed question: when does the load of weak
deleterious passengers outweigh rare strong drivers? Mean birth rate and
population over time for several p_p, the distribution of birth rates, and
the outcome (adaptation or decline) over a grid of (p_d, p_p) with seeds.

### 7. Discrete excitable media (Syga, Nava-Sedeño, Brusch, Deutsch 2019, in The Frontiers Collection, pp. 253–264)

Classical, hexagonal lattice, absorbing boundaries; activators X in the 6
velocity channels, inhibitors Y in a = K − 6 rest channels. Per step:
R_Y once (birth with P⁺ = ρ_X, death with P⁻ = ρ_Y), R_X N times (birth with
ρ_X²(1 + (ρ_Y + B)/A), death with ρ_X(ρ_Y + B)/A + ρ_X³), mixing of X over the
velocity channels, propagation. This is `excitable_medium` (A = alpha,
B = beta). Mean field: the Barkley model ∂ρ_X = D∇²ρ_X + N f,
∂ρ_Y = g, f = ρ_X(1 − ρ_X)(ρ_X − (ρ_Y + B)/A), g = ρ_X − ρ_Y, D = 1/4 in lattice
units.

Reproduced: Fig 2 with A = 0.75, B = 0.02, N = 50, K = 23, starting from
four quadrants (ρ_X, ρ_Y) = (0, 0), (1, 0), (0, 1), (1, 1): (a) the LGCA
spiral, red ∝ ρ_X and green ∝ ρ_Y, early and late (break-up into smaller
spirals); (b) the Barkley model from the same start, solved with two
`pde` operators and a `@reaction`; (c) the nullclines of f and g with the
direction of ρ_X; (d) a histogram of the LGCA node states (ρ_X, ρ_Y) after
a transient with the PDE orbit of one node on top. Things to try: the mean
return time against A (Fig 3a).

### 8. Jamming transitions in invasion (Ilina et al. 2020, Nat Cell Biol 22: 1103–1115)

From the model definition in the supplement repository
(github.com/sisyga/jamminglgca, `definition.pdf`, and the code). Classical
hexagonal LGCA with volume exclusion, b = 6 and a = 3 rest channels,
reflecting boundaries, 50 × 50 nodes. One Boltzmann reorientation with
E = E_steric + E_ECM + E_aggregation + E_alignment:

- E_steric = −β_steric j(s′) · Σ c_i ρ̃(r + c_i), ρ̃ = (n − ρ_0)/(K − ρ_0) above
  the homeostatic density ρ_0 = 3, else 0 (β_steric = 5);
- E_ECM = ρ_ECM × (resting cells of s′);
- E_aggregation = β j(s′) · ∇u, u = n_nb (1 − n_nb/n_crit)⁺ / (2 n_crit),
  n_nb the cells of the neighbours, n_crit = (b + 1) ρ_0;
- E_alignment = β (j(s′) · g_flux / (2b) + n_rest(s′) n_rest(nb) / (b ρ_0));

then the ECM is degraded, ρ_ECM ← ρ_ECM (1 − α n/K) with α = 1, and cells
flow in at the rows y ∈ {0, 1}: every free channel is filled with
probability r_b = 0.05. Start: two rows at density ρ_0; 50 steps of
transient, then 200 observed. Observables: cumulative number of single
cells (no other cell in the neighbourhood) and the mean next-neighbour
velocity correlation.

Reproduced: Fig 5d, the four regimes (low and high adhesion β × low and
high ECM density: flux vectors coloured by the local velocity correlation,
single cells marked), and a coarse version of the phase diagram Fig 5e
(β and ρ_ECM on a small log grid; the paper's 51 × 51 grid with
`full=True`). Beyond the paper (user decision 2026-09-26): invasion in 2D
from a spheroid, a disc of cells at ρ_0 in the centre of the lattice that
keeps supplying cells (the influx acts on the disc's nodes, as the
`spheroid` mask of the original code does), instead of the pseudo-1D sheet
moving up from one edge.

New pieces written in the module: the four energy terms as
`@reorientation_term`s (reading the `ecm` field), ECM degradation as a
small field rule, and the influx at the bottom rows as a `birth_death`
rule.

## Order of work

1. `lgca.zoo` package, card, catalogue, docs page, test layout, and entry 1
   as the template.
2. Entry 3, then 8 (the most new pieces), then 4–6, then 2 and 7 once the
   papers are here.

## Decisions

Decided 2026-09-26: the entries and figures above; entries 4–6 as
proposed; entry 8 follows the definition's adhesion potential
u = n_nb (1 − n_nb/n_crit)⁺ / (2 n_crit) (the original code has four times
that; noted in the notebook) and adds the 2D spheroid; the papers of
entries 2 and 7 are in `docs/development/papers/` (ignored by git, not
distributed).

## Entry 1 as built

- Order R1–R4 as in the paper: `go_or_grow.growth` (death, then division
  of resting cells into free rest channels), `go_or_rest`, `random_walk`
  over the velocity channels. The go-or-grow of `get_lgca` switches first.
- Initial condition: uniform random density ϱ₀ over the whole lattice (the
  "averaged cell density" of Fig 3). Defaults 50 × 50, 1000 steps;
  `full=True` 100 × 100, 5000.
- The movies S1/S2 (κ = 1.1, θ = 0.375) cannot be reproduced: with Eq (1)
  as printed r_s r_b > r_d at every density, and four runs from the movies'
  disc on 100 × 100 all grew to the capacity. The notebook says so and uses
  Fig 3's parameters (κ = 4.4, θ = 0.75) for "same start, two fates".
- Found: the simulated threshold is ϱ₀ ≈ 0.25 (12 seeds per density: all
  decline at ≤ 0.24, all grow at ≥ 0.28, bimodal at 0.26: 3 of 12 extinct,
  9 at capacity 0.86 after 4000 steps), far below the mean-field 0.415.
  Averaging the per-capita rate over binomially occupied nodes gives 0.240
  (`per_capita_growth_nodes`): the threshold is set by fluctuations of the
  node density. The notebook explains this.
- Runtime of the notebook: about 85 s (a 4000-step sweep of 12 seeds and a
  1000-step sweep of 8 densities × 12 seeds, both with 4 processes).

## Entry 3 as built

- `phenotypic_plasticity`: death (`birth_death`), switch (`go_or_rest`
  with the trait `kappa` and the neighbourhood density), division
  (`go_or_grow.growth`, r_b scaled by 1 − n/K without volume exclusion,
  mutation of κ in every daughter), random walk. `regime(1|2|3)` sets
  (θ, δ). `geometry="lin"` gives the paper's 1D runs (1001 nodes, K = 100,
  1000 steps, 3–8 s each), so Fig 3 A–F is reproduced at the paper's size
  as well; 2D defaults 120 × 120, 200 steps (S1–S3 Figs: 250, 300 with
  `full=True`).
- Checked (two seeds, 1D and 2D, also at the small sizes of the tests):
  regime 1 κ ≈ −0.5 everywhere, 61–69 % migrating; regime 2 mean κ of the
  inner half of the cells +2.5 (2D) / +5 (1D), of the outermost tenth −2.4
  / −0.4 to −1.2 (the 1D front is a few dozen nodes of κ ≈ −3 to −7);
  regime 3 κ < 0 throughout, −2.6 core, −0.9 rim (2D).
- Needed on the way: `lgca.plot_data.mean_trait` and "mean <trait>" views
  and trait measures in `lgca.explore`.
- Notebook runtime about 45 s.

## Entry 8 as built

- `jamming`: influx (`jamming.influx`, birth_death), one Boltzmann
  reorientation with `jamming.pressure` (flux, β_steric), `jamming.confinement`
  (rest, the `ecm` field) and `jamming.adhesion` (channels: aggregation and
  both alignment parts, so that one β scales all of adhesion, as in the
  paper), then `jamming.degradation` (a rule of the new kind `"field"`); the
  order of the original code. Neighbour sums see no cells beyond the walls.
- Needed on the way: rule kind `"field"` and `LatticeState.set_field`
  (fields keep edge-padded ghosts, as when the model is built).
- Adhesion potential: the definition's (user decision), with
  `adhesion_scale=4` for the code's. Measured (sheet, ρ̄_ECM = 0.2, two
  seeds, single cells over 200 steps / correlation): definition β = 0.2,
  1, 3, 5, 10: 2847/0.16, 2216/0.18, 1480/0.26, 1262/0.35, 520/0.54; code:
  2414/0.17, 1432/0.20, 161/0.31, 16/0.36, 1/0.49. The paper's region 2
  (no release) spans most of 0 ≤ β ≤ 10, as with the code's potential;
  flagged to the user. The notebook uses β = 10 as strong adhesion.
- One run takes about 0.8 s (sheet) and 1.3 s (spheroid, 80 × 80); the
  paper's 51 × 51 × 5 grid (`FULL_GRID`) about three hours on one core. The
  notebook runs a 6 × 6 log grid with two seeds, about 30 s in all.
