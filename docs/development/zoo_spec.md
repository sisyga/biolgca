# Model zoo: published LGCA models you can rerun

Status: draft, 2026-09-26 (roadmap 4.4, first part). The course pack follows
the zoo and gets its own section once the zoo is built.

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

Fig 3. The paper is not open to me (Springer asks for a login and there is
no preprint); the abstract describes a growing population whose inherited,
mutating genotype sets the proliferation rate, travelling-wave invasion
with the fastest cells at the leading edge, and a mean-field prediction.
Needs the PDF to specify.

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

Fig 2. The chapter is not open to me; the rule exists as
`excitable_medium` (Barkley-type: activators in velocity channels,
inhibitors in rest channels, N fast activator reactions per step). Needs
the chapter or a description of Fig 2 to specify; likely candidates are
spiral waves from a broken wave front and a travelling pulse.

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
`full=True`). In higher dimensions: the same model on a cubic lattice (b = 6,
invasion along z from a sheet), where the paper has no counterpart.

New pieces written in the module: the four energy terms as
`@reorientation_term`s (reading the `ecm` field), ECM degradation as a
small field rule, and the influx at the bottom rows as a `birth_death`
rule.

## Order of work

1. `lgca.zoo` package, card, catalogue, docs page, test layout, and entry 1
   as the template.
2. Entry 3, then 8 (the most new pieces), then 4–6, then 2 and 7 once the
   papers are here.

## Open questions

1. Entry 2 and 7: the PDFs (or which figure panels to reproduce).
2. Entry 8: the definition and the code agree except for the scale of the
   adhesion potential: the definition has u = n_nb (1 − n_nb/n_crit)⁺ / (2 n_crit),
   the code `nbs (1 − nbs/n_crit)⁺ / n_crit · 2`, four times larger (the
   pressure strength is called γ in the code, β_steric in the definition).
   Follow the code that made the figure, and note the factor?
3. Entry 8, "higher dims": a 3D cubic version, as assumed here?
4. Entries 4–6: the proposed questions and figures.
