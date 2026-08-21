# BioLGCA performance baselines

Run the end-to-end benchmark matrix from the repository root:

```powershell
conda run -n biolgca python profiling.py --repeats 3 --output benchmarks/results.json
```

Use repeated `--scenario NAME` options to select a subset. Results include the
seed, lattice dimensions, step count, state family, interaction, Python/NumPy/
BioLGCA versions, median wall time, and peak Python-tracked memory. Observers
are disabled so the default baseline measures simulation dynamics rather than
trajectory retention.

These are diagnostic baselines, not pass/fail timing tests. Hardware, BLAS,
Python, and NumPy changes can move absolute timings. Compare changes on the
same machine and environment, and preserve scientific invariant/parity tests
before accepting an optimization. The default pytest suite only smoke-tests
the harness and its output schema.

The matrix covers classical volume exclusion, NoVE, identity-based VE,
identity-based NoVE, and multispecies states. It also includes propagation
scenarios for 1-D, square/hex, cubic, and Moore geometries where meaningful.
