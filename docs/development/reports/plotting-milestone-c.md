# Milestone C square-density rendering comparison

Measured on 2026-08-21 with Python 3.13.5, NumPy 2.4.6, Matplotlib's Agg
backend, and no colorbar. Each row is one seeded render; peak memory is from
`tracemalloc`. Compare the unchanged `aidevelop` checkout with the Milestone C
worktree on the same machine and environment.

| Lattice | `aidevelop` time | Milestone C time | `aidevelop` peak | Milestone C peak |
|---:|---:|---:|---:|---:|
| 20 x 20 | 0.118 s | 0.041 s | 2.27 MB | 0.44 MB |
| 40 x 40 | 0.330 s | 0.017 s | 7.68 MB | 0.33 MB |
| 256 x 256 | 14.315 s | 0.020 s | 298.37 MB | 1.47 MB |

The optimized square path creates one `AxesImage` instead of one
`RegularPolygon` per site. Hexagonal rendering continues to use collections
because its staggered geometry is not represented by a rectangular image.
Absolute timings are diagnostic rather than a CI threshold.
