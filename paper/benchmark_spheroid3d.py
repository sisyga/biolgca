"""Time the 3D tumor spheroid that the paper's speed sentence refers to.

Usage (from the repository root, pinned to one core on Linux):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 \
        uv run python paper/benchmark_spheroid3d.py 40 240

40 = nodes per side (40^3 = 64,000 nodes of 30 micrometers, a 1.2 mm cube),
240 = one-hour steps (ten simulated days). The model is the shipped example
`lgca.examples.tumour_spheroid` on the cubic lattice with 26 neighbors.
"""
import resource
import sys
import time

import numpy as np

from lgca.examples import tumour_spheroid
from lgca.model import run_model

n, steps = int(sys.argv[1]), int(sys.argv[2])
spec = tumour_spheroid.build_spec(geometry="moore", dims=(n, n, n), steps=steps, record_every=steps)
start = time.perf_counter()
result = run_model(spec, showprogress=False)
seconds = time.perf_counter() - start
density = np.asarray(result.data["density"])[-1]
per_class = density.reshape(-1, density.shape[-1]).sum(0)
print(f"nodes={n**3} steps={steps} time={seconds:.1f}s per_step={seconds / steps:.3f}s "
      f"cells_end={int(per_class.sum())} (proliferating, quiescent, necrotic)={per_class.astype(int).tolist()} "
      f"peak_memory={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}GB")
