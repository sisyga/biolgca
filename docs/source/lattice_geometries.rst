Lattice geometries
==================

The LGCA can be simulated on several lattice types.  The geometry determines
the neighbourhood of each node and therefore how particles propagate.  The
following sections outline the supported geometries and how to interpret the
coordinates used by the simulator.

1-dimensional lattice
---------------------

Nodes are arranged on a line and indexed by a single coordinate ``x``.  Each
node has two opposing velocity channels so particles can move left or right
in every timestep.

2-dimensional square lattice
----------------------------

Nodes form a rectangular grid indexed by ``(x, y)`` coordinates.  Four velocity
channels point to the orthogonal neighbours and an optional resting channel may
be present.

2-dimensional hexagonal lattice
-------------------------------

Nodes are arranged on a hexagonal tiling.  Six velocity channels point to the
adjacent hexagons which gives rise to isotropic propagation on the plane.

3-dimensional cubic lattice
---------------------------

Nodes occupy points on a regular cube grid indexed by ``(x, y, z)``.  Each node
has six velocity channels pointing to the neighbouring cubes along the cardinal
directions.  A resting channel may also be present.  Propagation moves particles
to adjacent cubes in three dimensions.

