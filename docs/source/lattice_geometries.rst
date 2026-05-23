Lattice geometries
==================

The geometry determines the coordinate system, velocity channels and
propagation neighbourhood. Pass a geometry name or alias to
:func:`lgca.get_lgca`.

.. list-table::
   :header-rows: 1

   * - Geometry
     - Common aliases
     - Velocity channels
     - Coordinates
   * - 1D linear
     - ``"1d"``, ``"lin"``, ``"linear"``
     - 2
     - ``x``
   * - 2D square
     - ``"square"``, ``"sq"``, ``"rect"``, ``"rectangular"``
     - 4
     - ``(x, y)``
   * - 2D hexagonal
     - ``"hex"``, ``"hx"``, ``"hexagonal"``
     - 6
     - ``(x, y)``
   * - 3D cubic
     - ``"cubic"``, ``"cb"``
     - 6
     - ``(x, y, z)``
   * - 3D Moore
     - ``"moore"``, ``"moore3d"``
     - 26
     - ``(x, y, z)``

1-dimensional lattice
---------------------

Nodes are arranged on a line. The two velocity channels represent motion to the
left and right. Integer ``dims`` values create a lattice of that length.

2-dimensional square lattice
----------------------------

Nodes form a rectangular grid. Four velocity channels point to the orthogonal
neighbours. Optional rest channels can be appended after the velocity channels.

2-dimensional hexagonal lattice
-------------------------------

Nodes are arranged on a hexagonal tiling. Six velocity channels point to
adjacent hexagons, giving a more isotropic planar neighbourhood than the square
lattice.

3-dimensional cubic lattice
---------------------------

Nodes occupy a regular cube grid. Six velocity channels point along the
positive and negative coordinate axes.

3-dimensional Moore lattice
---------------------------

The Moore lattice extends the cubic neighbourhood to all 26 surrounding nodes:
all combinations of ``(dx, dy, dz)`` in ``{-1, 0, 1}`` except the origin.

Example
-------

.. code-block:: python

   from lgca import get_lgca

   lgca = get_lgca(geometry="moore", dims=(10, 10, 10), seed=1)
