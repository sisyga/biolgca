Types of LGCA
=============

The simulator implements four flavours of lattice--gas cellular automata.
Each variant differs in the way particles occupy lattice channels and
whether individual particles carry their own properties.  The sections
below summarise the characteristics of each type.

Here we detail the specificities of each LGCA type and the corresponding data structure including figures.


Classical LGCA
--------------

Only one particle may reside in each velocity channel and particles do not carry individual properties. This variant corresponds to the traditional volume--exclusion LGCA used to model populations of identical cells.


LGCA without volume exclusion
-----------------------------

Channels can hold arbitrarily many particles. All particles share the same
properties and collisions are ignored.

Identity-based LGCA
-------------------

Only one particle fits into a channel, but every particle stores its own state. These individual properties allow the simulation of heterogeneous cell populations.


Identity-based LGCA without volume exclusion
--------------------------------------------

The most general model removes the volume--exclusion constraint while keeping individual particle properties. Many cells may occupy the same node and still be distinguished throughout the simulation.
