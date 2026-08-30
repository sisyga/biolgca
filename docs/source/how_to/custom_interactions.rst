Custom interactions
===================

An interaction extension consists of one operator class, one
:class:`~lgca.operator_base.PluginInfo` contract, and one explicit registration
call. Import the module containing the registration before loading a ModelSpec
that names the plugin. BioLGCA does not discover or execute arbitrary modules
from model files.

The following complete operator preserves particle number by permuting each
full local channel state. Its metadata declares the supported backend,
parameters, conservation law, and written state:

.. code-block:: python

   from lgca.operator_base import (
       ConservationLaw,
       ParameterSpec,
       PluginInfo,
       ReorientationOperator,
   )
   from lgca.plugins import register_plugin

   INFO = PluginInfo(
       name="my_package.channel_shuffle",
       operator_kind="reorientation",
       backend_families=("classical",),
       parameters={
           "enabled": ParameterSpec(default=True, type_label="boolean"),
       },
       conservation_law=ConservationLaw(
           conserves_total_particles=True,
           conserves_phenotype_particles=True,
           conserves_momentum=False,
           changes=("channel occupancy",),
       ),
       port_status="native",
       test_status="unit_tested",
       description="Uniformly permute complete local channel states.",
   )

   class ChannelShuffle(ReorientationOperator):
       def validate(self, context):
           if not context.spec.state.volume_exclusion:
               raise ValueError("channel_shuffle requires volume exclusion")

       def apply(self, context, step):
           if self.parameters["enabled"]:
               lgca = context.lgca
               lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)

   def factory(parameters=None):
       values = {"enabled": True, **dict(parameters or {})}
       return ChannelShuffle(INFO, values)

   register_plugin(INFO, factory)

Use ``{"name": "my_package.channel_shuffle"}`` in
``dynamics.operators`` after importing this module. Test the numerical operator
through a small seeded ModelSpec and assert its scientific invariant—for this
example, total particle count before and after a step—not merely that the
factory was called. Keep registration explicit; Python entry-point discovery
is intentionally outside the supported contract.
