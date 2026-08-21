import pytest

from lgca.classical_operators import (
    CLASSICAL_RANDOM_WALK_INFO,
    NativeClassicalRandomWalkOperator,
    create_classical_random_walk,
)
from lgca.operator_base import (
    ConservationLaw as BaseConservationLaw,
    ParameterSpec as BaseParameterSpec,
    PluginInfo as BasePluginInfo,
)
from lgca.operator_registry import PluginRegistry as FocusedPluginRegistry
from lgca.plugins import (
    ConservationLaw,
    ParameterSpec,
    PluginInfo,
    PluginRegistry,
    create_plugin,
    describe_plugin,
    interaction_coverage_table,
    list_plugins,
)


def test_focused_extension_contract_keeps_plugins_facade_compatible():
    """Moving contract types must not split public type identity."""

    assert ParameterSpec is BaseParameterSpec
    assert ConservationLaw is BaseConservationLaw
    assert PluginInfo is BasePluginInfo
    assert PluginRegistry is FocusedPluginRegistry


def test_classical_random_walk_is_a_complete_extracted_vertical_slice():
    """The registered factory must resolve the extracted implementation."""

    operator = create_plugin("classical.random_walk")

    assert type(operator) is NativeClassicalRandomWalkOperator
    assert operator.info is CLASSICAL_RANDOM_WALK_INFO
    assert operator.info.name == "classical.random_walk"
    assert operator.info.parameter_specs == {}
    assert type(create_classical_random_walk()) is NativeClassicalRandomWalkOperator


EXPECTED_LEGACY_INTERACTIONS = {
    "classical.aggregation",
    "classical.alignment",
    "classical.birth",
    "classical.birthdeath",
    "classical.chemotaxis",
    "classical.contact_guidance",
    "classical.excitable_medium",
    "classical.go_or_grow",
    "classical.go_or_rest",
    "classical.nematic",
    "classical.only_propagation",
    "classical.persistent_walk",
    "classical.random_walk",
    "classical.wetting",
    "ib.birth",
    "ib.birthdeath",
    "ib.birthdeath_discrete",
    "ib.go_and_grow_mutations",
    "ib.go_or_grow",
    "ib.random_walk",
    "multispecies.birth",
    "multispecies.birthdeath",
    "multispecies.excitable_medium_ms",
    "multispecies.go_or_grow",
    "nove.dd_alignment",
    "nove.di_alignment",
    "nove.go_or_grow",
    "nove.go_or_rest",
    "nove.random_walk",
    "nove_ib.birth",
    "nove_ib.birthdeath",
    "nove_ib.birthdeath_cancerdfe",
    "nove_ib.evo_steric",
    "nove_ib.go_or_grow",
    "nove_ib.go_or_grow_glioblastoma",
    "nove_ib.go_or_grow_kappa",
    "nove_ib.go_or_grow_kappa_chemo",
    "nove_ib.random_walk",
}


def test_registry_rejects_alias_that_shadows_canonical_name_atomically():
    registry = PluginRegistry()
    factory = lambda parameters=None: None
    registry.register(
        PluginInfo(name="canonical.one", operator_kind="test", backend_families=("test",)),
        factory,
    )

    with pytest.raises(ValueError, match="canonical.one"):
        registry.register(
            PluginInfo(
                name="canonical.two",
                aliases=("canonical.one",),
                operator_kind="test",
                backend_families=("test",),
            ),
            factory,
        )

    assert [plugin.name for plugin in registry.list()] == ["canonical.one"]


def test_registry_rejects_canonical_name_that_shadows_alias_atomically():
    registry = PluginRegistry()
    factory = lambda parameters=None: None
    registry.register(
        PluginInfo(
            name="canonical.one",
            aliases=("shared",),
            operator_kind="test",
            backend_families=("test",),
        ),
        factory,
    )

    with pytest.raises(ValueError, match="shared"):
        registry.register(
            PluginInfo(
                name="shared",
                aliases=("unused",),
                operator_kind="test",
                backend_families=("test",),
            ),
            factory,
        )

    assert [plugin.name for plugin in registry.list()] == ["canonical.one"]
    with pytest.raises(KeyError):
        registry.resolve("unused")


def test_every_existing_public_interaction_is_registered():
    registered = {plugin.name for plugin in list_plugins(kind="interaction")}

    assert EXPECTED_LEGACY_INTERACTIONS <= registered


def test_ported_plugin_metadata_exposes_morpheus_style_contract():
    plugin = describe_plugin("classical.excitable_medium")

    assert plugin.name == "classical.excitable_medium"
    assert plugin.legacy_source == "lgca.interactions.excitable_medium"
    assert plugin.operator_kind == "birth_death"
    assert plugin.backend_families == ("classical",)
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    assert plugin.conservation_law.conserves_total_particles is False
    assert plugin.conservation_law.conserves_phenotype_particles is False
    assert plugin.conservation_law.conserves_momentum is False


def test_interaction_coverage_table_reports_port_and_test_status_for_all_plugins():
    rows = interaction_coverage_table()
    plugins = list_plugins(kind="interaction")

    assert len(rows) == len(plugins)
    assert {row["name"] for row in rows} == {plugin.name for plugin in plugins}
    for row in rows:
        assert {
            "name",
            "operator_kind",
            "backend_families",
            "port_status",
            "test_status",
            "legacy_source",
            "conservation_law",
        } <= row.keys()
        assert row["port_status"] in {"native", "legacy_wrapper", "planned_native", "deprecated"}
        assert row["test_status"] in {"unit_tested", "smoke_tested", "invariant_tested", "unverified"}

    native_birth_death = next(row for row in rows if row["name"] == "birth_death")
    assert native_birth_death["port_status"] == "native"
    assert native_birth_death["test_status"] == "unit_tested"
    assert {row["port_status"] for row in rows} == {"native"}
