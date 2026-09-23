"""Plugin names and aliases must stay unambiguous; a rejected registration changes nothing."""

import pytest

from lgca.plugins import PluginInfo, PluginRegistry


def _factory(parameters=None):
    return None


def test_registry_rejects_alias_that_shadows_canonical_name_atomically():
    registry = PluginRegistry()
    registry.register(PluginInfo(name="canonical.one", operator_kind="test", backend_families=("test",)), _factory)

    with pytest.raises(ValueError, match="canonical.one"):
        registry.register(
            PluginInfo(name="canonical.two", aliases=("canonical.one",), operator_kind="test",
                       backend_families=("test",)),
            _factory,
        )

    assert [plugin.name for plugin in registry.list()] == ["canonical.one"]


def test_registry_rejects_canonical_name_that_shadows_alias_atomically():
    registry = PluginRegistry()
    registry.register(
        PluginInfo(name="canonical.one", aliases=("shared",), operator_kind="test", backend_families=("test",)),
        _factory,
    )

    with pytest.raises(ValueError, match="shared"):
        registry.register(
            PluginInfo(name="shared", aliases=("unused",), operator_kind="test", backend_families=("test",)),
            _factory,
        )

    assert [plugin.name for plugin in registry.list()] == ["canonical.one"]
    with pytest.raises(KeyError):
        registry.resolve("unused")


def _other_module_factory(parameters=None):
    return None


_other_module_factory.__module__ = "some_other_package"


def test_reregistering_from_the_same_module_replaces_the_plugin_and_its_aliases():
    registry = PluginRegistry()
    registry.register(
        PluginInfo(name="mine", aliases=("old",), operator_kind="test", backend_families=("test",)), _factory
    )
    registry.register(
        PluginInfo(name="mine", aliases=("new",), operator_kind="test", backend_families=("test",),
                   description="second version"),
        _factory,
    )

    assert registry.describe("mine").description == "second version"
    assert registry.describe("new").name == "mine"
    with pytest.raises(KeyError):
        registry.resolve("old")


def test_replacing_a_plugin_of_another_module_requires_replace():
    registry = PluginRegistry()
    registry.register(PluginInfo(name="builtin", operator_kind="test", backend_families=("test",)), _factory)
    info = PluginInfo(name="builtin", operator_kind="test", backend_families=("test",), description="mine")

    with pytest.raises(ValueError, match="replace=True"):
        registry.register(info, _other_module_factory)
    assert registry.describe("builtin").description == ""

    registry.register(info, _other_module_factory, replace=True)
    assert registry.resolve("builtin") is _other_module_factory


def test_every_builtin_parameter_is_explained():
    from lgca.plugins import list_plugins

    missing = [f"{plugin.name}.{name}" for plugin in list_plugins()
               for name, spec in plugin.parameter_specs.items() if not spec.description]

    assert missing == []


def test_plugin_card_shows_defaults_and_meanings():
    from lgca.plugins import describe_plugin

    card = str(describe_plugin("classical.go_or_grow"))

    assert "kappa (default 5.0)" in card
    assert "Positive kappa makes crowded cells rest" in card
