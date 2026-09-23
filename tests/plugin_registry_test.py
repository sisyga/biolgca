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
