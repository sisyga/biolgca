"""Focused registry mechanics for interaction operators."""

from __future__ import annotations

from typing import Callable, Mapping, Any

from .operator_base import InteractionOperator, PluginInfo


PluginFactory = Callable[[Mapping[str, Any] | None], InteractionOperator]


class PluginRegistry:
    """Collision-safe registry for explicitly imported simulation plugins."""

    def __init__(self):
        self._plugins: dict[str, tuple[PluginInfo, PluginFactory]] = {}
        self._aliases: dict[str, str] = {}

    def register(self, info: PluginInfo, factory: PluginFactory, replace: bool = False) -> None:
        """Register ``factory`` under ``info.name`` and its aliases.

        Registering a name again from the module that registered it first
        replaces the entry, so re-running a notebook cell works. Replacing a
        plugin from another module, such as a built-in, requires
        ``replace=True``. A rejected registration changes nothing.
        """
        previous = self._plugins.get(info.name)
        if previous is not None and not replace:
            previous_module = getattr(previous[1], "__module__", None)
            if previous_module != getattr(factory, "__module__", None):
                raise ValueError(
                    f"Plugin name {info.name!r} is already registered by module "
                    f"{previous_module!r}. Choose another name or pass replace=True."
                )
        own_aliases = {alias for alias, name in self._aliases.items() if name == info.name}
        occupied = (set(self._plugins) - {info.name}) | (set(self._aliases) - own_aliases)
        if info.name in occupied:
            raise ValueError(f"Plugin name {info.name!r} is already registered as an alias.")
        aliases = tuple(info.aliases)
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"Plugin {info.name!r} declares duplicate aliases.")
        for alias in aliases:
            if alias == info.name or alias in occupied:
                raise ValueError(
                    f"Plugin alias {alias!r} is already registered as a name or alias."
                )
        for alias in own_aliases:
            del self._aliases[alias]
        self._plugins[info.name] = (info, factory)
        for alias in aliases:
            self._aliases[alias] = info.name

    def resolve(self, name: str) -> PluginFactory:
        canonical = self._aliases.get(name, name)
        try:
            return self._plugins[canonical][1]
        except KeyError as exc:
            raise KeyError(f"Unknown plugin {name!r}.") from exc

    def describe(self, name: str) -> PluginInfo:
        canonical = self._aliases.get(name, name)
        try:
            return self._plugins[canonical][0]
        except KeyError as exc:
            raise KeyError(f"Unknown plugin {name!r}.") from exc

    def list(self, kind: str | None = None) -> list[PluginInfo]:
        plugins = [entry[0] for entry in self._plugins.values()]
        if kind is None or kind == "interaction":
            return sorted(plugins, key=lambda plugin: plugin.name)
        return sorted(
            [plugin for plugin in plugins if plugin.operator_kind == kind],
            key=lambda plugin: plugin.name,
        )
