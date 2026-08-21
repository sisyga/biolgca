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

    def register(self, info: PluginInfo, factory: PluginFactory) -> None:
        occupied = set(self._plugins) | set(self._aliases)
        if info.name in occupied:
            raise ValueError(f"Plugin name {info.name!r} is already registered as a name or alias.")
        aliases = tuple(info.aliases)
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"Plugin {info.name!r} declares duplicate aliases.")
        for alias in aliases:
            if alias == info.name or alias in occupied:
                raise ValueError(
                    f"Plugin alias {alias!r} is already registered as a name or alias."
                )
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
