"""In-process tool harness compatible with AgentTune DictToolHarness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from actproto.tools.registry import as_harness_dict, call_tool


@dataclass
class ToolResult:
    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]


@dataclass
class DictToolHarness:
    tools: dict[str, Callable[..., dict[str, Any]]] = field(default_factory=as_harness_dict)
    max_steps: int = 6
    final_state: dict[str, Any] = field(default_factory=dict)

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        arguments = dict(arguments or {})
        if name not in self.tools and name != "no_tool":
            output = {"ok": False, "error": {"type": "unknown_tool", "message": f"unknown tool {name}"}}
        else:
            output = call_tool(name, arguments)
        if name in {"finish", "no_tool"} and output.get("ok"):
            self.final_state["answer"] = output["data"]["answer"]
            self.final_state["done"] = True
        if name == "lookup_record" and not output.get("ok"):
            self.final_state["lookup_error"] = True
        return ToolResult(name=name, arguments=arguments, output=output)

    def reset(self) -> None:
        self.final_state = {}


def build_harness(max_steps: int = 6) -> DictToolHarness:
    try:
        from agenttune.agentic import DictToolHarness as ATHarness  # type: ignore

        return ATHarness(as_harness_dict(), max_steps=max_steps)  # pragma: no cover
    except ImportError:
        return DictToolHarness(as_harness_dict(), max_steps=max_steps)
