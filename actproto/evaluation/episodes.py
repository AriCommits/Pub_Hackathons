"""Convert EventLogs to recorded episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from actproto.agent.trajectory import EventLog, eventlog_to_openai_messages


@dataclass
class Episode:
    case_id: str
    messages: list[dict[str, Any]]
    final_answer: str | None = None
    final_state: dict[str, Any] = field(default_factory=dict)
    latency_s: float | None = None


def episode_from_openai_messages(
    messages: list[dict[str, Any]],
    *,
    case_id: str,
    final_answer: str | None = None,
) -> Episode:
    return Episode(case_id=case_id, messages=list(messages), final_answer=final_answer)


def eventlog_to_episode(log: EventLog, case_id: str) -> Episode:
    ep = episode_from_openai_messages(
        eventlog_to_openai_messages(log),
        case_id=case_id,
        final_answer=log.final_answer,
    )
    ep.final_state = dict(log.final_state)
    return ep


def try_auditkit_episode(messages: list[dict[str, Any]], case_id: str, final_answer: str | None) -> Episode:
    try:
        from auditkit.agent_eval import episode_from_openai_messages as ak  # type: ignore

        ep = ak(messages, case_id=case_id, final_answer=final_answer)
        wrapped = Episode(case_id=case_id, messages=messages, final_answer=final_answer)
        if hasattr(ep, "final_state"):
            wrapped.final_state = dict(ep.final_state or {})
        return wrapped
    except ImportError:
        return episode_from_openai_messages(messages, case_id=case_id, final_answer=final_answer)
