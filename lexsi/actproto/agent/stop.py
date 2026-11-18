"""Decide when a decoded completion already contains one tool call."""

from __future__ import annotations

import re

_CLOSED_TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_END_OF_TURN = "<|END_OF_TURN_TOKEN|>"


def completion_should_stop(text: str) -> bool:
    """True once new tokens hold one closed tool call or an end-of-turn token.

    A half-open ``<tool_call>`` does not stop. An empty string does not stop.
    """
    if not isinstance(text, str) or text == "":
        return False
    if _END_OF_TURN in text:
        return True
    return _CLOSED_TOOL_CALL.search(text) is not None
