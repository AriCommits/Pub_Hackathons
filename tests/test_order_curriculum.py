from collections import Counter

from actproto.data.generator import generate_tasks
from actproto.data.order_curriculum import TWELVE_CALLS, order_tasks, result_token
from actproto.tools.registry import call_tool


def test_order_curriculum_counts_and_finish_digest():
    once = order_tasks(1)
    assert len(once) == 12 + 24 + 2
    assert len(order_tasks(4)) == 152
    long = next(task for task in once if task.id == "order-12-0")
    assert long.gold_tools == list(TWELVE_CALLS)
    assert long.gold_args[-1]["answer"] not in long.prompt
    tokens = []
    for name, args in zip(long.gold_tools, long.gold_args, strict=True):
        if name == "finish":
            continue
        tokens.append(result_token(name, call_tool(name, args)))
    assert long.oracle.answer == "|".join(tokens)
    for task in once:
        if task.id.startswith("order-12-"):
            continue
        assert max(Counter(task.gold_tools).values()) <= 2
    families = {task.template_family for task in generate_tasks()}
    assert "order_curriculum" not in families
