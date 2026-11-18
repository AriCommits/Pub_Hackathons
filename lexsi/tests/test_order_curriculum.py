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


def test_all_formats_cover_header_positions_and_subsets():
    from actproto.data.order_curriculum import ALL_FORMATS

    tasks = order_tasks(1, formats=ALL_FORMATS)
    assert len(tasks) == 266
    by_id = {task.id: task for task in tasks}
    assert len(by_id) == len(tasks)
    first = by_id["order-calculator-search_docs-header_first-only_first-0"]
    last = by_id["order-calculator-search_docs-header_last-last_then_first-0"]
    assert first.prompt.startswith("Only execute step 1: calculator.")
    assert "Step 2: search_docs" in first.prompt
    assert first.gold_tools == ["calculator", "finish"]
    assert last.prompt.endswith("Execute steps 2 and 1: search_docs, then calculator.")
    assert last.gold_tools == ["search_docs", "calculator", "finish"]
    assert first.oracle.answer not in first.prompt
    assert last.oracle.answer not in last.prompt
    formats = {task.metadata["format"] for task in tasks}
    assert formats == {"sequence", "header_first", "header_last"}
