from collections import Counter

from actproto.data.combinations import TWELVE_CALLS, combination_tasks
from actproto.data.generator import generate_tasks


def test_combination_set_covers_orders_and_a_twelve_call_chain():
    tasks = combination_tasks()
    by_id = {task.id: task for task in tasks}
    assert len(by_id) == len(tasks)
    assert len(tasks) == 12 + 24 + 2
    long = by_id["combo-12"]
    assert long.gold_tools == list(TWELVE_CALLS)
    assert len(long.gold_tools) == 12
    twice = by_id["combo-twice"]
    assert Counter(name for name in twice.gold_tools if name != "finish").most_common(1)[0][1] == 2
    for task in tasks:
        if task.id == "combo-12":
            continue
        counts = Counter(task.gold_tools)
        assert max(counts.values()) <= 2
    families = {task.template_family for task in generate_tasks()}
    assert "tool_combination" not in families
