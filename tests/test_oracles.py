from actproto.data.generator import generate_tasks
from actproto.data.oracles import check_final_state, replay_gold


def test_gold_replay_matches_oracles_on_sample():
    tasks = generate_tasks()
    sample = tasks[:25]
    for task in sample:
        replay = replay_gold(task)
        result = check_final_state(task.oracle, replay["state"])
        assert result["verdict"] == "success", (task.id, result, replay["state"])
