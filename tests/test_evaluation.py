from actproto.evaluation.cases import AgentCase, FinalStateAssertion
from actproto.evaluation.episodes import episode_from_openai_messages
from actproto.evaluation.runner import AgentEvalRunner, AgentEvalSpec


def build_episodes():
    ep_a = episode_from_openai_messages(
        [
            {"role": "user", "content": "Check inventory and ship item A if possible."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "lookup_record", "arguments": '{"record_id": "REC-001"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "in_stock=true"},
            {"role": "assistant", "content": "Shipment for A is ready."},
        ],
        case_id="inventory-001",
        final_answer="Shipment for A is ready.",
    )
    ep_a.final_state = {"shipment_decision": "ready"}

    ep_b = episode_from_openai_messages(
        [
            {"role": "user", "content": "Check inventory and ship item B if possible."},
            {"role": "assistant", "content": "Shipment for B is ready."},
        ],
        case_id="inventory-002",
        final_answer="Shipment for B is ready.",
    )
    ep_b.final_state = {"shipment_decision": "blocked"}
    return [ep_a, ep_b]


def build_cases():
    return [
        AgentCase(
            id="inventory-001",
            task="Check inventory and ship item A if possible.",
            category="tool_use",
            allowed_tools=["lookup_record"],
            reference_turns=[[{"name": "lookup_record", "arguments": {"record_id": "REC-001"}}]],
            outcome=FinalStateAssertion("shipment_decision", equals="ready"),
        ),
        AgentCase(
            id="inventory-002",
            task="Check inventory and ship item B if possible.",
            category="tool_use",
            allowed_tools=["lookup_record"],
            reference_turns=[],
            outcome=FinalStateAssertion("shipment_decision", equals="ready"),
        ),
    ]


def test_fabricated_answer_fails_state_assertion():
    spec = AgentEvalSpec(cases=build_cases(), mode="recorded", episodes=build_episodes(), trials=1)
    result = AgentEvalRunner().run(spec)
    verdicts = {r.case_id: r.outcome["verdict"] for r in result.rows}
    metrics = {r.case_id: r.metrics for r in result.rows}
    averages = result.as_dict()["averages"]
    assert verdicts["inventory-001"] == "success"
    assert verdicts["inventory-002"] == "failure"
    assert metrics["inventory-001"]["first_tool_accuracy"] == 1.0
    assert metrics["inventory-001"]["first_argument_correctness"] == 1.0
    assert metrics["inventory-002"]["first_tool_accuracy"] == 0.0
    assert metrics["inventory-002"]["first_argument_correctness"] == 0.0
    assert averages["first_tool_accuracy"] == 0.5
    assert averages["first_argument_correctness"] == 0.5
    assert averages["argument_correctness"] == 0.5
    assert "task_success" in averages
