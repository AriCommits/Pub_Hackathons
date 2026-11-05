"""Fail-closed checks for the pinned AgentTune/TRL integration."""
from __future__ import annotations

import importlib.metadata
import inspect
from typing import Any

AGENTTUNE_VERSION = "1.1.3"
TRL_VERSION = "1.7.1"


class BackendCompatibilityError(RuntimeError):
    """The installed backend cannot safely execute the requested operation."""


def validate_versions() -> dict[str, str]:
    versions = {}
    for package, expected in (("agenttune", AGENTTUNE_VERSION), ("trl", TRL_VERSION)):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise BackendCompatibilityError(f"Install the isolated RL requirements: missing {package}") from exc
        if actual != expected:
            raise BackendCompatibilityError(f"{package} {actual} is unsupported; expected {expected}")
        versions[package] = actual
    return versions


def create_trainer(**kwargs: Any) -> Any:
    """Construct the wrapper, refusing keys upstream would silently discard."""
    validate_versions()
    from trl import GRPOConfig, GRPOTrainer
    from agenttune.core.backend_factory import create_agentic_trainer

    allowed = (set(inspect.signature(GRPOConfig.__init__).parameters)
               | set(inspect.signature(GRPOTrainer.__init__).parameters)) - {"self", "args"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise BackendCompatibilityError(f"Unsupported trainer options: {', '.join(sorted(unknown))}")
    if kwargs.get("tools") or kwargs.get("rollout_func") or kwargs.get("environment_factory"):
        raise BackendCompatibilityError("Agentic rollout is not yet verified for Tiny Aya")
    if kwargs.get("use_cpu") is not True:
        raise BackendCompatibilityError("Only CPU trainer construction is currently verified")
    if kwargs.get("use_vllm"):
        raise BackendCompatibilityError("vLLM is outside the verified baseline")
    if not kwargs.get("train_dataset"):
        raise BackendCompatibilityError("A nonempty training dataset is required")
    if not kwargs.get("reward_funcs"):
        raise BackendCompatibilityError("A reward function is required")
    return create_agentic_trainer("grpo", backend="trl", **kwargs)


def render_tokens(tokenizer: Any, messages: list[dict], *, generation: bool) -> list[int]:
    from actproto.agent.prompts import messages_for_template
    encoded = tokenizer.apply_chat_template(messages_for_template(messages), tokenize=True,
                                             add_generation_prompt=generation)
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    return list(encoded)


def episode_token_mask(tokenizer: Any, initial: list[dict], action: str,
                       observation: str, finish: str) -> tuple[list[int], list[int]]:
    """Render a single work call and finish; reject templates that rewrite history.

    The mask describes template spans, not evidence of the backend's actual loss.
    """
    stages = [
        render_tokens(tokenizer, initial, generation=True),
        render_tokens(tokenizer, initial + [{"role": "assistant", "content": action}], generation=False),
        render_tokens(tokenizer, initial + [{"role": "assistant", "content": action},
                      {"role": "tool", "content": observation}], generation=True),
        render_tokens(tokenizer, initial + [{"role": "assistant", "content": action},
                      {"role": "tool", "content": observation},
                      {"role": "assistant", "content": finish}], generation=False),
    ]
    for earlier, later in zip(stages, stages[1:]):
        if later[:len(earlier)] != earlier:
            raise BackendCompatibilityError("Chat template rewrites earlier tokens; cannot safely mask episode")
    mask = [0] * len(stages[0])
    for previous, current, policy in zip(stages, stages[1:], (1, 0, 1)):
        mask.extend([policy] * (len(current) - len(previous)))
    if not all(len(stages[i]) < len(stages[i + 1]) for i in range(3)):
        raise BackendCompatibilityError("Empty action or observation token span")
    return stages[-1], mask



def assemble_episode_tokens(tokenizer: Any, initial: list[dict],
                            decisions: list[tuple[str, list[int], str | None]]) -> tuple[list[int], list[int]]:
    """Assemble rendered sampled tokens, marking only sampled policy tokens.

    Each decision is (decoded action, exact sampled token IDs, observation).
    The last decision has no observation. Reject any tokenization that rewrites
    sampled history instead of silently training on a different context.
    """
    if not decisions or decisions[-1][2] is not None:
        raise BackendCompatibilityError("Episode must end with a terminal decision")
    messages = list(initial)
    tokens = render_tokens(tokenizer, messages, generation=True)
    mask = [0] * len(tokens)
    for index, (action, sampled, observation) in enumerate(decisions):
        if not sampled or (observation is None) != (index == len(decisions) - 1):
            raise BackendCompatibilityError("Invalid sampled action or termination position")
        messages.append({"role": "assistant", "content": action})
        rendered = render_tokens(tokenizer, messages, generation=False)
        expected = tokens + list(sampled)
        if rendered[:len(expected)] != expected:
            raise BackendCompatibilityError("Sampled action tokens do not match the rendered context")
        mask.extend([1] * len(sampled))
        mask.extend([0] * (len(rendered) - len(expected)))
        tokens = rendered
        if observation is not None:
            messages.append({"role": "tool", "content": observation})
            rendered = render_tokens(tokenizer, messages, generation=True)
            if rendered[:len(tokens)] != tokens:
                raise BackendCompatibilityError("Observation rewrites earlier sampled tokens")
            mask.extend([0] * (len(rendered) - len(tokens)))
            tokens = rendered
    return tokens, mask


def run_fixture_episode(tokenizer: Any, initial: list[dict], generate: Any,
                        execute_tool: Any, *, max_decisions: int = 4) -> dict[str, Any]:
    """Exercise the proposed template boundary with injected CPU fixtures.

    This is a feasibility probe, not a production collector or optimizer.
    """
    from actproto.agent.inference_v2 import parse_single_call

    if not 1 <= max_decisions <= 32:
        raise BackendCompatibilityError("max_decisions must be between 1 and 32")
    messages = list(initial)
    decisions = []
    observations = []
    for _ in range(max_decisions):
        prompt_ids = render_tokens(tokenizer, messages, generation=True)
        action, sampled = generate(list(prompt_ids))
        call = parse_single_call(action)
        if call['name'] == 'finish':
            decisions.append((action, list(sampled), None))
            tokens, mask = assemble_episode_tokens(tokenizer, initial, decisions)
            return {"token_ids": tokens, "policy_mask": mask,
                    "observations": observations, "finish": call}
        observation = str(execute_tool(call['name'], call['arguments']))
        decisions.append((action, list(sampled), observation))
        observations.append(observation)
        messages.extend([{"role": "assistant", "content": action},
                         {"role": "tool", "content": observation}])
    raise BackendCompatibilityError("Episode exhausted its decision budget before finish")
