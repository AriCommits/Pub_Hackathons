"""Fail-closed checks for the pinned AgentTune/TRL integration."""
from __future__ import annotations

import importlib.metadata
import inspect
from typing import Any

from actproto.training.rl_config import GRPO_GPU_PROFILE

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


def _require_verified_device(use_cpu: Any, gpu_profile: Any) -> None:
    """Allow use_cpu false only for the named GPU profile."""
    if gpu_profile is None:
        if use_cpu is not True:
            raise BackendCompatibilityError("Only CPU trainer construction is currently verified")
        return
    if gpu_profile != GRPO_GPU_PROFILE:
        raise BackendCompatibilityError(f"gpu_profile must be {GRPO_GPU_PROFILE}")
    if use_cpu is not False:
        raise BackendCompatibilityError("The grpo_v2_gpu profile requires use_cpu false")


def create_trainer(**kwargs: Any) -> Any:
    """Construct the wrapper, refusing keys upstream would silently discard."""
    gpu_profile = kwargs.pop("gpu_profile", None)
    validate_versions()
    validate_agenttune_source()
    from trl import GRPOConfig, GRPOTrainer
    from agenttune.core.backend_factory import create_agentic_trainer

    allowed = (set(inspect.signature(GRPOConfig.__init__).parameters)
               | set(inspect.signature(GRPOTrainer.__init__).parameters)) - {"self", "args"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise BackendCompatibilityError(f"Unsupported trainer options: {', '.join(sorted(unknown))}")
    if kwargs.get("tools") or kwargs.get("environment_factory"):
        raise BackendCompatibilityError("Only the explicit AgenticRolloutAdapter is supported")
    rollout = kwargs.get("rollout_func")
    if rollout is not None and not isinstance(rollout, AgenticRolloutAdapter):
        raise BackendCompatibilityError("Only the explicit AgenticRolloutAdapter is supported")
    _require_verified_device(kwargs.get("use_cpu"), gpu_profile)
    if kwargs.get("use_vllm"):
        raise BackendCompatibilityError("vLLM is outside the verified baseline")
    if not kwargs.get("train_dataset"):
        raise BackendCompatibilityError("A nonempty training dataset is required")
    if not kwargs.get("reward_funcs"):
        raise BackendCompatibilityError("A reward function is required")
    return create_agentic_trainer("grpo", backend="trl", **kwargs)


def render_tokens(tokenizer: Any, messages: list[dict], *, generation: bool,
                  chat_template: str | None = None) -> list[int]:
    """Render messages. A dict of named templates requires an explicit string.

    ``messages_for_template`` still rewrites tool observations as user turns
    before the tokenizer sees them. A string ``tokenizer.chat_template`` keeps
    the historical call with no ``chat_template`` argument.
    """
    from actproto.agent.prompts import messages_for_template

    prepared = messages_for_template(messages)
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": generation}
    if chat_template is not None:
        if not isinstance(chat_template, str) or not chat_template:
            raise BackendCompatibilityError("chat template must be an explicit string")
        kwargs["chat_template"] = chat_template
    elif isinstance(getattr(tokenizer, "chat_template", None), dict):
        raise BackendCompatibilityError("named chat template required before sampling")
    encoded = tokenizer.apply_chat_template(prepared, **kwargs)
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    return list(encoded)


def _with_named_default(tokenizer: Any, chat_template: str | None) -> str | None:
    """Pass ``default`` when the tokenizer stores a dict and the caller did not."""
    if chat_template is not None:
        return chat_template
    if isinstance(getattr(tokenizer, "chat_template", None), dict):
        from actproto.training.sft_masks import selected_template
        return selected_template(tokenizer, "default")
    return None


def episode_token_mask(tokenizer: Any, initial: list[dict], action: str,
                       observation: str, finish: str) -> tuple[list[int], list[int]]:
    """Render a single work call and finish; reject templates that rewrite history.

    The mask describes template spans, not evidence of the backend's actual loss.
    """
    template = _with_named_default(tokenizer, None)
    stages = [
        render_tokens(tokenizer, initial, generation=True, chat_template=template),
        render_tokens(tokenizer, initial + [{"role": "assistant", "content": action}], generation=False,
                      chat_template=template),
        render_tokens(tokenizer, initial + [{"role": "assistant", "content": action},
                      {"role": "tool", "content": observation}], generation=True, chat_template=template),
        render_tokens(tokenizer, initial + [{"role": "assistant", "content": action},
                      {"role": "tool", "content": observation},
                      {"role": "assistant", "content": finish}], generation=False, chat_template=template),
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
    template = _with_named_default(tokenizer, None)
    messages = list(initial)
    tokens = render_tokens(tokenizer, messages, generation=True, chat_template=template)
    mask = [0] * len(tokens)
    for index, (action, sampled, observation) in enumerate(decisions):
        if not sampled or (observation is None) != (index == len(decisions) - 1):
            raise BackendCompatibilityError("Invalid sampled action or termination position")
        messages.append({"role": "assistant", "content": action})
        rendered = render_tokens(tokenizer, messages, generation=False, chat_template=template)
        expected = tokens + list(sampled)
        if rendered[:len(expected)] != expected:
            raise BackendCompatibilityError("Sampled action tokens do not match the rendered context")
        mask.extend([1] * len(sampled))
        mask.extend([0] * (len(rendered) - len(expected)))
        tokens = rendered
        if observation is not None:
            messages.append({"role": "tool", "content": observation})
            rendered = render_tokens(tokenizer, messages, generation=True, chat_template=template)
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
    template = _with_named_default(tokenizer, None)
    messages = list(initial)
    decisions = []
    observations = []
    for _ in range(max_decisions):
        prompt_ids = render_tokens(tokenizer, messages, generation=True, chat_template=template)
        action, sampled = generate(list(prompt_ids))
        sampled = list(sampled)
        rendered = render_tokens(tokenizer, messages + [{"role": "assistant", "content": action}],
                                 generation=False, chat_template=template)
        if not sampled or rendered[:len(prompt_ids) + len(sampled)] != prompt_ids + sampled:
            raise BackendCompatibilityError("Sampled action tokens do not match the rendered context")
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


class AgenticRolloutAdapter:
    """Bridge an injected sampling policy and tool executor to AgentTune's TRL API.

    The policy receives exact prompt IDs and the trainer, and must return action
    text, the exact sampled IDs and their finite sampling log probabilities.
    Tool execution and episode isolation are supplied by the caller.
    """
    def __init__(self, tokenizer: Any, generate: Any, execute_tool: Any, *, max_decisions: int = 4):
        self.tokenizer = tokenizer
        self.generate = generate
        self.execute_tool = execute_tool
        self.max_decisions = max_decisions

    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        import math

        result: dict[str, Any] = {key: [] for key in
            ("prompt_ids", "completion_ids", "logprobs", "env_mask", "observations", "finish")}
        for prompt in prompts:
            if isinstance(prompt, str):
                initial = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list) and all(
                isinstance(message, dict) and message.get("role") in {"system", "user"}
                and isinstance(message.get("content"), str) for message in prompt
            ):
                initial = [dict(message) for message in prompt]
            else:
                raise BackendCompatibilityError("Prompts must contain only public system/user messages")
            if not initial:
                raise BackendCompatibilityError("An episode requires a public prompt")
            sampled_logprobs: list[float] = []
            def sample(ids):
                action, token_ids, logprobs = self.generate(ids, trainer)
                if not token_ids or len(token_ids) != len(logprobs):
                    raise BackendCompatibilityError("Sampled IDs and logprobs must have matching nonzero lengths")
                values = [float(value) for value in logprobs]
                if not all(math.isfinite(value) for value in values):
                    raise BackendCompatibilityError("Sampled logprobs must be finite")
                sampled_logprobs.extend(values)
                return action, token_ids
            episode = run_fixture_episode(self.tokenizer, initial, sample, self.execute_tool,
                                          max_decisions=self.max_decisions)
            prompt_length = len(render_tokens(
                self.tokenizer, initial, generation=True,
                chat_template=_with_named_default(self.tokenizer, None)))
            completion = episode["token_ids"][prompt_length:]
            mask = episode["policy_mask"][prompt_length:]
            values = iter(sampled_logprobs)
            aligned_logprobs = [next(values) if item else 0.0 for item in mask]
            if next(values, None) is not None:
                raise BackendCompatibilityError("Unused sampled logprobs in episode")
            result["prompt_ids"].append(episode["token_ids"][:prompt_length])
            result["completion_ids"].append(completion)
            result["logprobs"].append(aligned_logprobs)
            result["env_mask"].append(mask)
            result["observations"].append(episode["observations"])
            result["finish"].append(episode["finish"])
        return result



def validate_agenttune_source() -> str:
    """Verify PEP 610 provenance, or compare an installed local checkout byte-for-byte."""
    import hashlib
    import json
    import subprocess
    from pathlib import Path
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    revision = "d4db4c12d52342227c4113a70183d0f64aa2f345"
    dist = importlib.metadata.distribution("agenttune")
    raw = dist.read_text("direct_url.json")
    if not raw:
        raise BackendCompatibilityError("AgentTune source provenance missing; install the pinned git revision")
    info = json.loads(raw)
    vcs = info.get("vcs_info", {})
    if vcs:
        if vcs.get("vcs") != "git" or vcs.get("commit_id") != revision:
            raise BackendCompatibilityError("AgentTune git revision does not match the pinned source")
        return revision
    parsed = urlparse(info.get("url", ""))
    if parsed.scheme != "file" or not info.get("dir_info") == {}:
        raise BackendCompatibilityError("Unsupported AgentTune source provenance")
    path = url2pathname(unquote(parsed.path))
    # Windows file URLs use /C:/...; url2pathname removes the extra slash there.
    checkout = Path(path)
    try:
        head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
                               check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BackendCompatibilityError("Cannot verify local AgentTune source checkout") from exc
    if head != revision or dirty:
        raise BackendCompatibilityError("Local AgentTune checkout is modified or at the wrong revision")
    checked = 0
    for entry in dist.files or []:
        relative = Path(str(entry))
        if not relative.parts or relative.parts[0] != "agenttune" or relative.suffix != ".py":
            continue
        source = checkout / "src" / relative
        installed = Path(dist.locate_file(entry))
        if not source.is_file() or hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(installed.read_bytes()).digest():
            raise BackendCompatibilityError(f"Installed AgentTune source differs from its checkout: {relative}")
        checked += 1
    if not checked:
        raise BackendCompatibilityError("No AgentTune source files available for verification")
    return revision
