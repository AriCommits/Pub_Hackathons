"""Collect seeded, execution-grounded v2 episodes without optimization."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from actproto.agent.inference_v2 import parse_single_call
from actproto.agent.prompts import system_prompt
from actproto.experiments.contracts import ExecutionTrace, TaskSpec
from actproto.integrations.agenttune_backend import AgenticRolloutAdapter, render_tokens
from actproto.tools.schemas import validate_args
from actproto.training.rl_tools import new_session


@dataclass(frozen=True)
class PolicySample:
    text: str
    token_ids: list[int]
    logprobs: list[float] | None = None
    truncated: bool = False


def task_digest(task: TaskSpec) -> str:
    value = json.dumps(task.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(value.encode()).hexdigest()


def rollout_id(run_id: str, task_id: str, group_index: int, sample_index: int) -> str:
    raw = json.dumps([run_id, task_id, group_index, sample_index], separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def rollout_seed(run_id: str, task_id: str, group_index: int, sample_index: int, seed: int) -> int:
    raw = json.dumps([seed, run_id, task_id, group_index, sample_index], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], 'big') % (2 ** 31)


def collect_episode(task: TaskSpec, *, run_id: str, group_index: int, sample_index: int,
                    seed: int, policy_hash: str, tokenizer: Any, policy_factory: Callable,
                    environment_factory: Callable, max_generated_tokens: int = 512,
                    max_seconds: float = 60, session=None,
                    allow_unscoped_fixture: bool = False, fixture_only: bool = False) -> dict[str, Any]:
    """Collect a fixture rollout. Time checks cannot interrupt a blocking call."""
    if not fixture_only:
        raise ValueError('behavior collection requires a hard-deadline executor; only synthetic fixtures are supported')
    if task.split != 'train':
        raise ValueError('only training tasks may be collected')
    if not run_id or not task.task_id or group_index < 0 or sample_index < 0:
        raise ValueError('invalid rollout identity')
    if len(policy_hash) != 64 or any(c not in '0123456789abcdef' for c in policy_hash):
        raise ValueError('policy_hash must be a lowercase SHA-256 digest')
    if max_generated_tokens < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('invalid token or time budget')
    local_seed = rollout_seed(run_id, task.task_id, group_index, sample_index, seed)
    try:
        policy = policy_factory(local_seed)
        policy_error = None
    except Exception as exc:
        policy = None
        policy_error = str(exc)
    session = session if session is not None else new_session(
        task, environment_factory, allow_unscoped_fixture=allow_unscoped_fixture)
    if session.unscoped_fixture and not allow_unscoped_fixture:
        raise ValueError('training collection requires manifest-verified ScopedFixtures')
    if session.task != task:
        raise ValueError('session task does not match rollout task')
    messages = [{'role': 'system', 'content': system_prompt()}, {'role': 'user', 'content': task.prompt}]
    initial_ids = render_tokens(tokenizer, messages, generation=True)
    token_ids = list(initial_ids)
    policy_mask = [0] * len(initial_ids)
    aligned_logprobs: list[float | None] = [None] * len(initial_ids)
    spans, generated_tokens = [], 0
    started = monotonic()
    completion = 'budget_exhausted'
    for _ in range(task.call_budget):
        if policy is None:
            completion = 'policy_error'
            session.events.append({'kind': 'policy_error', 'error': policy_error})
            break
        if monotonic() - started >= max_seconds:
            completion = 'timeout'
            session.events.append({'kind': 'timeout'})
            break
        prompt_ids = render_tokens(tokenizer, messages, generation=True)
        if prompt_ids != token_ids:
            completion = 'template_mismatch'
            session.events.append({'kind': 'template_mismatch'})
            break
        remaining = max_generated_tokens - generated_tokens
        if remaining <= 0:
            completion = 'token_budget_exhausted'
            session.events.append({'kind': 'token_budget_exhausted'})
            break
        try:
            sample = policy(list(prompt_ids), remaining)
        except Exception as exc:
            completion = 'policy_error'
            session.events.append({'kind': 'policy_error', 'error': str(exc)})
            break
        if not isinstance(sample, PolicySample) or not isinstance(sample.text, str):
            completion = 'policy_error'
            session.events.append({'kind': 'policy_error', 'error': 'policy must return PolicySample'})
            break
        ids = sample.token_ids
        try:
            valid_probs = (sample.logprobs is None or
                (isinstance(sample.logprobs, (list, tuple)) and len(sample.logprobs) == len(ids)
                 and all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)
                         for x in sample.logprobs)))
        except (TypeError, OverflowError, ValueError):
            valid_probs = False
        if (not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
                or not valid_probs):
            completion = 'policy_error'
            session.events.append({'kind': 'policy_error', 'error': 'invalid sampled tokens or logprobs'})
            break
        generated_tokens += len(ids)
        span = {'raw': sample.text, 'token_ids': list(ids), 'sampling_logprobs': deepcopy(sample.logprobs),
                'prompt_token_ids': list(prompt_ids), 'truncated': bool(sample.truncated), 'source': 'model'}
        spans.append(span)
        if monotonic() - started >= max_seconds:
            completion = 'timeout'
            session.events.append({'kind': 'timeout'})
            break
        if len(ids) > remaining:
            completion = 'policy_budget_overrun'
            session.events.append({'kind': 'policy_budget_overrun', 'raw': sample.text,
                                   'sampled_tokens': len(ids), 'remaining_tokens': remaining})
            break
        rendered = render_tokens(tokenizer, messages + [{'role': 'assistant', 'content': sample.text}], generation=False)
        if rendered[:len(prompt_ids) + len(ids)] != prompt_ids + ids:
            completion = 'template_mismatch'
            session.events.append({'kind': 'template_mismatch', 'raw': sample.text})
            break
        token_ids.extend(ids)
        policy_mask.extend([1] * len(ids))
        aligned_logprobs.extend(sample.logprobs if sample.logprobs is not None else [None] * len(ids))
        padding = rendered[len(token_ids):]
        token_ids.extend(padding)
        policy_mask.extend([0] * len(padding))
        aligned_logprobs.extend([None] * len(padding))
        messages.append({'role': 'assistant', 'content': sample.text})
        if sample.truncated or len(ids) > remaining:
            completion = 'truncated'
            session.events.append({'kind': 'truncation', 'raw': sample.text})
            break
        try:
            call = parse_single_call(sample.text)
            valid, reason = validate_args(call['name'], call['arguments'])
            if not valid:
                raise ValueError(reason)
        except (ValueError, TypeError) as exc:
            completion = 'parse_error'
            session.events.append({'kind': 'parse_error', 'raw': sample.text, 'error': str(exc)})
            break
        output = session.call(call['name'], call['arguments'])
        if monotonic() - started >= max_seconds:
            completion = 'timeout'
            session.events.append({'kind': 'timeout'})
            break
        if call['name'] in ('finish', 'no_tool') and output.get('ok'):
            completion = 'finish'
            break
        messages.append({'role': 'tool', 'content': json.dumps(output, sort_keys=True)})
        rendered = render_tokens(tokenizer, messages, generation=True)
        if rendered[:len(token_ids)] != token_ids:
            completion = 'template_mismatch'
            session.events.append({'kind': 'template_mismatch'})
            break
        added = rendered[len(token_ids):]
        token_ids.extend(added)
        policy_mask.extend([0] * len(added))
        aligned_logprobs.extend([None] * len(added))
    trace = ExecutionTrace(task.task_id, deepcopy(session.events), completion, 'harness',
        model_spans=spans, generated_tokens=generated_tokens, model_invocations=len(spans),
        final_answer=session.state.get('answer'), final_state=deepcopy(session.state))
    return {'schema_version': 'rl_rollout_v2', 'run_id': run_id, 'task_id': task.task_id,
            'group_id': f'{task.task_id}:{group_index}', 'rollout_id': rollout_id(run_id, task.task_id, group_index, sample_index),
            'group_index': group_index, 'sample_index': sample_index, 'seed': local_seed,
            'policy_hash': policy_hash, 'task_hash': task_digest(task), 'public_messages': deepcopy(messages), 'token_ids': token_ids,
            'attention_mask': [1] * len(token_ids), 'policy_mask': policy_mask,
            'sampling_logprobs': aligned_logprobs, 'prompt_length': len(initial_ids),
            'trace': trace.to_dict(), 'completion_reason': completion,
            'unscoped_fixture': session.unscoped_fixture, 'sampling_mode': 'synthetic_fixture',
            'elapsed_seconds': monotonic() - started}


def collect_group(task: TaskSpec, *, group_size: int, environment_factory: Callable, **kwargs) -> list[dict[str, Any]]:
    if group_size < 2:
        raise ValueError('group_size must be at least 2')
    sessions, records = [], []
    for index in range(group_size):
        session = new_session(task, environment_factory, allow_unscoped_fixture=kwargs.get('allow_unscoped_fixture', False))
        if any(session.environment is existing.environment for existing in sessions):
            raise ValueError('environment factory reused a rollout environment')
        sessions.append(session)
        records.append(collect_episode(task, sample_index=index, environment_factory=environment_factory,
                                       session=session, **kwargs))
    if len({row['seed'] for row in records}) != len(records):
        raise ValueError('seed collision in rollout group')
    return records


def read_rollouts(path: str | Path) -> dict[str, dict[str, Any]]:
    target = Path(path)
    existing: dict[str, dict[str, Any]] = {}
    if target.exists():
        for number, line in enumerate(target.read_text(encoding='utf-8').splitlines(), 1):
            try:
                row = json.loads(line)
                key = row['rollout_id']
                expected = rollout_id(row['run_id'], row['task_id'], row['group_index'], row['sample_index'])
                if row['schema_version'] != 'rl_rollout_v2' or key != expected or key in existing:
                    raise ValueError('invalid or duplicate rollout ID')
                trace = ExecutionTrace.from_dict(row['trace'])
                if trace.task_id != row['task_id'] or trace.completion_reason != row['completion_reason']:
                    raise ValueError('invalid trace association')
                existing[key] = row
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'invalid existing rollout at line {number}') from exc
    return existing


def append_new_rollouts(path: str | Path, records: list[dict[str, Any]]) -> int:
    """Append completed records once; fail closed on malformed or colliding identities."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = read_rollouts(target)
    pending = []
    for row in records:
        key = row['rollout_id']
        prior = existing.get(key)
        identity_fields = ('run_id', 'task_id', 'group_index', 'sample_index', 'seed', 'policy_hash', 'task_hash', 'policy_step', 'sampling_mode', 'batch_id')
        if prior is not None:
            if any(prior.get(field) != row.get(field) for field in identity_fields):
                raise ValueError('rollout ID collision with mismatched provenance')
            continue
        existing[key] = row
        pending.append(row)
    with target.open('a', encoding='utf-8', newline='\n') as stream:
        for row in pending:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return len(pending)


def append_abandoned_group(path: str | Path, event: dict[str, Any]) -> int:
    """Append an idempotent durable group-abandonment sidecar event."""
    if event.get('schema_version') != 'rl_rollout_group_resolution_v1':
        raise ValueError('invalid abandonment event')
    identity = (event.get('run_id'), event.get('task_id'), event.get('group_index'))
    if not isinstance(identity[0], str) or not identity[0] or not isinstance(identity[1], str) or not identity[1] or type(identity[2]) is not int or identity[2] < 0:
        raise ValueError('invalid abandonment identity')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        seen = set()
        for line in target.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = (row.get('run_id'), row.get('task_id'), row.get('group_index'))
            if row.get('schema_version') != 'rl_rollout_group_resolution_v1' or key in seen:
                raise ValueError('invalid or duplicated abandonment sidecar')
            seen.add(key)
            if key == identity:
                if row != event:
                    raise ValueError('abandonment identity collision')
                return 0
    with target.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return 1


def read_batch_events(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    events = []
    if target.exists():
        seen = set()
        for line in target.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            key = (event.get('batch_id'), event.get('phase'))
            if (event.get('schema_version') != 'rl_rollout_batch_v1' or not isinstance(key[0], str)
                    or key[1] not in {'prepare', 'ready', 'aborted'} or key in seen):
                raise ValueError('invalid or duplicate rollout batch event')
            phases = {prior['phase'] for prior in events if prior['batch_id'] == key[0]}
            if ((key[1] == 'prepare' and phases) or
                    (key[1] != 'prepare' and ('prepare' not in phases or {'ready', 'aborted'} & set(phases)))):
                raise ValueError('out-of-order rollout batch events')
            seen.add(key)
            events.append(event)
    return events


def append_batch_event(path: str | Path, event: dict[str, Any]) -> int:
    """Durably append an idempotent prepare/ready/aborted batch event."""
    if (event.get('schema_version') != 'rl_rollout_batch_v1' or not isinstance(event.get('batch_id'), str)
            or event.get('phase') not in {'prepare', 'ready', 'aborted'}):
        raise ValueError('invalid rollout batch event')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    events = read_batch_events(target)
    key = (event['batch_id'], event['phase'])
    for prior in events:
        if (prior['batch_id'], prior['phase']) == key:
            if prior != event:
                raise ValueError('batch event identity collision')
            return 0
    phases = {row['phase'] for row in events if row['batch_id'] == event['batch_id']}
    if event['phase'] == 'prepare':
        if phases:
            raise ValueError('batch prepare must be first')
    elif 'prepare' not in phases or {'ready', 'aborted'} & phases:
        raise ValueError('batch terminal event is invalid')
    with target.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return 1


@dataclass
class BehaviorEpisode:
    """In-process episode. The session is the scoreable evidence, not the trace dict."""
    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]
    env_mask: list[int]
    session: Any
    completion_reason: str


def _positive_deadline(value: object) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)) and float(value) > 0)


def _public_user_text(prompt: Any) -> str:
    if isinstance(prompt, str) and prompt:
        return prompt
    if isinstance(prompt, list):
        users = [message.get("content") for message in prompt
                 if isinstance(message, dict) and message.get("role") == "user"
                 and isinstance(message.get("content"), str) and message.get("content")]
        if len(users) == 1:
            return users[0]
    raise ValueError("trainer must provide one public user prompt")


def run_trusted_behavior_episode(task: TaskSpec, *, tokenizer: Any, policy: Callable,
                                 environment_factory: Callable, deadline_seconds: float,
                                 max_generated_tokens: int = 256, session=None,
                                 clock: Callable[[], float] | None = None,
                                 chat_template: str | None = None,
                                 context_limit: int | None = None) -> BehaviorEpisode:
    """Execute one train episode on a fresh scoped harness and keep the live session.

    Each sampled decision gets ``max_generated_tokens`` new tokens. That number
    is a per-turn cap, not a pool shared by later turns. A deadline is required.
    Timeout, truncation, context overflow, a parse failure, and an exhausted
    decision cap are invalid attempts. They are not a successful finish. The
    caller cannot supply the session or a serialized trace.
    """
    from actproto.training.rewards_v2 import TrustedRewardSession
    from actproto.agent.harness import ToolResult

    if session is not None:
        raise ValueError("caller-supplied sessions are not permitted")
    if not _positive_deadline(deadline_seconds):
        raise ValueError("behavior collection requires a hard-deadline executor; only synthetic fixtures are supported")
    if isinstance(max_generated_tokens, bool) or not isinstance(max_generated_tokens, int):
        raise ValueError("max_generated_tokens must be between 1 and 256")
    if max_generated_tokens < 1 or max_generated_tokens > 256:
        raise ValueError("max_generated_tokens must be between 1 and 256")
    if context_limit is not None and (isinstance(context_limit, bool) or not isinstance(context_limit, int)
                                       or context_limit < 1):
        raise ValueError("model context limit unavailable")
    per_turn = max_generated_tokens
    if task.split != "train" or not task.prompt:
        raise ValueError("only public training tasks may be rolled out")
    tool_session = new_session(task, environment_factory, allow_unscoped_fixture=False)

    class _Harness:
        @property
        def final_state(self):
            return tool_session.state

        def execute(self, name, arguments):
            output = tool_session.call(name, arguments)
            return ToolResult(name, deepcopy(arguments), output)

    reward = TrustedRewardSession(task, _Harness())
    now = clock or monotonic
    started = now()
    messages = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": task.prompt}]

    def _render(current: list[dict], generation: bool) -> list[int]:
        return render_tokens(tokenizer, current, generation=generation, chat_template=chat_template)

    token_ids = list(_render(messages, True))
    policy_mask = [0] * len(token_ids)
    aligned: list[float | None] = [None] * len(token_ids)
    prompt_length = len(token_ids)
    completion = "budget_exhausted"
    decision_cap = max(int(task.call_budget), 2)

    def _append_sample(sample: PolicySample, prompt_ids: list[int]) -> None:
        nonlocal token_ids, policy_mask, aligned
        rendered = _render(messages + [{"role": "assistant", "content": sample.text}], False)
        if rendered[:len(prompt_ids) + len(sample.token_ids)] != prompt_ids + list(sample.token_ids):
            raise ValueError("template_mismatch")
        token_ids.extend(sample.token_ids)
        policy_mask.extend([1] * len(sample.token_ids))
        aligned.extend(list(sample.logprobs))
        padding = rendered[len(token_ids):]
        token_ids.extend(padding)
        policy_mask.extend([0] * len(padding))
        aligned.extend([None] * len(padding))
        messages.append({"role": "assistant", "content": sample.text})

    def _keep_raw(sample: PolicySample) -> None:
        """Keep the sampled ids when the chat template cannot render them.

        An end-of-sequence or unparsable sample is still a policy decision.
        It is not executed and it is not a successful finish.
        """
        nonlocal token_ids, policy_mask, aligned
        token_ids.extend(sample.token_ids)
        policy_mask.extend([1] * len(sample.token_ids))
        aligned.extend(list(sample.logprobs))

    def _append_observation(output: dict) -> None:
        nonlocal token_ids, policy_mask, aligned
        messages.append({"role": "tool", "content": json.dumps(output, sort_keys=True)})
        rendered = _render(messages, True)
        if rendered[:len(token_ids)] != token_ids:
            raise ValueError("template_mismatch")
        added = rendered[len(token_ids):]
        token_ids.extend(added)
        policy_mask.extend([0] * len(added))
        aligned.extend([None] * len(added))

    for _ in range(decision_cap):
        if now() - started >= float(deadline_seconds):
            reward.invalid_attempt("timeout")
            completion = "timeout"
            break
        prompt_ids = _render(messages, True)
        if prompt_ids != token_ids:
            reward.invalid_attempt("template_mismatch")
            completion = "template_mismatch"
            break
        if context_limit is not None and len(prompt_ids) + per_turn > context_limit:
            reward.invalid_attempt("context_overflow")
            completion = "context_overflow"
            break
        try:
            sample = policy(list(prompt_ids), per_turn)
        except Exception as exc:
            reward.invalid_attempt(f"policy_error: {exc}")
            completion = "policy_error"
            break
        if now() - started >= float(deadline_seconds):
            reward.invalid_attempt("timeout")
            completion = "timeout"
            break
        if not isinstance(sample, PolicySample) or not isinstance(sample.text, str):
            reward.invalid_attempt("policy_error")
            completion = "policy_error"
            break
        ids = sample.token_ids
        probs = sample.logprobs
        valid_probs = (isinstance(probs, (list, tuple)) and len(probs) == len(ids)
                       and all(not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(item)
                               for item in probs))
        if (not isinstance(ids, list) or not ids or any(type(item) is not int or item < 0 for item in ids)
                or not valid_probs):
            reward.invalid_attempt("policy_error")
            completion = "policy_error"
            break
        if len(ids) > per_turn or sample.truncated:
            try:
                _append_sample(sample, prompt_ids)
            except ValueError:
                _keep_raw(sample)
                reward.invalid_attempt("template_mismatch")
                completion = "template_mismatch"
                break
            reward.invalid_attempt("truncation")
            completion = "truncated"
            break
        try:
            _append_sample(sample, prompt_ids)
        except ValueError:
            _keep_raw(sample)
            reward.invalid_attempt("template_mismatch")
            completion = "template_mismatch"
            break
        try:
            call = parse_single_call(sample.text)
            valid, reason = validate_args(call["name"], call["arguments"])
            if not valid:
                raise ValueError(reason)
        except (ValueError, TypeError) as exc:
            reward.invalid_attempt(f"parse_error: {exc}")
            completion = "parse_error"
            break
        if now() - started >= float(deadline_seconds):
            reward.invalid_attempt("timeout")
            completion = "timeout"
            break
        output = reward.execute(call["name"], call["arguments"])
        if call["name"] in ("finish", "no_tool") and isinstance(output, dict) and output.get("ok") is True:
            completion = "finish"
            break
        try:
            _append_observation(output)
        except ValueError:
            reward.invalid_attempt("template_mismatch")
            completion = "template_mismatch"
            break
    else:
        reward.invalid_attempt("budget_exhausted")
        completion = "budget_exhausted"
    if not any(policy_mask) and completion != "context_overflow":
        raise ValueError("behavior episode produced no policy tokens")
    completion_ids = token_ids[prompt_length:]
    mask = policy_mask[prompt_length:]
    logprobs = [float(value) if value is not None else 0.0 for value in aligned[prompt_length:]]
    if len(completion_ids) != len(mask) or len(mask) != len(logprobs):
        raise ValueError("behavior episode token alignment failed")
    if not completion_ids and completion != "context_overflow":
        raise ValueError("behavior episode token alignment failed")
    return BehaviorEpisode(token_ids[:prompt_length], completion_ids, logprobs, mask, reward, completion)


def _rewards_tied(rewards: list[float]) -> bool:
    return len(rewards) >= 2 and all(item == rewards[0] for item in rewards)


def _finalize_prefix(episode: BehaviorEpisode):
    """Score a live episode once with prefix credit and return that ledger."""
    from actproto.training.rewards_v2 import PREFIX_VERSION

    existing = episode.session.finalized_ledger()
    if existing is not None:
        return existing
    return episode.session.score(completion_reason=episode.completion_reason, version=PREFIX_VERSION)


class BehaviorToolRollout(AgenticRolloutAdapter):
    """Trainer callback that samples a policy and executes train-task tools in-process."""

    def __init__(self, tokenizer, tasks, policy_factory, environment_factory, *, deadline_seconds,
                 max_generated_tokens=256, seed=42, run_id="grpo-tools", chat_template=None,
                 context_limit_reader=None, group_size=2):
        if not _positive_deadline(deadline_seconds):
            raise ValueError("behavior collection requires a hard-deadline executor; only synthetic fixtures are supported")
        if not callable(policy_factory) or not callable(environment_factory):
            raise ValueError("behavior rollout requires a policy and a scoped environment factory")
        super().__init__(tokenizer, self._refuse_smoke_policy, self._refuse_smoke_execute, max_decisions=2)
        self.tasks = {}
        for task in tasks:
            if task.split != "train" or not task.prompt or task.prompt in self.tasks or not task.task_id:
                raise ValueError("training prompts and task IDs must be nonempty and unique")
            self.tasks[task.prompt] = task
        if not self.tasks:
            raise ValueError("training task list is empty")
        self.policy_factory = policy_factory
        self.environment_factory = environment_factory
        self.deadline_seconds = float(deadline_seconds)
        self.max_generated_tokens = max_generated_tokens
        self.chat_template = chat_template
        self.context_limit_reader = context_limit_reader
        self.group_size = group_size
        self.seed = seed
        self.run_id = run_id
        self.last_skipped = False

    @staticmethod
    def _refuse_smoke_policy(prompt_ids, trainer):
        del prompt_ids, trainer
        raise RuntimeError("the study rollout does not use the smoke policy")

    @staticmethod
    def _refuse_smoke_execute(name, arguments):
        del name, arguments
        raise RuntimeError("the study rollout does not use the smoke executor")

    def _episode(self, prompt, index, trainer, draw):
        text = _public_user_text(prompt)
        task = self.tasks.get(text)
        if task is None:
            raise ValueError("public prompt does not match a train task")
        encoded = json.dumps(prompt)
        secret = json.dumps(task.oracle, sort_keys=True)
        if secret and secret in encoded:
            raise ValueError("training prompt leaked private oracle data")
        local_seed = rollout_seed(self.run_id, task.task_id, draw, index, self.seed)
        policy = self.policy_factory(local_seed, trainer)
        limit = self.context_limit_reader(trainer) if self.context_limit_reader is not None else None
        return run_trusted_behavior_episode(
            task, tokenizer=self.tokenizer, policy=policy,
            environment_factory=self.environment_factory,
            deadline_seconds=self.deadline_seconds,
            max_generated_tokens=self.max_generated_tokens,
            chat_template=self.chat_template,
            context_limit=limit,
        )

    def __call__(self, prompts, trainer):
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("trainer must provide a nonempty batch of public prompts")
        group_size = self.group_size
        grouped = group_size >= 2 and len(prompts) >= group_size and len(prompts) % group_size == 0
        episodes = []
        skipped = False
        if grouped:
            from actproto.training.rewards_v2 import ledger_components

            for start in range(0, len(prompts), group_size):
                chunk = prompts[start:start + group_size]
                drawn = [self._episode(chunk[offset], start + offset, trainer, 0) for offset in range(group_size)]
                rewards = [_finalize_prefix(episode).reward for episode in drawn]
                if _rewards_tied(rewards):
                    drawn = [self._episode(chunk[offset], start + offset, trainer, 1) for offset in range(group_size)]
                    rewards = [_finalize_prefix(episode).reward for episode in drawn]
                    if _rewards_tied(rewards):
                        skipped = True
                episodes.extend(drawn)
            components = [
                ledger_components(_finalize_prefix(episode), episode.completion_reason) for episode in episodes
            ]
        else:
            components = None
            for index, prompt in enumerate(prompts):
                episodes.append(self._episode(prompt, index, trainer, 0))
        self.last_skipped = skipped
        result = {key: [] for key in ("prompt_ids", "completion_ids", "logprobs", "env_mask",
                                       "reward_sessions", "completion_reasons")}
        for episode in episodes:
            result["prompt_ids"].append(episode.prompt_ids)
            result["completion_ids"].append(episode.completion_ids)
            result["logprobs"].append(episode.logprobs)
            result["env_mask"].append(episode.env_mask)
            result["reward_sessions"].append(episode.session)
            result["completion_reasons"].append(episode.completion_reason)
        if components is not None:
            result["reward_components"] = components
            result["skipped_tied_batch"] = skipped
        return result


class TrustedRolloutAdapter:
    """Synthetic-only adapter with durable batch and group audit journals.

    A ready event records that the callback prepared to return, not that TRL
    consumed it or optimized its samples. On resume every prepared group is
    resolved and skipped, even if all its records exist and ready was written.
    """
    def __new__(cls, tokenizer, tasks, policy_factory, environment_factory, *, run_id,
                policy_hash, record_sink=None, group_size=2, seed=42, max_generated_tokens=512,
                max_seconds=60, resume_records=None, resume_batch_events=None,
                sampling_mode='behavior', abandonment_sink=None, batch_event_sink=None,
                trainer_callback=False, deadline_seconds=None):
        from actproto.integrations.agenttune_backend import AgenticRolloutAdapter
        import weakref

        if sampling_mode == 'behavior':
            if trainer_callback is not True or not _positive_deadline(deadline_seconds):
                raise ValueError('behavior collection requires a hard-deadline executor; only synthetic fixtures are supported')
            if resume_records or resume_batch_events:
                raise ValueError('deserialized JSON is not execution evidence')
            if max_generated_tokens > 256:
                max_generated_tokens = 256
            return BehaviorToolRollout(
                tokenizer, tasks, policy_factory, environment_factory,
                deadline_seconds=deadline_seconds, max_generated_tokens=max_generated_tokens,
                seed=seed, run_id=run_id,
            )
        if group_size < 2 or not callable(record_sink):
            raise ValueError('group_size >= 2 and a durable record sink are required')
        if sampling_mode not in {'behavior', 'synthetic_fixture'}:
            raise ValueError('invalid sampling mode')
        if sampling_mode != 'synthetic_fixture':
            raise ValueError('behavior collection requires a hard-deadline executor; only synthetic fixtures are supported')
        if not callable(batch_event_sink) or not callable(abandonment_sink):
            raise ValueError('durable batch and group resolution sinks are required')
        task_map = {}
        task_by_id = {}
        for task in tasks:
            if (task.split != 'train' or not task.prompt or task.prompt in task_map
                    or not task.task_id or task.task_id in task_by_id):
                raise ValueError('training prompts and task IDs must be nonempty and unique')
            task_map[task.prompt] = task
            task_by_id[task.task_id] = task
        if not task_map:
            raise ValueError('training task list is empty')

        def resolve_group(group, step, reason):
            task = task_by_id[group['task_id']]
            event = {'schema_version': 'rl_rollout_group_resolution_v1', 'run_id': run_id,
                     'task_id': task.task_id, 'task_hash': task_digest(task), 'policy_hash': policy_hash,
                     'group_index': group['group_index'], 'policy_step': step,
                     'rollout_ids': list(group['rollout_ids']), 'reason': reason}
            if abandonment_sink(event) not in (0, 1):
                raise ValueError('failed to durably resolve rollout group')

        class _TrustedAdapter(AgenticRolloutAdapter):
            def __init__(self):
                self.indices = {prompt: 0 for prompt in task_map}
                self._environments = []
                self._last_step = 0
                rows = {key: row for key, row in (resume_records or {}).items() if row.get('run_id') == run_id}
                if rows and not resume_batch_events:
                    raise ValueError('batch journal required to resume persisted rollout records')
                batches = {}
                for event in resume_batch_events or []:
                    if event.get('run_id') != run_id:
                        continue
                    if event.get('schema_version') != 'rl_rollout_batch_v1':
                        raise ValueError('invalid resume batch event')
                    batch_id, phase = event.get('batch_id'), event.get('phase')
                    if not isinstance(batch_id, str) or phase not in {'prepare', 'ready', 'aborted'}:
                        raise ValueError('invalid resume batch identity')
                    phases = batches.setdefault(batch_id, {})
                    if (phase in phases or (phase == 'prepare' and phases)
                            or (phase != 'prepare' and ('prepare' not in phases or {'ready', 'aborted'} & set(phases)))):
                        raise ValueError('invalid resume batch event order')
                    phases[phase] = event
                seen_groups, seen_rollouts = {}, set()
                for phases in batches.values():
                    prepare = phases.get('prepare')
                    if prepare is None or ('ready' in phases and 'aborted' in phases):
                        raise ValueError('inconsistent resume batch journal')
                    step = prepare.get('policy_step')
                    if type(step) is not int or step < 0 or prepare.get('policy_hash') != policy_hash:
                        raise ValueError('resume batch provenance mismatch')
                    groups = prepare.get('groups')
                    if not isinstance(groups, list) or not groups:
                        raise ValueError('resume batch has no groups')
                    raw = json.dumps([run_id, step, groups], sort_keys=True, separators=(',', ':'))
                    if hashlib.sha256(raw.encode()).hexdigest() != prepare['batch_id']:
                        raise ValueError('resume batch identity mismatch')
                    self._last_step = max(self._last_step, step)
                    for group in groups:
                        task = task_by_id.get(group.get('task_id'))
                        index = group.get('group_index')
                        if task is None or type(index) is not int or index < 0 or group.get('task_hash') != task_digest(task):
                            raise ValueError('resume group task or provenance mismatch')
                        key = (task.task_id, index)
                        expected = [rollout_id(run_id, task.task_id, index, sample) for sample in range(group_size)]
                        if key in seen_groups or group.get('rollout_ids') != expected:
                            raise ValueError('duplicate or invalid resume group')
                        seen_groups[key] = group
                        seen_rollouts.update(expected)
                        for sample, rollout in enumerate(expected):
                            row = rows.get(rollout)
                            if row is None:
                                if 'ready' in phases:
                                    raise ValueError('ready batch is missing rollout records')
                                continue
                            if (row.get('rollout_id') != rollout or row.get('task_id') != task.task_id or row.get('task_hash') != task_digest(task)
                                    or row.get('group_index') != index or row.get('sample_index') != sample
                                    or row.get('policy_hash') != policy_hash or row.get('policy_step') != step
                                    or row.get('batch_id') != prepare['batch_id']
                                    or row.get('sampling_mode') != sampling_mode
                                    or row.get('seed') != rollout_seed(run_id, task.task_id, index, sample, seed)):
                                raise ValueError('resume rollout provenance mismatch')
                        reason = 'ready_consumption_unknown' if 'ready' in phases else 'batch_not_ready'
                        resolve_group(group, step, reason)
                if set(rows) - seen_rollouts:
                    raise ValueError('persisted rollout missing from batch journal')
                for prompt, task in task_map.items():
                    groups = sorted(index for task_id, index in seen_groups if task_id == task.task_id)
                    if groups != list(range(len(groups))):
                        raise ValueError('resume rollout groups are not contiguous')
                    self.indices[prompt] = len(groups) * group_size

            def __call__(self, prompts, trainer):
                if not isinstance(prompts, list) or not prompts:
                    raise ValueError('trainer must provide a nonempty batch of public prompts')
                step = getattr(getattr(trainer, 'state', None), 'global_step', None)
                if type(step) is not int or step < self._last_step:
                    raise ValueError('trainer global step is absent or older than persisted samples')
                if len(prompts) % group_size:
                    raise ValueError('batch does not contain complete rollout groups')
                projected = dict(self.indices)
                groups, assignments = [], []
                for start in range(0, len(prompts), group_size):
                    chunk = prompts[start:start + group_size]
                    if any(not isinstance(prompt, str) for prompt in chunk) or len(set(chunk)) != 1:
                        raise ValueError('consecutive rollout groups must each share one public task')
                    prompt = chunk[0]
                    if prompt not in projected or projected[prompt] % group_size:
                        raise ValueError('unknown task or incomplete prior rollout group')
                    task = task_map[prompt]
                    group_index = projected[prompt] // group_size
                    ids = [rollout_id(run_id, task.task_id, group_index, i) for i in range(group_size)]
                    seeds = {rollout_seed(run_id, task.task_id, group_index, i, seed) for i in range(group_size)}
                    if len(seeds) != group_size:
                        raise ValueError('seed collision in rollout group')
                    groups.append({'task_id': task.task_id, 'task_hash': task_digest(task),
                                   'group_index': group_index, 'rollout_ids': ids})
                    assignments.extend((prompt, group_index, i) for i in range(group_size))
                    projected[prompt] += group_size
                raw = json.dumps([run_id, step, groups], sort_keys=True, separators=(',', ':'))
                batch_id = hashlib.sha256(raw.encode()).hexdigest()
                base = {'schema_version': 'rl_rollout_batch_v1', 'run_id': run_id, 'batch_id': batch_id}
                prepare = {**base, 'phase': 'prepare', 'policy_hash': policy_hash,
                           'policy_step': step, 'groups': groups}
                if batch_event_sink(prepare) != 1:
                    raise ValueError('batch prepare was not durably recorded; resume the journal')
                self.indices = projected
                self._last_step = step
                output = {key: [] for key in ('prompt_ids', 'completion_ids', 'logprobs', 'env_mask',
                                               'execution_traces', 'task_ids', 'rollout_ids')}
                try:
                    for prompt, group_index, sample_index in assignments:
                        task = task_map[prompt]
                        session = new_session(task, environment_factory, allow_unscoped_fixture=True)
                        self._environments = [ref for ref in self._environments if ref() is not None]
                        if any(ref() is session.environment for ref in self._environments):
                            raise ValueError('environment factory reused a rollout environment')
                        try:
                            self._environments.append(weakref.ref(session.environment))
                        except TypeError as exc:
                            raise ValueError('environment must support identity verification') from exc
                        row = collect_episode(task, run_id=run_id, group_index=group_index,
                            sample_index=sample_index, seed=seed, policy_hash=policy_hash,
                            tokenizer=tokenizer, policy_factory=lambda local_seed: policy_factory(local_seed, trainer),
                            environment_factory=environment_factory, session=session,
                            max_generated_tokens=max_generated_tokens, max_seconds=max_seconds,
                            allow_unscoped_fixture=True, fixture_only=True)
                        row['policy_step'] = step
                        row['sampling_mode'] = sampling_mode
                        row['batch_id'] = batch_id
                        if record_sink(deepcopy(row)) != 1:
                            raise ValueError('record sink did not persist a new rollout; resume from existing records')
                        start = row['prompt_length']
                        completion, mask = row['token_ids'][start:], row['policy_mask'][start:]
                        probs = row['sampling_logprobs'][start:]
                        if (not completion or not any(mask)
                                or row['completion_reason'] in {'template_mismatch', 'policy_error', 'timeout', 'policy_budget_overrun'}
                                or any(active and prob is None for active, prob in zip(mask, probs))):
                            raise ValueError('recorded episode cannot be safely passed to the trainer')
                        output['prompt_ids'].append(row['token_ids'][:start])
                        output['completion_ids'].append(completion)
                        output['env_mask'].append(mask)
                        output['logprobs'].append([float(p) if active else 0.0 for p, active in zip(probs, mask)])
                        output['execution_traces'].append(row['trace'])
                        output['task_ids'].append(row['task_id'])
                        output['rollout_ids'].append(row['rollout_id'])
                    if batch_event_sink({**base, 'phase': 'ready'}) != 1:
                        raise ValueError('batch ready status was not durably recorded')
                except Exception:
                    # A missing terminal event remains visibly prepared on crash.
                    # Do not infer that any record in an aborted batch was consumed.
                    batch_event_sink({**base, 'phase': 'aborted'})
                    for group in groups:
                        resolve_group(group, step, 'batch_not_ready')
                    raise
                return output
        return _TrustedAdapter()


def collect_scored_fixture_episode(task: TaskSpec, *, environment_factory: Callable,
                                   reward_version: str = 'verified_success_v2', **kwargs):
    """Collect and score one fixture episode from the same live tool session.

    This API is deliberately fixture-only. Its returned ledger is a live local
    value, not proof that an arbitrary persisted JSON record is trustworthy.
    """
    from actproto.agent.harness import ToolResult
    from actproto.evaluation.verification_v2 import verify
    from actproto.training.rewards_v2 import TrustedRewardSession

    if kwargs.get('fixture_only') is not True or kwargs.get('allow_unscoped_fixture', False):
        raise ValueError('scored collection requires scoped fixtures and fixture_only=True')
    if 'session' in kwargs:
        raise ValueError('caller-supplied sessions are not permitted')
    task = deepcopy(task)
    tool_session = new_session(task, environment_factory, allow_unscoped_fixture=False)

    class _Harness:
        @property
        def final_state(self):
            return tool_session.state

        def execute(self, name, arguments):
            output = tool_session.call(name, arguments)
            return ToolResult(name, deepcopy(arguments), output)

    reward = TrustedRewardSession(task, _Harness())

    class _CollectorSession:
        @property
        def task(self):
            return tool_session.task

        @property
        def unscoped_fixture(self):
            return tool_session.unscoped_fixture

        @property
        def state(self):
            return tool_session.state

        @property
        def events(self):
            return tool_session.events

        def call(self, name, arguments):
            # TrustedRewardSession calls _Harness.execute, which invokes the
            # same ToolSession exactly once and records the identical event.
            return reward.execute(name, arguments)

    row = collect_episode(task, environment_factory=environment_factory,
                          session=_CollectorSession(), **kwargs)
    # The recorder may not be supplied from outside the collector. Reject any
    # divergence between its directly captured calls and the reward capture.
    calls = [event for event in tool_session.events if event.get('kind') == 'tool_call']
    if calls != reward.captured_tool_events():
        raise ValueError('live reward capture does not match the tool session')
    for event in tool_session.events:
        if event.get('kind') != 'tool_call':
            reward.invalid_attempt(str(event.get('kind', 'invalid event')))
    ledger = reward.score(completion_reason=row['completion_reason'], version=reward_version)
    live_trace = ExecutionTrace(task.task_id, deepcopy(tool_session.events), row['completion_reason'], 'harness',
        final_answer=tool_session.state.get('answer'), final_state=deepcopy(tool_session.state))
    verdict = verify(task, live_trace)
    if (ledger.verified_success != verdict.verified_success_v2 or
            ledger.answer_success != verdict.answer_success or
            ledger.execution_compliance != verdict.execution_compliance):
        raise ValueError('live reward and collector verification disagree')
    if row['trace']['events'] != live_trace.events or row['trace']['final_state'] != live_trace.final_state:
        raise ValueError('collector record does not match live execution')
    return row, ledger
