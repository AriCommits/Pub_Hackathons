"""Collect trusted tool observations without exposing private TaskSpec fields."""
from __future__ import annotations
import json
from time import perf_counter

from actproto.agent.inference_v2 import Generation, parse_single_call
from actproto.agent.prompts import system_prompt
from actproto.experiments.contracts import ExecutionTrace
from actproto.tools.schemas import validate_args


def rollout(task, env, policy, *, interventions=()):
    if any(item != 'retry_parse' for item in interventions) or len(interventions) != len(set(interventions)):
        raise ValueError('unsupported or duplicate intervention')
    if env.identifier != task.environment_id or env.digest != task.fixture_hash:
        raise ValueError('fixture environment mismatch')
    messages = [{'role': 'system', 'content': system_prompt()}, {'role': 'user', 'content': task.prompt}]
    events, spans, state, tokens = [], [], {}, 0
    start = perf_counter()
    completion = 'budget_exhausted'
    for _ in range(task.call_budget):
        call = None
        for attempt in range(2 if 'retry_parse' in interventions else 1):
            prompt = [dict(message) for message in messages]
            if attempt:
                prompt.extend([{'role': 'assistant', 'content': generated.text},
                               {'role': 'user', 'content': 'Your previous response was invalid. Emit exactly one valid tool call.'}])
            generated = policy(prompt)
            if not isinstance(generated, Generation) or generated.tokens < 0:
                raise ValueError('policy must return Generation with nonnegative token count')
            tokens += generated.tokens
            spans.append({'raw': generated.text, 'tokens': generated.tokens, 'truncated': generated.truncated,
                          'source': 'model', 'interventions': ['retry_parse'] if attempt else []})
            if generated.truncated:
                events.append({'kind': 'truncation', 'raw': generated.text})
                completion = 'truncated'
                break
            try:
                call = parse_single_call(generated.text)
                valid, reason = validate_args(call['name'], call['arguments'])
                if not valid:
                    raise ValueError(reason)
            except ValueError as exc:
                events.append({'kind': 'parse_error', 'raw': generated.text, 'error': str(exc),
                               'intervention': 'retry_parse' if attempt == 0 and 'retry_parse' in interventions else None})
                completion = 'parse_error'
                call = None
                continue
            break
        if call is None:
            break
        tool, args = call['name'], call['arguments']
        before = dict(state)
        output = env.call(task.environment_id, tool, args)
        if tool in ('finish', 'no_tool') and output.get('ok'):
            state.update({'answer': output['data']['answer'], 'done': True})
        events.append({'kind': 'tool_call', 'source': 'harness', 'tool': tool, 'arguments': args,
                       'output': output, 'state_before': before, 'state_after': dict(state)})
        messages.append({'role': 'assistant', 'content': generated.text})
        if tool in ('finish', 'no_tool') and output.get('ok'):
            completion = 'finish'
            break
        messages.append({'role': 'tool', 'content': json.dumps(output, sort_keys=True)})
    return ExecutionTrace(task.task_id, events, completion, 'harness', model_spans=spans,
                          generated_tokens=tokens, model_invocations=len(spans),
                          final_answer=state.get('answer'), final_state=state), perf_counter() - start
