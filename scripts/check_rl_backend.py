"""Probe real AgentTune construction and the locally cached Tiny Aya template."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true", help="verify the CPU fixture only")
    args = parser.parse_args()
    from actproto.integrations.agenttune_backend import AgenticRolloutAdapter, create_trainer, episode_token_mask, run_fixture_episode, validate_versions
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    import torch

    versions = validate_versions()
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "world": 4}
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_layer=1, n_head=1,
                                      n_embd=16, n_positions=64, eos_token_id=2, pad_token_id=0))
    def reward(completions, **kwargs):
        return [0.0] * len(completions)
    with tempfile.TemporaryDirectory() as output:
        wrapper = create_trainer(model=model, processing_class=tokenizer, reward_funcs=reward,
            train_dataset=Dataset.from_dict({"prompt": ["hello", "world"]}), output_dir=output,
            num_generations=2, per_device_train_batch_size=2, gradient_accumulation_steps=1,
            bf16=False, fp16=False, report_to="none", use_cpu=True, beta=0.0)
        wrapper.setup_data()
        wrapper.setup_trainer()
        if wrapper.trainer is None:
            raise RuntimeError("AgentTune did not construct its TRL trainer")
        trainer_type = type(wrapper.trainer).__module__ + "." + type(wrapper.trainer).__name__
        # Check the installed wrapper and TRL really propagate an environment mask.
        from agenttune.core.backend_factory import create_agentic_trainer
        def fake_rollout(prompts, trainer):
            n = len(prompts)
            return {"prompt_ids": [[3]] * n, "completion_ids": [[3, 4, 3]] * n,
                    "logprobs": [[0.0, 0.0, 0.0]] * n, "env_mask": [[1, 0, 1]] * n}
        def distinct_rewards(completions, **kwargs):
            return [float(i % 2) for i in range(len(completions))]
        masked_wrapper = create_agentic_trainer("grpo", backend="trl", model=model,
            processing_class=tokenizer, reward_funcs=distinct_rewards,
            train_dataset=Dataset.from_dict({"prompt": ["hello", "world"]}),
            rollout_func=fake_rollout, output_dir=output, num_generations=2,
            per_device_train_batch_size=2, gradient_accumulation_steps=1,
            bf16=False, fp16=False, report_to="none", use_cpu=True, beta=0.0)
        masked_wrapper.setup_data()
        masked_wrapper.setup_trainer()
        scored = masked_wrapper.trainer._generate_and_score_completions(
            [{"prompt": "hello"}, {"prompt": "world"}])
        if scored["tool_mask"].tolist() != [[1, 0, 1], [1, 0, 1]]:
            raise RuntimeError("TRL did not preserve the custom rollout environment mask")
    aya = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global", local_files_only=True)
    ids, mask = episode_token_mask(aya, [{"role": "user", "content": "calculate 1+1"}],
        '<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}}</tool_call>',
        "2", '<tool_call>{"name":"finish","arguments":{"answer":"2"}}</tool_call>')
    fixture_actions = iter([
        '<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}}</tool_call>',
        '<tool_call>{"name":"finish","arguments":{"answer":"2"}}</tool_call>',
    ])
    fixture_prompts = []
    def fixture_generate(prompt_ids):
        fixture_prompts.append(prompt_ids)
        action = next(fixture_actions)
        return action, aya.encode(action, add_special_tokens=False)
    fixture = run_fixture_episode(aya, [{"role": "user", "content": "calculate 1+1"}],
                                  fixture_generate, lambda name, arguments: "2")
    if len(fixture_prompts) != 2 or fixture["finish"]["name"] != "finish":
        raise RuntimeError("Fixture loop failed to terminate on finish")
    if "Tool result:\n2" not in aya.decode(fixture_prompts[1]):
        raise RuntimeError("Fixture did not feed the observation into the second prompt")
    integrated_prompts = []
    def integrated_generate(prompt_ids, trainer):
        integrated_prompts.append(list(prompt_ids))
        action = [
            '<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"answer":"2"}}</tool_call>',
        ][(len(integrated_prompts) - 1) % 2]
        sampled = aya.encode(action, add_special_tokens=False)
        return action, sampled, [-0.5] * len(sampled)  # fixed probe policy, not a model sample
    adapter = AgenticRolloutAdapter(aya, integrated_generate, lambda name, arguments: "2")
    tiny_aya_vocab_model = GPT2LMHeadModel(GPT2Config(vocab_size=len(aya), n_layer=1, n_head=1,
        n_embd=8, n_positions=1024, bos_token_id=aya.bos_token_id,
        eos_token_id=aya.eos_token_id, pad_token_id=aya.pad_token_id))
    with tempfile.TemporaryDirectory() as output:
        integrated = create_trainer(model=tiny_aya_vocab_model, processing_class=aya,
            reward_funcs=lambda completions, **kwargs: [float(i % 2) for i in range(len(completions))],
            train_dataset=Dataset.from_dict({"prompt": ["calculate", "calculate"]}),
            rollout_func=adapter, output_dir=output, num_generations=2,
            per_device_train_batch_size=2, gradient_accumulation_steps=1,
            bf16=False, fp16=False, report_to="none", use_cpu=True, beta=0.0)
        integrated.setup_data()
        integrated.setup_trainer()
        scored = integrated.trainer._generate_and_score_completions(
            [{"prompt": "calculate"}, {"prompt": "calculate"}])
        if len(integrated_prompts) != 4 or "Tool result:\n2" not in aya.decode(integrated_prompts[1]):
            raise RuntimeError("Integrated tool loop did not supply its observation or stop at finish")
        if scored["tool_mask"].shape != scored["completion_ids"].shape:
            raise RuntimeError("Integrated policy mask does not align with completion IDs")
        if int(scored["tool_mask"][0].sum()) != sum(len(aya.encode(x, add_special_tokens=False)) for x in [
            '<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"answer":"2"}}</tool_call>',
        ]):
            raise RuntimeError("Integrated trainer lost the policy mask")
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

    class FakeEngine:
        def __init__(self):
            self.seen = []
        def _get_tokenizer(self):
            return aya
        def generate(self, prompts, tools, gen_cfg):
            self.seen.append([dict(message) for message in prompts])
            decisions = [
                '<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}}</tool_call>',
                '<tool_call>{"name":"finish","arguments":{"answer":"2"}}</tool_call>',
                'done',
            ]
            completion = decisions[min(len(self.seen) - 1, len(decisions) - 1)]
            return {"completions": [completion], "metadata": {"completion_ids": aya.encode(completion, add_special_tokens=False)}}

    def calculator(expression: str) -> str:
        """Calculate an expression.

        Args:
            expression: The expression.
        """
        return "2"

    def finish(answer: str) -> str:
        """Finish with an answer.

        Args:
            answer: The answer.
        """
        return answer

    engine = FakeEngine()
    rollout = create_rollout_fn(rollout_engine=engine, tools=[calculator, finish], max_steps=3,
                                tool_result_format="Tool result:\n{content}")
    result = rollout(["calculate 1+1"])
    if len(engine.seen) < 2 or not any("Tool result:\n2" in m.get("content", "") for m in engine.seen[1]):
        raise RuntimeError("Actual tool observation was not inserted into the second decision")
    mask_zero_count = result["env_mask"][0].count(0)
    if not mask_zero_count:
        raise RuntimeError("Backend did not mask any tool-result tokens")
    print(json.dumps({"versions": versions, "trainer": trainer_type, "tokens": len(ids),
                      "policy_tokens": sum(mask), "cuda_available": torch.cuda.is_available(),
                      "rollout_mask_verified": False, "fake_engine_decisions": len(engine.seen), "raw_result_masked_tokens": mask_zero_count, "fixture_decisions": len(fixture_prompts), "fixture_policy_tokens": sum(fixture["policy_mask"]), "custom_mask_reaches_trainer": True, "integrated_fixture_decisions": len(integrated_prompts)}, indent=2))
    if args.fixture:
        return 0
    # A constructed trainer and template probe do not establish backend loss masking.
    print("INCOMPLETE: actual tool-loop loss mask and GPU support remain unverified", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Backend probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
