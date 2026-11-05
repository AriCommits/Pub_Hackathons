"""Validated training primitives for v2 SFT; no historical paths are modified."""
from __future__ import annotations

import json
import math
from pathlib import Path

from actproto.training.sft_masks import encode_checked, pad_batch


def read_rows(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('training data is empty')
    for row in rows:
        if row.get('split') != 'train':
            raise ValueError('only explicit train rows are permitted')
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'assistant':
            raise ValueError('missing assistant finish message')
        content = messages[-1].get('content', '')
        try:
            call = json.loads(content.removeprefix('<tool_call>').removesuffix('</tool_call>'))
        except (ValueError, AttributeError) as exc:
            raise ValueError('invalid final tool call') from exc
        if not isinstance(call, dict) or call.get('name') != 'finish':
            raise ValueError('training trajectory must end in finish')
    return rows


def preflight(tokenizer, rows, config):
    from actproto.agent.prompts import messages_for_template, system_prompt
    if not 1 <= config['epochs'] <= 2 or config['max_length'] < 1:
        raise ValueError('invalid epochs or max length')
    if config['dtype'] != 'bfloat16' or config['quantization'] != 'none':
        raise ValueError('only unquantized bfloat16 is supported; no QLoRA claim')
    encoded, excluded = [], []
    for index, row in enumerate(rows):
        messages = row['messages']
        if any(set(message) != {'role', 'content'} or message['role'] not in
               {'system', 'user', 'assistant', 'tool'} for message in messages):
            raise ValueError('training messages must contain only public role/content')
        if any(message['role'] == 'system' for message in messages):
            raise ValueError('system prompt is supplied by the shared training contract')
        public_messages = messages_for_template([{'role': 'system', 'content': system_prompt()}, *messages])
        try:
            value = encode_checked(tokenizer, public_messages, config['max_length'], config['template_name'])
        except ValueError as exc:
            if 'exceeds max_length' not in str(exc):
                raise
            excluded.append({'index': index, 'reason': str(exc)})
            continue
        encoded.append(value)
    if not encoded:
        raise ValueError('no trainable rows remain after length checks')
    encoded.sort(key=lambda row: len(row['input_ids']))
    return encoded, {'input_rows': len(rows), 'accepted_rows': len(encoded),
                     'excluded_rows': len(excluded), 'excluded': excluded,
                     'lengths': [len(row['input_ids']) for row in encoded]}


def selection_key(metrics):
    """Higher is better; NaN and missing data cannot win selection."""
    names = ('macro_verified_success_v2', 'argument_correctness', 'invalid_calls', 'generated_tokens')
    values = [float(metrics[name]) for name in names]
    if any(not math.isfinite(x) for x in values):
        raise ValueError('selection metrics must be finite')
    return values[0], values[1], -values[2], -values[3]


def select_checkpoint(candidates):
    if not candidates:
        raise ValueError('no validation checkpoint candidates')
    if any(candidate.get('split') != 'val' for candidate in candidates):
        raise ValueError('checkpoint selection is restricted to validation')
    return max(candidates, key=lambda candidate: selection_key(candidate['metrics']))


def train(tokenizer, encoded, config, run_dir, evaluator, heartbeat=lambda: None):
    """Train and evaluate each saved epoch using the live model, on validation only."""
    if evaluator is None:
        raise ValueError('a shared validation evaluator is required before model loading')
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for real training')
    output = Path(run_dir)
    candidates = []

    class ValidationOnSave(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            heartbeat()
            return control

        def on_save(self, args, state, control, **kwargs):
            checkpoint = output / f'checkpoint-{state.global_step}'
            metrics = evaluator(str(checkpoint), split='val', model=kwargs['model'], tokenizer=tokenizer)
            item = {'checkpoint': str(checkpoint), 'split': 'val', 'metrics': metrics}
            selection_key(metrics)
            candidates.append(item)
            (output / 'selection.json').write_text(json.dumps({'candidates': candidates,
                'selected': select_checkpoint(candidates)}, indent=2), encoding='utf-8')
            return control

    model = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['revision'],
                                                dtype=torch.bfloat16)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=config['rank'], lora_alpha=config['alpha'],
        lora_dropout=config['dropout'], target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM'))
    import inspect
    grouping = ({'train_sampling_strategy': 'group_by_length'}
                if 'train_sampling_strategy' in inspect.signature(TrainingArguments).parameters
                else {'group_by_length': True})
    args = TrainingArguments(output_dir=str(output), learning_rate=config['learning_rate'],
        num_train_epochs=config['epochs'], per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        save_strategy='epoch', logging_steps=10, bf16=True, report_to='none', seed=config['seed'],
        remove_unused_columns=False, **grouping)

    def collate(batch):
        return {k: torch.tensor(v, dtype=torch.long) for k, v in
                pad_batch(batch, tokenizer.pad_token_id).items()}

    trainer = Trainer(model=model, args=args, train_dataset=encoded, data_collator=collate,
                      callbacks=[ValidationOnSave()])
    result = trainer.train()
    if trainer.state.global_step < 1 or not candidates or not math.isfinite(result.training_loss):
        raise RuntimeError('training did not produce a finite updated checkpoint and validation')
    trainer.save_state()
    selected = select_checkpoint(candidates)
    if not Path(selected['checkpoint'], 'adapter_config.json').is_file():
        raise RuntimeError('selected checkpoint lacks a saved adapter')
    return {'global_step': trainer.state.global_step, 'training_loss': result.training_loss,
            'selected': selected, 'dtype': config['dtype'], 'quantization': config['quantization']}
