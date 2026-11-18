"""Results browser for ActProto-LoRA. Does not load Tiny Aya."""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

RESULTS = json.loads((Path(__file__).parent / "results.json").read_text(encoding="utf-8"))

SCHEMA_EXAMPLE = """user: How many sales were recorded for item2000001 in 2020? Return only the requested value as the final answer.
assistant: <tool_call>{"name": "query_database", "arguments": {"product": "item2000001", "year": 2020}}</tool_call>
tool: {"ok": true, "data": {"rows": [{"sales": 1234}], "count": 1}}
assistant: <tool_call>{"name": "finish", "arguments": {"answer": "1234"}}</tool_call>"""


def render() -> str:
    return "\n".join(
        [
            RESULTS["claim"],
            "",
            f"Validation macro verified success: {RESULTS['val_macro_verified_success']} on {RESULTS['val_tasks']} tasks.",
            RESULTS["val_source"],
            f"Stored improvement field (not a before-score): {RESULTS['improvement_field']}",
            "",
            f"GRPO before the sampling fix: reward {RESULTS['grpo_prefix_reward']}.",
            f"GRPO after the sampling fix: reward {RESULTS['grpo_after_fix_reward']}.",
            RESULTS["grpo_note"],
            "",
            f"Article: {RESULTS['article']}",
            f"Dataset: {RESULTS['dataset']}",
            f"Space: {RESULTS['space']}",
            f"GitHub: {RESULTS['github']}",
            "",
            "Schema illustration, not a test item:",
            SCHEMA_EXAMPLE,
        ]
    )


with gr.Blocks() as demo:
    gr.Textbox(value=render(), label="ActProto-LoRA", lines=24)


if __name__ == "__main__":
    demo.launch()
