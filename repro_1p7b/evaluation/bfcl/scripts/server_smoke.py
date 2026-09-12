#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

port = os.environ.get("VLLM_PORT", "1053")
client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")

plain_prompt = "<|im_start|>user\nReply with exactly: BFCL server ready.<|im_end|>\n<|im_start|>assistant\n"
plain = client.completions.create(
    model=args.model_path,
    prompt=plain_prompt,
    temperature=0.0,
    max_tokens=64,
)
plain_text = plain.choices[0].text
if not plain_text.strip():
    raise RuntimeError("ordinary generation returned an empty string")

handler = QwenFCHandler("Qwen/Qwen3-1.7B-FC", temperature=0.7)
handler.model_path_or_id = args.model_path
functions = [{
    "name": "get_weather",
    "description": "Get the weather for one city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]
messages = [{"role": "user", "content": "Use get_weather to check Singapore. You must call the tool."}]
tool_prompt = handler._format_prompt(messages, functions)
tool_response = client.completions.create(
    model=args.model_path,
    prompt=tool_prompt,
    temperature=0.7,
    max_tokens=256,
)
raw_tool_text = tool_response.choices[0].text
parsed_calls = handler._extract_tool_calls(raw_tool_text)
decoded_calls = handler.decode_execute(raw_tool_text)
if not parsed_calls or not decoded_calls:
    raise RuntimeError(f"official QwenFCHandler could not parse a tool call: {raw_tool_text!r}")

assistant_message = {
    "role": "assistant",
    "content": "",
    "tool_calls": parsed_calls,
}
refill_messages = messages + [
    assistant_message,
    {"role": "tool", "content": '{"city":"Singapore","condition":"sunny"}'},
]
refill_prompt = handler._format_prompt(refill_messages, functions)
if "<tool_response>" not in refill_prompt:
    raise RuntimeError("multi-turn tool result was not rendered as <tool_response>")

output = {
    "model_path": args.model_path,
    "ordinary_generation_nonempty": True,
    "ordinary_generation": plain_text,
    "tool_call_raw": raw_tool_text,
    "tool_calls_parsed": parsed_calls,
    "tool_calls_decoded": decoded_calls,
    "tool_response_refill_present": True,
}
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(output, ensure_ascii=False))
