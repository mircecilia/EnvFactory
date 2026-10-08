"""GPU SGLang adapter for the frozen same-state recovery collector.

CPU artifacts and their protocol stay immutable. GPU results use a separate
namespace; no verifier, intervention, cost, state or statistical gate changes.
"""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from repro_1p7b.graph_frontier import verified_recovery_collect as c
from repro_1p7b.graph_frontier import verified_recovery_metrics as m
from repro_1p7b.graph_frontier import collect_preference_rollouts as api
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable

RUN = c.feasibility.RUN / 'gpu_sglang_v1'
ENGINE = 'SGLANG_GPU_BF16_QWEN3_QWEN25'


def canonical_message(message, seed):
    result = copy.deepcopy(message)
    for i, call in enumerate(result.get('tool_calls') or []):
        call['id'] = f'gpu_{seed}_{i}'
        arguments = call['function']['arguments']
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError('non-JSON-object tool arguments')
        call['function']['arguments'] = arguments
    return result


class GPUPolicy:
    def __init__(self, model, threads=4):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
        self.endpoint = os.environ['VERIFIED_RECOVERY_ENDPOINT']
        self.cache = {}

    def generate(self, messages, schemas, seed, config):
        ids = self.tokenizer.apply_chat_template(messages, tools=schemas, tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
        if len(ids) + config['max_tokens_per_turn'] > 32768:
            raise ValueError('frozen context limit exceeded')
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        key = (digest, seed, config['max_tokens_per_turn'])
        if key in self.cache:
            message, finish, audit = copy.deepcopy(self.cache[key])
            audit.update(cached_identical_fresh_input=True,
                         source_inference_seconds=audit['model_seconds'], model_seconds=0.0)
            return message, finish, audit
        started = time.monotonic()
        message, _, usage, finish = api.generate_action(self.endpoint, 'dynamic-v1', messages,
            schemas, seed=seed, temperature=config['temperature'], max_tokens=config['max_tokens_per_turn'])
        message = canonical_message(message, seed)
        audit = {'input_token_ids_sha256': digest, 'prompt_tokens': len(ids),
            'completion_tokens': usage.get('completion_tokens'), 'usage': usage,
            'model_seconds': time.monotonic()-started, 'device': 'cuda', 'dtype': 'bfloat16',
            'engine': ENGINE, 'parser': 'SGLang server qwen25', 'seed': seed,
            'cached_identical_fresh_input': False,
            'token_hash_scope': 'local locked tokenizer rendering; server usage recorded independently'}
        self.cache[key] = copy.deepcopy((message, finish, audit))
        return message, finish, audit


def install():
    c.RUN = RUN
    m.RUN = RUN
    c.CPUPolicy = GPUPolicy
    original_collect = c.collect
    async def gpu_collect(*args, **kwargs):
        result = await original_collect(*args, **kwargs)
        result['policy_engine'] = ENGINE
        return result
    c.collect = gpu_collect
    original_write = m.atomic_json
    def gpu_write(path, value):
        if path.name.endswith('_metrics.json') or path.name == 'paired_metrics.json':
            value.update(no_gpu=False, gpu_authorized_by_user=True, no_training=True, policy_engine=ENGINE)
        original_write(path, value)
    m.atomic_json = gpu_write


def protocol():
    f = c.feasibility
    for name in ('subset_plan.json', 'subset_preflight_summary.json'):
        f.immutable(RUN/name, f.read(f.RUN/name))
    original_immutable = f.immutable
    def write(path, value):
        if path == RUN/'collection_protocol.json':
            value = copy.deepcopy(value)
            value.update(schema='verified_recovery_gpu_sglang_paired_v1', no_gpu=False,
                gpu_authorized_by_user=True, engine=ENGINE,
                paired_sampling='2 isolated MCP workers, GPU0/1 one SGLang replica each; same task FD then Recovery',
                kv_cache='SGLang standard radix cache; no CPU KV implementation used',
                memoization='identical fresh tokenized input/seed/budget within paired task only; never CPU/historical output')
            value['code_sha256'][Path(__file__).name] = sha256(Path(__file__))
            value['code_sha256'][Path(api.__file__).name] = sha256(Path(api.__file__))
            value['server_protocol'] = {
                'context_length':32768, 'dtype':'bfloat16', 'temperature':0,
                'top_p':.95, 'enable_thinking':False, 'tool_parser':'qwen25',
                'reasoning_parser':'qwen3', 'attention_backend':'triton',
                'sampling_backend':'pytorch', 'cuda_graph':False,
                'mem_fraction_static':.60, 'ports':[8260,8261]}
        original_immutable(path, value)
    f.immutable = write
    try:
        c.make_collection_protocol()
    finally:
        f.immutable = original_immutable
    return f.read(RUN/'collection_protocol.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['protocol','worker','metrics'])
    parser.add_argument('--shard', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--expected', type=int)
    args = parser.parse_args()
    install()
    if args.command == 'protocol':
        print(stable(protocol()))
    elif args.command == 'worker':
        c.worker(args.shard, args.limit, 4)
    else:
        m.summarize(args.expected)
