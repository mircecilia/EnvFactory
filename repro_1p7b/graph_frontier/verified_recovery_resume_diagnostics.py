"""Observation-only resume of the immutable GPU v3 collector.

Keep the original collector, protocol, equality tests, and trajectories intact.
Persist each completed arm before the original paired assertions can fail.
No exception normalization, response caching, task exclusion or retry-until-PASS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def first_difference(left, right, path='$'):
    if type(left) is not type(right):
        return {'path': path, 'left': left, 'right': right, 'reason': 'TYPE'}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {'path': path, 'left_keys': sorted(left), 'right_keys': sorted(right), 'reason': 'KEYS'}
        for key in left:
            difference = first_difference(left[key], right[key], path + '.' + key)
            if difference is not None:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return {'path': path, 'left_length': len(left), 'right_length': len(right), 'reason': 'LENGTH'}
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f'{path}[{index}]')
            if difference is not None:
                return difference
    elif left != right:
        return {'path': path, 'left': left, 'right': right, 'reason': 'VALUE'}
    return None


def observe_collect(original, directory, write, protocol_hash):
    """Returns the exact original result object; only adds independent files."""
    async def observed(task, entry, row, gold, schemas, policy, config, arm):
        result = await original(task, entry, row, gold, schemas, policy, config, arm)
        path = directory / task / f'{arm}.json'
        if path.exists():
            raise RuntimeError(f'refusing to overwrite diagnostic arm:{path}')
        write(path, {'task_id': task, 'arm': arm, 'collection_protocol_sha256': protocol_hash,
                     'observation_only': True, 'result': result})
        return result
    return observed


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['worker', 'check'])
    parser.add_argument('--shard', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    from repro_1p7b.graph_frontier import verified_recovery_gpu_v3 as v3
    v3.install()
    c = v3.v.c
    # Also verifies package versions and all frozen source files, not only model.
    v3.protocol()
    plan, parent, lock = c.verify_protocol()
    digest = file_hash(v3.RUN / 'collection_protocol.json')
    directory = v3.RUN / 'resume_diagnostics' / args.attempt
    manifest_path = directory / 'diagnostic_driver.json'
    manifest = {'collection_protocol_sha256': digest, 'driver_sha256': file_hash(Path(__file__)),
                'behavior_change': False, 'attempt': args.attempt, 'task_exclusion': False,
                'retry_policy': 'one attempt; exact original paired assertions; stop on failure'}
    c.feasibility.immutable(manifest_path, manifest)
    if args.command == 'check':
        print(json.dumps({'status': 'FROZEN_PROTOCOL_VERIFIED', **manifest}), flush=True)
        return
    c.collect = observe_collect(c.collect, directory, c.atomic_json, digest)
    try:
        c.worker(args.shard, args.limit, 4)
    except Exception as exc:
        differences = []
        for task_dir in directory.iterdir():
            if not task_dir.is_dir():
                continue
            paths = [task_dir / f'{arm}.json' for arm in ('FD', 'RECOVERY')]
            if not all(path.is_file() for path in paths):
                continue
            arms = [json.loads(path.read_text())['result'] for path in paths]
            for a, b in zip(arms[0]['steps'], arms[1]['steps']):
                if b['prior_interventions']:
                    break
                for field in ('model_message', 'state_before_action', 'state_after_action',
                              'typed_events', 'verifier'):
                    difference = first_difference(a[field], b[field])
                    if difference is not None:
                        differences.append({'task_id': task_dir.name, 'request_index': a['step_index'],
                                            'field': field, 'difference': difference})
        out = {'status': 'FAILED_NOT_RELAXED', 'exception': f'{type(exc).__name__}:{exc}',
               'collection_protocol_sha256': digest, 'shard': args.shard, 'differences': differences}
        c.atomic_json(directory / f'failure_shard{args.shard}_{os.getpid()}.json', out)
        print(json.dumps(out, ensure_ascii=False), flush=True)
        raise


if __name__ == '__main__':
    main()
