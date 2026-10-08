"""User-authorized network attrition; original 399 protocol stays FAILED.

No response normalization, fake HTTP results, or verifier changes. Preserve
the original worker/paired assertions; skip only execution-evidenced network
failures and summarize the available cohort in a separate output namespace.
"""
import argparse
import copy
import json
from pathlib import Path

NETWORK_MARKERS = (
    'NameResolutionError', 'Temporary failure in name resolution',
    'Network is unreachable', 'No route to host', 'Failed to resolve',
    'ConnectTimeout', 'ReadTimeout', 'Connection timed out',
    '[Errno 101]', '[Errno 113]', '[Errno 110]',
)


def is_network_error(error):
    if not isinstance(error, dict):
        return False
    message = str(error.get('message', ''))
    return any(marker in message for marker in NETWORK_MARKERS)


def network_events(episode):
    evidence = []
    for arm_name, arm in episode.get('arms', {}).items():
        for step in arm.get('steps', []):
            for event in step.get('typed_events', []):
                if event.get('execution_success') is False and is_network_error(event.get('exception')):
                    evidence.append({'arm': arm_name, 'tool': event['tool_name'],
                                     'exception': event['exception']})
        for target in arm.get('targets', []):
            if is_network_error(target.get('inline_exception')):
                evidence.append({'arm': arm_name, 'tool': target.get('gold_action'),
                                 'exception': target['inline_exception']})
    return evidence


def failure_is_network(exc, records):
    evidence = [r for r in records if r.get('success') is False and is_network_error(r.get('exception'))]
    if not evidence:
        return False
    # Never excuse hash/schema/model/state/assertion failures using unrelated HTTP noise.
    if isinstance(exc, AssertionError):
        return str(exc) == 'paired response drift'
    return (isinstance(exc, RuntimeError)
            and str(exc) == 'independent same-prestate gold target validation failed'
            and any(r['role'] in ('TARGET_VALIDATION', 'TARGET_PREFIX_REPLAY') for r in evidence))


def setup():
    from repro_1p7b.graph_frontier import verified_recovery_gpu_v3 as v3
    v3.install()
    c = v3.v.c
    v3.protocol()  # All original file/package/model hashes still must match.
    plan, parent, lock = c.verify_protocol()
    out = v3.RUN / 'network_skip_v1'
    return v3, c, plan, parent, lock, out


def prepare(v3, c, plan, out):
    frozen = c.feasibility.read(v3.RUN / 'collection_protocol.json')
    retained, skipped = {}, {}
    for task in plan['task_ids']:
        path = v3.RUN / 'paired_episodes' / f'{task}.json'
        if not path.exists():
            continue
        e = c.feasibility.read(path)
        assert e['collection_protocol_sha256'] == c.sha256(v3.RUN / 'collection_protocol.json')
        if e['status'] == 'COMPLETE':
            evidence = network_events(e)
            if evidence:
                skipped[task] = {'evidence': evidence, 'source': str(path), 'source_sha256': c.sha256(path)}
            else:
                retained[task] = c.sha256(path)
        else:
            arms = {}
            for arm in ('FD', 'RECOVERY'):
                candidates = sorted(v3.RUN.glob(f'resume_diagnostics/*/{task}/{arm}.json'))
                assert len(candidates) == 1, f'failed task lacks unambiguous raw diagnostic:{task}'
                diag = c.feasibility.read(candidates[0])
                assert diag['collection_protocol_sha256'] == e['collection_protocol_sha256']
                arms[arm] = diag['result']
            evidence = network_events({'arms': arms})
            assert evidence and e['exception'] == 'AssertionError:paired response drift'
            skipped[task] = {'evidence': evidence, 'source': str(path), 'source_sha256': c.sha256(path),
                             'preserved_diagnostic_arms': arms}
    value = {'schema': 'available_network_skip_feasibility_v1',
             'parent_collection_protocol_sha256': c.sha256(v3.RUN / 'collection_protocol.json'),
             'parent_subset_plan_sha256': c.sha256(c.feasibility.RUN / 'subset_plan.json'),
             'driver_sha256': c.sha256(Path(__file__)), 'method_protocol': frozen,
             'rule': 'exclude task with actual tool execution DNS/network/timeout evidence; no replacement; never PASS unavailable execution',
             'user_authorized': True, 'user_authorization': '2026-10-08 skip existing network problems and silently skip future such problems',
             'network_markers': list(NETWORK_MARKERS), 'retained_before': retained,
             'initial_skipped': skipped, 'planned_parent_tasks': plan['task_ids'],
             'parent_399_gate': 'FAILED_PRESERVED', 'training_allowed': False,
             'scope': 'available-case feasibility with network attrition, NOT full399 determinism PASS or benchmark'}
    c.feasibility.immutable(out / 'scope_protocol.json', value)
    print(c.stable({'status': 'NETWORK_SKIP_SCOPE_PREPARED', 'retained': len(retained), 'skipped': len(skipped)}), flush=True)
    return value


def checked(v3, c, plan, out):
    scope = c.feasibility.read(out / 'scope_protocol.json')
    assert scope['driver_sha256'] == c.sha256(Path(__file__))
    assert scope['parent_collection_protocol_sha256'] == c.sha256(v3.RUN / 'collection_protocol.json')
    assert scope['parent_subset_plan_sha256'] == c.sha256(c.feasibility.RUN / 'subset_plan.json')
    assert scope['planned_parent_tasks'] == plan['task_ids']
    for task, digest in scope['retained_before'].items():
        assert c.sha256(v3.RUN / 'paired_episodes' / f'{task}.json') == digest
    for info in scope['initial_skipped'].values():
        assert c.sha256(Path(info['source'])) == info['source_sha256']
    return scope


def worker(v3, c, plan, out, shard):
    scope = checked(v3, c, plan, out)
    records = {}
    base = c.CountedEpisode
    class ObservedEpisode(base):
        async def call(self, name, arguments):
            ledger = records.setdefault(self.task_id, [])
            record = {'index': len(ledger) + 1, 'role': c.ROLE.get(), 'tool': name,
                      'arguments': copy.deepcopy(arguments)}
            ledger.append(record)
            response, ok, error = await super().call(name, arguments)
            record.update(response=response, success=ok, exception=error)
            c.atomic_json(out / 'call_audit' / f'{self.task_id}.json', {'task_id': self.task_id, 'calls': ledger})
            return response, ok, error
    c.CountedEpisode = ObservedEpisode
    v3.v.SourceOrderedEpisode = ObservedEpisode  # Paid independent validation uses same observer.
    original = c.collect
    async def observed(*args, **kwargs):
        result = await original(*args, **kwargs)
        c.atomic_json(out / 'arm_audit' / args[0] / f'{args[7]}.json', {'result': result})
        return result
    c.collect = observed
    # Original worker skips persisted artifacts. A network failure is never retried.
    for _ in range(len(plan['task_ids'][shard::2]) + 1):
        try:
            c.worker(shard, None, 4)
            break
        except Exception as exc:
            failed = []
            for task, calls in records.items():
                path = v3.RUN / 'paired_episodes' / f'{task}.json'
                if path.exists() and c.feasibility.read(path)['status'] == 'FAILED':
                    if not (out / 'network_skips' / f'{task}.json').exists():
                        failed.append((task, calls, path))
            if len(failed) != 1 or not failure_is_network(exc, failed[0][1]):
                raise
            task, calls, path = failed[0]
            c.atomic_json(out / 'network_skips' / f'{task}.json', {
                'status': 'NETWORK_SKIPPED_NOT_PASS', 'task_id': task, 'failed_source_sha256': c.sha256(path),
                'evidence': [r for r in calls if is_network_error(r.get('exception'))],
                'all_attempt_environment_calls': len(calls), 'exception': f'{type(exc).__name__}:{exc}'})
            # Continue to the next missing task without changing the failure file.
    else:
        raise RuntimeError('bounded worker resume exhausted')
    print(c.stable({'status': 'NETWORK_SKIP_SHARD_COMPLETE', 'shard': shard}), flush=True)


def summarize(v3, c, plan, out):
    scope = checked(v3, c, plan, out)
    m = v3.v.g.m
    episodes, skips, missing, failures = [], dict(scope['initial_skipped']), [], []
    for task in plan['task_ids']:
        path = v3.RUN / 'paired_episodes' / f'{task}.json'
        if not path.exists():
            missing.append(task); continue
        episode = c.feasibility.read(path)
        assert episode['collection_protocol_sha256'] == scope['parent_collection_protocol_sha256']
        if task in skips:
            continue
        if episode['status'] == 'COMPLETE':
            evidence = network_events(episode)
            if evidence:
                skips[task] = {'evidence': evidence, 'source_sha256': c.sha256(path)}
            else:
                episodes.append(episode)
        elif (out / 'network_skips' / f'{task}.json').exists():
            skips[task] = c.feasibility.read(out / 'network_skips' / f'{task}.json')
        else:
            failures.append(task)
    assert not missing and not failures, f'non-network incomplete/failed: {missing}/{failures}'
    assert len(episodes) + len(skips) == 399
    rows = [m.task_metrics(e) for e in episodes]
    aggregate = m.aggregate(episodes, rows)
    statistics = m.paired_statistics(rows)
    span = aggregate['RECOVERY']['post_recovery_span']
    criteria = {'A': statistics['corrections_per100calls']['pooled_rate_difference_CI95'][0] > 0,
                'B': statistics['deep_per100calls']['pooled_rate_difference_CI95'][0] > 0,
                'C': span['tools_only_zero_ratio'] is not None and span['tools_only_zero_ratio'] < .5,
                'D': statistics['new_student_reached_deeper_states']['paired_mean_difference_CI95'][0] > 0}
    result = {'status': 'AVAILABLE_COHORT_COMPLETE', 'parent_planned': 399, 'completed_available': len(episodes),
              'network_skipped': len(skips), 'skipped_tasks': skips, 'aggregate': aggregate,
              'paired_statistics': statistics, 'criteria': criteria,
              'verdict': 'DATA-GO' if criteria['A'] and criteria['B'] and (criteria['C'] or criteria['D']) else 'NO-GO',
              'scope_protocol_sha256': c.sha256(out / 'scope_protocol.json'),
              'parent_collection_protocol_sha256': scope['parent_collection_protocol_sha256'],
              'original399_gate': 'FAILED_PRESERVED', 'original400_gate': 'FAILED_PRESERVED',
              'scope': scope['scope'], 'no_training': True, 'gpu_authorized': True,
              'attrition_caveat': 'network-observed task exclusion may be policy-dependent; available cohort is not an unbiased benchmark',
              'manual_review': 'PENDING'}
    for arm in ('FD', 'RECOVERY'):
        aggregate[arm]['model_inference_seconds_sum'] = aggregate[arm].pop('model_cpu_seconds_sum')
    fixed = set(plan['task_ids'][:100])
    random_rows = [r for r in rows if r['task_id'] not in fixed]
    result['random_available_tasks'] = len(random_rows)
    result['random_available_sensitivity'] = m.paired_statistics(random_rows)
    ordered = sorted(zip(rows, episodes), key=lambda p: (p[0]['arms']['RECOVERY']['max_frontier'] - p[0]['arms']['FD']['max_frontier'],
                      p[0]['arms']['RECOVERY']['new_student_reached_states'],
                      p[0]['arms']['RECOVERY']['corrections'] - p[0]['arms']['FD']['corrections']), reverse=True)
    better = [p for p in ordered if p[0]['arms']['RECOVERY']['max_frontier'] > p[0]['arms']['FD']['max_frontier']][:10]
    used = {p[0]['task_id'] for p in better}
    worse = [p for p in reversed(ordered) if p[0]['task_id'] not in used][:10]
    result['case_shortfall'] = max(0, 10 - len(better))
    result['case_manifest'] = []
    for group, cases in (('frontier_improved', better), ('no_help_or_worse', worse)):
        for row, episode in cases:
            path = out / 'cases' / f'{group}_{row["task_id"]}.json'
            c.atomic_json(path, {'case_group': group, 'metrics': row, 'full_paired_episode': episode, 'manual_review': 'PENDING'})
            result['case_manifest'].append({'task_id': row['task_id'], 'group': group, 'path': str(path), 'sha256': c.sha256(path)})
    c.atomic_json(out / 'paired_metrics.json', result)
    (out / 'paired_task_results.jsonl').write_text(''.join(c.stable(r) + '\n' for r in rows))
    print(c.stable({k: result[k] for k in ('status', 'completed_available', 'network_skipped', 'criteria', 'verdict')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'worker', 'metrics'])
    p.add_argument('--shard', type=int)
    a = p.parse_args()
    v3, c, plan, parent, lock, out = setup()
    if a.command == 'prepare': prepare(v3, c, plan, out)
    elif a.command == 'worker': worker(v3, c, plan, out, a.shard)
    else: summarize(v3, c, plan, out)
