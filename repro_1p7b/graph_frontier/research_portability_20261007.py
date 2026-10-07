"""CPU diagnostic only. Never changes/promotes the failed recovery gate."""
import asyncio
import collections
import json

from repro_1p7b.graph_frontier.research_recovery_20261007 import RUN,V1,read
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import IsolatedEpisode,atomic_json,typed_equal
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256,stable


def bag_value(value):
    # Diagnostic sensitivity ONLY. List order may be semantically meaningful.
    if isinstance(value,dict): return {k:bag_value(v) for k,v in value.items()}
    if isinstance(value,list): return sorted((bag_value(v) for v in value),key=stable)
    return value


async def inspect(item):
    gold=read(V1/'cycle0/rollouts'/f"{item['task_id']}.json")['gold'];p=item['frontier_depth']
    before=gold['initial_state'] if p==0 else gold['states_after'][p-1]
    env=IsolatedEpisode(item['task_id'],list(before),before)
    try:
        await env.open();initial=await env.save()
        action=gold['actions'][p];response,ok,error=await env.call(action['name'],action['arguments'])
        after=await env.save()
    finally: await env.close()
    restored=IsolatedEpisode(item['task_id'],list(after),after)
    try:
        await restored.open();roundtrip=await restored.save()
    finally: await restored.close()
    expected=gold['observations'][p]
    return {'task_id':item['task_id'],'environment':item['environment'],'action':action,
            'snapshot_roundtrip_exact':typed_equal(initial,before),
            'execution_success':ok,'exception':error,
            'response_exact':typed_equal(response,expected),
            'response_bag_sensitivity_equal':typed_equal(bag_value(response),bag_value(expected)),
            'post_state_exact':typed_equal(after,gold['states_after'][p]),
            'post_state_bag_sensitivity_equal':typed_equal(bag_value(after),bag_value(gold['states_after'][p])),
            'post_state_roundtrip_exact':typed_equal(roundtrip,after),
            'response_observed':response,'response_expected':expected,
            'plan_sha256':sha256(RUN/'plan.json'),'gate_promoted':False}


def run():
    failed={x['task_id'] for x in read(RUN/'cpu_preflight_summary.json')['failures']}
    results=[]
    for item in read(RUN/'plan.json')['selected_tasks']:
        if item['task_id'] not in failed:continue
        path=RUN/'portability_diagnostics'/f"{item['task_id']}.json"
        if path.exists(): result=read(path)
        else:
            result=asyncio.run(inspect(item));atomic_json(path,result)
        results.append(result);print(stable({k:v for k,v in result.items() if not k.startswith('response_o') and not k.startswith('response_e')}),flush=True)
    summary={'original_gate':'CPU_GATE_FAILED','gate_promoted':False,'failed_tasks':len(results),
             'phase_counts':{key:sum(bool(x[key]) for x in results) for key in ('snapshot_roundtrip_exact','execution_success','response_exact','response_bag_sensitivity_equal','post_state_exact','post_state_roundtrip_exact')},
             'results':results,'plan_sha256':sha256(RUN/'plan.json')}
    atomic_json(RUN/'portability_summary.json',summary)


if __name__=='__main__':run()
