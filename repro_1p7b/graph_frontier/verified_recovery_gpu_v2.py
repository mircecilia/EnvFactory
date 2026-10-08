"""Order-faithful authoritative initial scenario + paid independent prefix replay.

No producer/verifier/gold patch. A JSON snapshot has unordered object identity
but cannot encode hidden dict insertion order. Never reset the live student env.
Independent target verification reconstructs its state by ACTUAL gold execution
from the ORIGINAL source initial scenario, with every prefix call charged.
"""
import argparse
import copy
import importlib.metadata
from pathlib import Path

from repro_1p7b.graph_frontier import verified_recovery_gpu as g
from repro_1p7b.graph_frontier.official_rl_audit_static import decode

c=g.c
RUN=c.feasibility.RUN/'gpu_sglang_v2'
ACTIVE=None
BaseEpisode=c.CountedEpisode


class SourceOrderedEpisode(BaseEpisode):
    def __init__(self,task_id,servers,initial):
        if ACTIVE is not None and task_id==ACTIVE['task_id'] and c.typed_equal(initial,ACTIVE['gold']['initial_state']):
            initial=copy.deepcopy(ACTIVE['initial'])
        super().__init__(task_id,servers,initial)


async def prefix_validate_target(task,gold,p):
    assert ACTIVE is not None and ACTIVE['task_id']==task
    env=SourceOrderedEpisode(task,list(gold['initial_state']),copy.deepcopy(ACTIVE['initial']))
    try:
        await env.open()
        if not c.typed_equal(await env.save(),gold['initial_state']):return False
        for i in range(p):
            response,ok,error=await c.call_as(env,'TARGET_PREFIX_REPLAY',gold['actions'][i])
            if not (ok is True and error is None and c.typed_equal(response,gold['observations'][i])
                    and c.typed_equal(await env.save(),gold['states_after'][i])):return False
        if not c.typed_equal(await env.save(),c.state_at(gold,p)):return False
        if p==len(gold['actions']):return c.typed_equal(await env.save(),gold['final_state'])
        response,ok,error=await c.call_as(env,'TARGET_VALIDATION',gold['actions'][p])
        return ok is True and error is None and c.typed_equal(response,gold['observations'][p]) and c.typed_equal(await env.save(),gold['states_after'][p])
    finally:await env.close()


def install():
    g.RUN=RUN
    g.install()
    c.CountedEpisode=SourceOrderedEpisode
    c.validate_target=prefix_validate_target
    original_collect=c.collect
    async def source_collect(task,entry,row,gold,schemas,policy,config,arm):
        global ACTIVE
        assert ACTIVE is None,'overlapping task in isolated worker'
        ACTIVE={'task_id':task,'gold':gold,'initial':decode(row['extra_info']['mcp_factory_kwargs']['initial_config'])}
        try:
            result=await original_collect(task,entry,row,gold,schemas,policy,config,arm)
            result['initial_scenario_source']='authoritative unsorted source initial_config; not canonical saved snapshot'
            result['independent_target_verification']='live original-initial -> paid verified gold prefix -> one gold target; never rollback student env'
            return result
        finally:ACTIVE=None
    c.collect=source_collect
    original_write=g.m.atomic_json
    def write(path,value):
        if path.name.endswith('_metrics.json') or path.name=='paired_metrics.json':
            value['target_validation_prefix_cost_included']=True
            for arm in value.get('aggregate',{}).values():
                if isinstance(arm,dict) and 'model_cpu_seconds_sum' in arm:
                    arm['model_inference_seconds_sum']=arm.pop('model_cpu_seconds_sum')
        original_write(path,value)
    g.m.atomic_json=write


def protocol():
    f=c.feasibility
    original=f.immutable
    def write(path,value):
        if path==RUN/'collection_protocol.json':
            value=copy.deepcopy(value)
            value['schema']='verified_recovery_gpu_sglang_paired_v2_source_order'
            value['code_sha256'][Path(__file__).name]=g.sha256(Path(__file__))
            value['initial_scenario_source']='original authoritative SOURCE row initial_config including object insertion order'
            value['target_validation']='independent environment ACTUALLY replays exact gold prefix from original source initial_config; every response and post-state exact; all TARGET_PREFIX_REPLAY calls in primary cost. Never resets actual student environment.'
            value['previous_epoch']='gpu_sglang_v1 preserved, invalid incomplete collection due snapshot-order local-target failure; no examples migrated'
            value['package_versions']={name:importlib.metadata.version(name) for name in ('sglang','torch','transformers')}
        original(path,value)
    f.immutable=write
    try:return g.protocol()
    finally:f.immutable=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['protocol','worker','metrics'])
    p.add_argument('--shard',type=int);p.add_argument('--limit',type=int);p.add_argument('--expected',type=int)
    a=p.parse_args();install()
    if a.command=='protocol':print(g.stable(protocol()))
    elif a.command=='worker':c.worker(a.shard,a.limit,4)
    else:g.m.summarize(a.expected)
