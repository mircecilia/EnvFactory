"""Versioned CPU-only determinism gate; gold answers/reward/equality unchanged.

Fresh runtime observations are a new replay cache, NOT changed ground_truth.
Legacy observations remain immutable; all differences are explicitly retained.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import random
import time
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import IsolatedEpisode,atomic_json,typed_equal
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256,stable
from repro_1p7b.graph_frontier.official_rl_audit_static import ROOT,SOURCE,OUT,decode,query_from_row
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import RUN as V1,PROTOCOL,CONFIG
from repro_1p7b.graph_frontier.recursive_opd_v1.mine import classify
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import parse_calls
from repro_1p7b.graph_frontier.state_verifier import compare_final_states

RUN=ROOT/'repro_1p7b/graph_frontier/verified_recovery_20261007_run/environment_v3'
DET_DIR='determinism_v3'
ENV_FILES=['GorillaFileSystem','WhatsApp','AirDNA','MetMuseum','EtsyServer','TickTick','GitHubServer']
VERIFIER_FILES=['cycle3_official_rl/runtime.py','recursive_opd_v1/mine.py','state_verifier.py','cycle3_official_rl/audit.py','cycle3_semantic_validation.py']
SEED=20261008


def read(path):return json.loads(path.read_text())


def digest(value):return hashlib.sha256(stable(value).encode()).hexdigest()


def env_hashes():
    # All executor modules, not only the seven repaired ones.
    return {p.name:sha256(p) for p in sorted((ROOT/'envs/tools').glob('*.py'))}


def immutable(path,value):
    if path.exists():assert typed_equal(read(path),value),'immutable research artifact drift'
    else:atomic_json(path,value)


def make_plan():
    protocol=read(PROTOCOL)
    original=read(V1.parent/'research_recovery_20261007_run/plan.json')
    fixed=[x['task_id'] for x in original['selected_tasks']]
    remaining=sorted(set(protocol['train_task_ids'])-set(fixed))
    random300=random.Random(SEED).sample(remaining,300)
    entries={e['audit_id']:e for e in read(OUT/'opd_source_manifest.json')['rows']}
    files={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'repro_1p7b/graph_frontier/cycle3_official_rl/runtime.py',ROOT/'repro_1p7b/graph_frontier/recursive_opd_v1/mine.py',ROOT/'repro_1p7b/graph_frontier/state_verifier.py')}
    value={'schema':'verified_recovery_determinism_v1','seed':SEED,'source_sha256':sha256(SOURCE),
           'source_manifest_sha256':sha256(OUT/'opd_source_manifest.json'),'protocol_sha256':sha256(PROTOCOL),
           'original_plan_sha256':sha256(V1.parent/'research_recovery_20261007_run/plan.json'),
           'env_hashes':env_hashes(),'unchanged_verifier_hashes':files,
           'fixed100':fixed,'random300':random300,'collection300':random300,
           'tasks':[{**entries[t],'historical_episode_sha256':sha256(V1/'cycle0/rollouts'/f'{t}.json')} for t in fixed+random300],
           'config':CONFIG,'checkpoint':protocol['pi0_model'],'checkpoint_sha256':protocol['pi0_model_sha256'],
           'gold_answer_contract':'ground_truth names/typed arguments/order, initial scenario, expected final scenario, historical gold states remain identical; only new versioned runtime observations may differ',
           'equality_contract':'existing typed_equal unchanged; fresh raw outputs compared exactly across all fresh replay replicas; legacy output differences reported, never overwritten',
           'budget_contract':'Phase0 audited separately; Phase2 counts EVERY forward student or corrective tool invocation. A wrong call AND its correction both consume the same cap11. Reset/load/save and verifier/replay invocations separately accounted; total costs must be reported.'}
    assert len(fixed)==100 and len(random300)==300 and not set(fixed)&set(random300)
    assert not set(random300)&set(protocol['heldout_task_ids'])
    immutable(RUN/'plan.json',value);return value


def checked_plan():
    p=read(RUN/'plan.json')
    assert p['source_sha256']==sha256(SOURCE)
    assert p['protocol_sha256']==sha256(PROTOCOL)
    assert p['env_hashes']==env_hashes(),'executor source drift'
    for path,value in p['unchanged_verifier_hashes'].items():assert sha256(ROOT/path)==value
    return p


def checked_audit_protocol():
    value={'parent_plan_sha256':sha256(RUN/'plan.json'),'audit_code_sha256':sha256(Path(__file__)),
           'directory':DET_DIR,'answer_projection':'same as official gold_replay: name/typed arguments/order; masked_arguments is non-executable metadata',
           'engineering_failure_preserved':'determinism/fixed100/replica0 (100 schema-wrapper failures, zero environment calls)'}
    immutable(RUN/'audit_protocol_v3.json',value)
    return value


def append_step(student,gold,p):
    action=gold['actions'][p];before=gold['initial_state'] if p==0 else gold['states_after'][p-1]
    count=len(student['messages']);call_id=f'gold_{p}'
    message={'role':'assistant','content':'','tool_calls':[{'id':call_id,'type':'function','function':{'name':action['name'],'arguments':stable(action['arguments'])}}]}
    calls,errors=parse_calls(message)
    student['steps'].append({'step_index':len(student['steps']),'prompt_message_count':count,'model_message':message,
        'structured_calls':calls,'parse_errors':errors,'typed_events':[{'tool_name':action['name'],'tool_arguments':action['arguments'],
        'tool_response':gold['observations'][p],'execution_success':True,'exception':None}],
        'state_before_action':before,'state_after_action':gold['states_after'][p]})
    student['messages']+=[message,{'role':'tool','tool_call_id':call_id,'name':action['name'],'content':stable(gold['observations'][p])}]


async def replay(item,row):
    old=read(V1/'cycle0/rollouts'/f"{item['audit_id']}.json")
    source_actions=decode(row['reward_model']['ground_truth'])
    assert isinstance(source_actions,list) and all(isinstance(a.get('name'),str) and isinstance(a.get('arguments'),dict) for a in source_actions)
    actions=[{'name':a['name'],'arguments':a['arguments']} for a in source_actions]
    factory=row['extra_info']['mcp_factory_kwargs']
    assert typed_equal(actions,old['gold']['actions']),'gold answer drift'
    servers=decode(factory['mcp_servers']);initial=decode(factory['initial_config']);expected=decode(factory['final_config'])
    assert typed_equal({s:initial[s] for s in servers},old['gold']['initial_state']),'initial answer drift'
    env=IsolatedEpisode(item['audit_id'],servers,initial)
    try:
        await env.open();schemas=copy.deepcopy(env.schemas)
        gold={'actions':actions,'initial_state':await env.save(),'observations':[],'states_after':[]}
        for index,action in enumerate(actions):
            response,ok,error=await env.call(action['name'],action['arguments'])
            assert ok is True and error is None,f'fixed gold answer execution failed:{index}:{error}'
            gold['observations'].append(response);gold['states_after'].append(await env.save())
            assert typed_equal(gold['states_after'][-1],old['gold']['states_after'][index]),f'historical gold state changed:{index}'
        gold['final_state']=await env.save()
        assert typed_equal(gold['final_state'],old['gold']['final_state']),'historical final gold state changed'
        assert compare_final_states(gold['final_state'],{s:expected[s] for s in servers}) is True,'official final verifier failed'
    finally:await env.close()
    metrics={x['task_id']:x for x in (json.loads(line) for line in (V1/'cycle0/metrics/per_task.jsonl').read_text().splitlines())}
    metric=metrics[item['audit_id']];p=metric['frontier_depth'] if metric['status']=='FIRST_ERROR' else 0
    state=gold['initial_state'] if p==0 else gold['states_after'][p-1]
    fork=IsolatedEpisode(item['audit_id'],servers,state)
    try:
        await fork.open();assert typed_equal(await fork.save(),state),'pre-error restore mismatch'
        if p<len(actions):
            action=actions[p];response,ok,error=await fork.call(action['name'],action['arguments'])
            assert ok and error is None and typed_equal(response,gold['observations'][p]),'restored observation mismatch'
            assert typed_equal(await fork.save(),gold['states_after'][p]),'restored transition mismatch'
        else:assert typed_equal(await fork.save(),gold['final_state']),'terminal restore mismatch'
    finally:await fork.close()
    student={'messages':[{'role':'user','content':query_from_row(row)}],'steps':[],
             'final_state':gold['final_state'],'termination_reason':'MODEL_FINAL'}
    for i in range(len(actions)):append_step(student,gold,i)
    student['steps'].append({'step_index':len(actions),'prompt_message_count':len(student['messages']),
        'model_message':{'role':'assistant','content':''},'structured_calls':[],'typed_events':[],
        'parse_errors':[],'state_before_action':gold['final_state'],'state_after_action':gold['final_state']})
    student['messages'].append({'role':'assistant','content':''})
    verified=await classify({'gold':gold,'student':student},item,row,schemas)
    assert verified['status']=='SUCCESS','gold trajectory verifier decision not SUCCESS'
    changes=[i for i,(a,b) in enumerate(zip(gold['observations'],old['gold']['observations'])) if not typed_equal(a,b)]
    return {'gold':gold,'schemas':schemas,'verifier_decision':verified,'legacy_observation_diff_steps':changes,
            'gold_answer_unchanged':True,'historical_states_unchanged':True,'restore_checked_pointer':p}


def worker(group,replica,shard,shards):
    plan=checked_plan();audit=checked_audit_protocol();raw=read(SOURCE);by_id={e['audit_id']:e for e in plan['tasks']}
    for index,task in enumerate(plan[group]):
        if index%shards!=shard:continue
        item=by_id[task];path=RUN/DET_DIR/group/f'replica{replica}'/f'{task}.json'
        if path.exists():
            assert read(path)['plan_sha256']==sha256(RUN/'plan.json');continue
        start=time.monotonic()
        value=None
        try:
            assert item['historical_episode_sha256']==sha256(V1/'cycle0/rollouts'/f'{task}.json')
            value=asyncio.run(replay(item,raw[item['source_row_index']]))
            baseline=RUN/DET_DIR/group/'replica0'/f'{task}.json'
            if replica>0:
                original=read(baseline);assert original['status']=='PASS','baseline failed'
                assert typed_equal(value,original['replay']),'state/observation/verifier decision differs between fresh processes'
            result={'task_id':task,'status':'PASS','replay':value}
        except Exception as exc:
            result={'task_id':task,'status':'FAILED','exception':f'{type(exc).__name__}:{exc}'}
            if value is not None:result['observed_replay']=value
        result.update(plan_sha256=sha256(RUN/'plan.json'),audit_protocol_sha256=sha256(RUN/'audit_protocol_v3.json'),wall_seconds=time.monotonic()-start,replica=replica)
        atomic_json(path,result);print(stable({'group':group,'replica':replica,'shard':shard,'task_id':task,'status':result['status']}),flush=True)


def summarize(group=None,replica=None):
    plan=checked_plan();checked_audit_protocol();summaries=[]
    selected=[(group,replica)] if group is not None else [('fixed100',r) for r in range(3)]+[('random300',r) for r in range(2)]
    for g,r in selected:
        rows=[read(RUN/DET_DIR/g/f'replica{r}'/f'{t}.json') for t in plan[g] if (RUN/DET_DIR/g/f'replica{r}'/f'{t}.json').exists()]
        summaries.append({'group':g,'replica':r,'planned':len(plan[g]),'persisted':len(rows),
            'passed':sum(x['status']=='PASS' for x in rows),'failures':[x for x in rows if x['status']!='PASS'],
            'legacy_diff_tasks':sum(bool(x.get('replay',{}).get('legacy_observation_diff_steps')) for x in rows)})
    ready=all(x['passed']==x['planned'] for x in summaries)
    summary={'status':'DETERMINISM_PASS' if ready else 'DETERMINISM_FAILED','replicas':summaries,'plan_sha256':sha256(RUN/'plan.json'),
        'equality_relaxed':False,'gold_answers_changed':False,'historical_cache_overwritten':False,
        'interpretation':'new canonical environment raw-output reproducibility gate; NOT historical raw-output equality. All fixed gold answers and historical states must remain unchanged.'}
    path=RUN/('determinism_summary_v3.json' if group is None else f'gate_v3_{group}_{replica}.json')
    atomic_json(path,summary);print(stable(summary),flush=True)
    if not ready:raise SystemExit(3)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','worker','summary']);p.add_argument('--group',choices=['fixed100','random300']);p.add_argument('--replica',type=int);p.add_argument('--shard',type=int);p.add_argument('--shards',type=int,default=4);a=p.parse_args()
    if a.command=='plan':print(stable(make_plan()))
    elif a.command=='worker':worker(a.group,a.replica,a.shard,a.shards)
    else:summarize(a.group,a.replica)
