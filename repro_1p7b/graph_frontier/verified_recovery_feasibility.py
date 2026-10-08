"""399-task explicitly scoped feasibility; no GPU or optimizer execution.

The original 400-task gate stays FAILED. This separate protocol excludes ONE
externally unavailable task by explicit user authority, not verifier relaxation.
"""
import argparse
import asyncio
import collections
import copy
import hashlib
import json
import time
from pathlib import Path

from repro_1p7b.graph_frontier import verified_recovery_determinism as det
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256,stable
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import atomic_json,typed_equal
from repro_1p7b.graph_frontier.official_rl_audit_static import ROOT,SOURCE,decode
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import RUN as V1,PROTOCOL,CONFIG
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import GOLD_RUN

RUN=ROOT/'repro_1p7b/graph_frontier/verified_recovery_feasibility_20261007_run'
EXCLUDED='official-rl-0382-c3b604ead122'


def read(p):return json.loads(p.read_text())


def immutable(p,v):
    if p.exists():assert typed_equal(read(p),v),f'immutable artifact drift:{p}'
    else:atomic_json(p,v)


def baseline(task,plan):
    group='fixed100' if task in plan['fixed100'] else 'random300'
    return det.RUN/det.DET_DIR/group/'replica0'/f'{task}.json'


def historical_availability(plan):
    e=next(x for x in plan['tasks'] if x['audit_id']==EXCLUDED)
    historical=V1/'cycle0/rollouts'/f'{EXCLUDED}.json';old=read(historical)
    assert sha256(historical)==e['historical_episode_sha256']
    origin=GOLD_RUN/'episodes'/f'{EXCLUDED}.json';value=read(origin)
    assert sha256(origin)==old['gold_source_episode_sha256']
    assert not value.get('gold_error') and value['gold']['replay_valid'] is True and value['gold']['final_state_match'] is True
    assert typed_equal(value['gold'],old['gold'])
    row=read(SOURCE)[e['source_row_index']]
    actions=[{'name':a['name'],'arguments':a['arguments']} for a in decode(row['reward_model']['ground_truth'])]
    assert typed_equal(actions,old['gold']['actions'])
    failed=det.RUN/det.DET_DIR/'random300/replica0'/f'{EXCLUDED}.json';failure=read(failed)
    diag=det.RUN/'diagnostics/clinicaltrials_external_dependency.json';diagnostic=read(diag)
    assert failure['status']=='FAILED' and 'name resolution' in failure['exception']
    assert diagnostic['status']=='DIAGNOSTIC_REPLAY_FAILED' and 'name resolution' in diagnostic['exception']
    return {'task_id':EXCLUDED,'classification':'EXTERNAL_AVAILABILITY_FAILURE','availability':'EXTERNAL_UNAVAILABLE',
        'historically_executable':True,'historical_episode':str(historical),'historical_episode_sha256':sha256(historical),
        'gold_origin':str(origin),'gold_origin_sha256':sha256(origin),'gold_observation_count':len(old['gold']['observations']),
        'gold_post_states':len(old['gold']['states_after']),
        'observation_hashes':[det.digest(x) for x in old['gold']['observations']],
        'state_hashes':[det.digest(x) for x in old['gold']['states_after']],
        'failure_sha256':sha256(failed),'diagnostic_sha256':sha256(diag),
        'current_live_execution':'FAILED','historical_response_used_for_live_execution':False,
        'interpretation':'Historical artifact proves prior execution only. Original overall gate remains FAILED.'}


def make_plan():
    parent=det.checked_plan();available=historical_availability(parent)
    tasks=parent['fixed100']+[t for t in parent['random300'] if t!=EXCLUDED]
    assert len(tasks)==399 and len(set(tasks))==399
    source={e['audit_id']:e for e in parent['tasks']}
    bases={t:sha256(baseline(t,parent)) for t in tasks}
    assert all(read(baseline(t,parent))['status']=='PASS' for t in tasks)
    protocol=read(PROTOCOL);assert not set(tasks)&set(protocol['heldout_task_ids'])
    value={'schema':'verified_recovery_feasibility_subset_v1','parent_plan_sha256':sha256(det.RUN/'plan.json'),
        'excluded':[available],'task_ids':tasks,'entries':[source[t] for t in tasks],
        'baseline_hashes':bases,'original_full_gate_status':'DETERMINISM_FAILED',
        'scope':'100 prior FIRST_ERROR-stratified + 299 independent random train tasks; feasibility only, NOT benchmark',
        'model':parent['checkpoint'],'model_sha256':parent['checkpoint_sha256'],'config':CONFIG,
        'max_interventions':3,'source_sha256':parent['source_sha256'],'env_hashes':parent['env_hashes'],
        'verifier_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in (
            ROOT/'repro_1p7b/graph_frontier/recursive_opd_v1/mine.py',
            ROOT/'repro_1p7b/graph_frontier/cycle3_official_rl/exec_verified.py',
            ROOT/'repro_1p7b/graph_frontier/cycle3_official_rl/runtime.py',
            ROOT/'repro_1p7b/graph_frontier/state_verifier.py')},
        'gpu_allowed':False,'training_allowed':False}
    immutable(RUN/'subset_plan.json',value);immutable(RUN/'external_availability.json',available)
    return value


def checked():
    p=read(RUN/'subset_plan.json');parent=det.checked_plan()
    assert p['parent_plan_sha256']==sha256(det.RUN/'plan.json') and p['env_hashes']==parent['env_hashes']
    assert p['source_sha256']==sha256(SOURCE)
    for path,digest in p['verifier_hashes'].items():assert sha256(ROOT/path)==digest
    for t,digest in p['baseline_hashes'].items():assert sha256(baseline(t,parent))==digest
    return p,parent


def preflight(shard,shards):
    p,parent=checked();raw=read(SOURCE);entries={e['audit_id']:e for e in p['entries']}
    # fixed100 already has 3 identical fresh replicas; independently repeat all random299.
    ids=[t for t in parent['random300'] if t!=EXCLUDED]
    for index,t in enumerate(ids):
        if index%shards!=shard:continue
        path=RUN/'subset_repeat'/f'{t}.json'
        if path.exists():assert read(path)['subset_plan_sha256']==sha256(RUN/'subset_plan.json');continue
        e=entries[t];started=time.monotonic()
        try:
            assert sha256(V1/'cycle0/rollouts'/f'{t}.json')==e['historical_episode_sha256']
            value=asyncio.run(det.replay(e,raw[e['source_row_index']]))
            assert typed_equal(value,read(baseline(t,parent))['replay']),'fresh state/observation/verifier mismatch'
            result={'task_id':t,'status':'PASS','replay_sha256':det.digest(value)}
        except Exception as exc:result={'task_id':t,'status':'FAILED','exception':f'{type(exc).__name__}:{exc}'}
        result.update(subset_plan_sha256=sha256(RUN/'subset_plan.json'),seconds=time.monotonic()-started)
        atomic_json(path,result);print(stable({'shard':shard,**{k:v for k,v in result.items() if k!='subset_plan_sha256'}}),flush=True)


def preflight_summary():
    p,parent=checked();ids=[t for t in parent['random300'] if t!=EXCLUDED]
    rows=[read(RUN/'subset_repeat'/f'{t}.json') for t in ids if (RUN/'subset_repeat'/f'{t}.json').exists()]
    value={'status':'SUBSET_READY' if len(rows)==299 and all(x['status']=='PASS' for x in rows) else 'SUBSET_FAILED_OR_INCOMPLETE',
        'fixed100_existing_replicas':[100,100,100],'random299_fresh_repeat_passed':sum(x['status']=='PASS' for x in rows),
        'random299_persisted':len(rows),'failures':[x for x in rows if x['status']!='PASS'],
        'subset_size':399,'excluded_external_unavailable':[EXCLUDED],'original_full_gate':'DETERMINISM_FAILED',
        'subset_plan_sha256':sha256(RUN/'subset_plan.json'),'verifier_relaxed':False}
    atomic_json(RUN/'subset_preflight_summary.json',value);print(stable(value))
    if value['status']!='SUBSET_READY':raise SystemExit(3)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['plan','preflight','summary']);a.add_argument('--shard',type=int);a.add_argument('--shards',type=int,default=4);v=a.parse_args()
    if v.command=='plan':print(stable(make_plan()))
    elif v.command=='preflight':preflight(v.shard,v.shards)
    else:preflight_summary()
