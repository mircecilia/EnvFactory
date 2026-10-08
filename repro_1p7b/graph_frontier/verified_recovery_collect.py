"""FD-stop vs SAME-STATE K<=3 Recovery; fresh CPU policy decisions, typed MCP.

The verifier gets a labelled GOLD_REFERENCE_PREFIX view to align a local
decision at global pointer p. These reference steps are NOT model history,
not collected student actions, not free live gold executions. Every current
before-state must equal the verified gold state. Student history is NEVER
rewritten, and an inline repair never rolls back the live environment.
"""
import argparse
import asyncio
import collections
import contextvars
import copy
import hashlib
import json
import time
from pathlib import Path

from repro_1p7b.graph_frontier import verified_recovery_feasibility as feasibility
from repro_1p7b.graph_frontier import verified_recovery_determinism as det
from repro_1p7b.graph_frontier.recursive_opd_v1 import mine
from repro_1p7b.graph_frontier.cycle3_official_rl import exec_verified
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import IsolatedEpisode,typed_equal,atomic_json
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256,stable
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import parse_calls
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE,query_from_row
from repro_1p7b.graph_frontier.verified_recovery_cpu_policy import CPUPolicy

RUN=feasibility.RUN
COST=contextvars.ContextVar('recovery_cost',default=None)
ROLE=contextvars.ContextVar('recovery_role',default='VERIFIER_REPLAY')


class CountedEpisode(IsolatedEpisode):
    async def open(self):
        if COST.get() is not None:COST.get()['admin']+=2*len(self.servers) # load + list-tools
        return await super().open()

    async def save(self):
        if COST.get() is not None:COST.get()['admin']+=len(self.clients)
        return await super().save()

    async def call(self,name,arguments):
        if COST.get() is not None:
            allowed={t['function']['name'] for t in self.schemas}
            COST.get()['ledger'].append({'index':len(COST.get()['ledger'])+1,'role':ROLE.get(),
                'name':name,'arguments':copy.deepcopy(arguments),'mcp_dispatched':name in allowed and name.split('-',1)[0] in self.clients})
        return await super().call(name,arguments)


def install_accounting():
    # Process-local constructor injection only; unchanged verifier conditions.
    mine.IsolatedEpisode=CountedEpisode;exec_verified.IsolatedEpisode=CountedEpisode


async def call_as(env,role,action):
    token=ROLE.set(role)
    try:return await env.call(action['name'],action['arguments'])
    finally:ROLE.reset(token)


def state_at(gold,p):return gold['initial_state'] if p==0 else gold['states_after'][p-1]


async def decision(gold,p,step,entry,row,schemas):
    if not typed_equal(step['state_before_action'],state_at(gold,p)):
        return {'status':'UNCERTAIN','reason':'current state not exactly verified gold frontier','frontier_depth':p}
    view={'messages':[{'role':'user','content':query_from_row(row)}],'steps':[],
          'final_state':step['state_after_action'],'termination_reason':step['termination_reason']}
    for i in range(p):det.append_step(view,gold,i)
    local=copy.deepcopy(step);local['step_index']=p;local['prompt_message_count']=len(view['messages'])
    view['steps'].append(local);view['messages'].append(local['model_message'])
    verdict=await mine.classify({'gold':gold,'student':view},entry,row,schemas)
    verdict['verification_view']='GOLD_REFERENCE_PREFIX_PLUS_CURRENT_REAL_STEP'
    verdict['reference_prefix_actions']=p
    if verdict['status']=='FIRST_ERROR' and verdict['student_step']!=p:raise RuntimeError('verifier error does not refer to current decision')
    return verdict


async def validate_target(task,gold,p):
    state=state_at(gold,p);env=CountedEpisode(task,list(state),state)
    try:
        await env.open()
        if p==len(gold['actions']):return typed_equal(await env.save(),gold['final_state'])
        response,ok,error=await call_as(env,'TARGET_VALIDATION',gold['actions'][p])
        return ok is True and error is None and typed_equal(response,gold['observations'][p]) and typed_equal(await env.save(),gold['states_after'][p])
    finally:await env.close()


def add_state(states,state,p,origin):
    key=det.digest(state)
    states.append({'state_hash':key,'pointer':p,'origin':origin,'state':copy.deepcopy(state)})


async def collect(task,entry,row,gold,schemas,policy,config,arm):
    cost={'ledger':[],'admin':0};token=COST.set(cost)
    env=CountedEpisode(task,list(gold['initial_state']),gold['initial_state'])
    messages=[{'role':'user','content':query_from_row(row)}]
    pointer=0;forward_calls=0;interventions=0;steps=[];targets=[];errors=[];states=[];spans=[];unexecuted=[];active=None
    termination='MAX_TURNS';started=time.monotonic()
    try:
        await env.open();assert typed_equal(env.schemas,schemas),'schema drift'
        add_state(states,await env.save(),0,'INITIAL')
        for request in range(config['max_turns']):
            before=await env.save();prefix=copy.deepcopy(messages)
            message,finish,audit=policy.generate(messages,schemas,config['seed']+entry['source_row_index']*1009+request,config)
            calls,parse_errors=parse_calls(message)
            # No partial batch, no fabricated failed calls at the tool cap.
            if forward_calls+len(calls)>config['tool_call_limit']:
                unexecuted.append({'request_index':request,'message':message,'model_audit':audit,'reason':'TOOL_BUDGET_NO_PARTIAL_BATCH'})
                termination='TOOL_BUDGET';break
            count=len(messages);messages.append(message);events=[]
            # Same simultaneous executor convention as official current collector.
            async def execute(c):
                response,ok,error=await call_as(env,'STUDENT',c)
                return {'tool_name':c['name'],'tool_arguments':c['arguments'],'tool_response':response,
                    'execution_success':ok,'exception':error}
            events=await asyncio.gather(*(execute(c) for c in calls));forward_calls+=len(calls)
            bad={e['position']:e for e in parse_errors};by_position={c['position']:e for c,e in zip(calls,events)}
            for index,raw in enumerate(message.get('tool_calls') or []):
                event=by_position.get(index);value=(event['tool_response'] if event['execution_success'] else {'error':event['exception']}) if event else {'error':bad[index]}
                messages.append({'role':'tool','name':str((raw.get('function') or {}).get('name') or 'unknown'),
                    'tool_call_id':str(raw.get('id') or f'call_{index}'),'content':stable(value)})
            after=await env.save()
            step={'step_index':request,'prompt_message_count':count,'model_message':message,'structured_calls':calls,
                'parse_errors':parse_errors,'typed_events':events,'state_before_action':before,'state_after_action':after,
                'termination_reason':('MODEL_FINAL' if finish!='length' else 'GENERATION_LENGTH') if not message.get('tool_calls') else 'MAX_TURNS',
                'source':'STUDENT','model_audit':audit,'pointer_before':pointer,'prior_interventions':interventions}
            verdict=await decision(gold,pointer,step,entry,row,schemas);step['verifier']=verdict;steps.append(step)
            if verdict['status']=='SUCCESS':
                if active is not None:active.update(actions=active['actions']+1,ended_by='STUDENT_SUCCESS',censored=False);active=None
                termination='STUDENT_SUCCESS';break
            if verdict['status']=='FIRST_ERROR':
                if active is not None:active.update(ended_by='NEXT_VERIFIED_ERROR',censored=False);active=None
                error={'pointer':pointer,'request_index':request,'type':verdict['error_regime'],'reason':verdict['reason']}
                errors.append(error)
                if arm=='RECOVERY' and interventions>=3:termination='INTERVENTION_LIMIT';break
                if not await validate_target(task,gold,pointer):raise RuntimeError('independent same-prestate gold target validation failed')
                target={**error,'task_id':task,'state_hash':det.digest(before),'state_before':before,'wrong_message':message,
                    'wrong_observed_state':after,'conversation_prefix':prefix,
                    'gold_action':gold['actions'][pointer] if pointer<len(gold['actions']) else 'EOS_ONLY',
                    'independently_executable':True,'environment_call_index':len(cost['ledger']),
                    'provenance':'PURE_STUDENT_PREFIX' if not interventions else 'POST_INTERVENTION_HYBRID',
                    'inline_repair_attempted':False,'inline_repair_success':False}
                targets.append(target)
                if arm=='FD':termination='FIRST_DIVERGENCE_STOP';break
                if pointer==len(gold['actions']):
                    target['inline_repair_attempted']=True
                    if not typed_equal(after,gold['final_state']):termination='UNRECOVERABLE_TERMINAL_STATE';break
                    target.update(inline_repair_success=True,state_after=after)
                    interventions+=1;termination='GOLD_CORRECTED_TERMINAL';break
                if forward_calls>=config['tool_call_limit']:termination='NO_BUDGET_FOR_CORRECTION';break
                # Crucial: SAME env at the actual POST-error state; no reset.
                action=gold['actions'][pointer];target['inline_repair_attempted']=True
                response,ok,error=await call_as(env,'GOLD_LOCAL_CORRECTION',action);forward_calls+=1
                repaired=await env.save()
                target.update(inline_response=response,inline_exception=error,state_after=repaired)
                if not (ok is True and error is None and typed_equal(response,gold['observations'][pointer]) and typed_equal(repaired,gold['states_after'][pointer])):
                    termination='UNRECOVERABLE_LOCAL_CORRECTION';break
                target['inline_repair_success']=True;interventions+=1;pointer+=1
                repair_id=f'gold_local_{interventions}'
                repair_message={'role':'assistant','content':'','tool_calls':[{'id':repair_id,'type':'function',
                    'function':copy.deepcopy(action)}]}
                messages.extend([repair_message,{'role':'tool','name':action['name'],'tool_call_id':repair_id,'content':stable(response)}])
                add_state(states,repaired,pointer,'GOLD_LOCAL_CORRECTION')
                active={'after_target_index':len(targets)-1,'actions':0,'tool_progress_actions':0,'nonprogress_reads':0,
                    'start_pointer':pointer,'censored':True,'ended_by':'UNFINISHED'};spans.append(active)
                continue
            # The established classifier calls an unfinished but verified prefix
            # UNCERTAIN. ONLY this exact reason allows another student decision.
            if verdict['status']=='UNCERTAIN' and verdict['reason']=='max-turn exhaustion or no verifiable next decision':
                consumed=verdict['frontier_depth']-pointer
                assert consumed>=0
                if consumed:
                    pointer+=consumed;add_state(states,after,pointer,'STUDENT')
                    if active is not None:active['actions']+=consumed;active['tool_progress_actions']+=consumed
                elif active is not None:active['nonprogress_reads']+=len(calls)
                continue
            termination='UNCERTAIN_VERIFIER';break
        if active is not None:active['ended_by']=termination # retains censored=True, not an immediate error
        final=await env.save()
        return {'task_id':task,'arm':arm,'status':'COMPLETE','termination':termination,'targets':targets,'errors':errors,
            'successful_interventions':interventions,'steps':steps,'messages':messages,'verified_states':states,
            'spans':spans,'unexecuted_proposals':unexecuted,'max_verified_frontier_depth':pointer,'final_state':final,
            'forward_calls':forward_calls,'environment_calls':len(cost['ledger']),
            'actual_mcp_domain_calls':sum(x['mcp_dispatched'] for x in cost['ledger']),
            'environment_call_ledger':cost['ledger'],'admin_mcp_calls':cost['admin'],
            'wall_seconds':time.monotonic()-started,'policy_engine':'CPU_FP32_TRANSFORMERS_SGLANG_QWEN25_PARSER',
            'dataset_target_definition':'independently execution-verified corrective target at real PRE-error state; not identical to successful inline intervention',
            'no_rollback':True,'no_history_truncation':True}
    finally:await env.close();COST.reset(token)


def make_collection_protocol():
    plan,parent=feasibility.checked();assert feasibility.read(RUN/'subset_preflight_summary.json')['status']=='SUBSET_READY'
    model=Path(plan['model']);assert sha256(model/'model.safetensors')==plan['model_sha256'],'original model hash drift'
    paths=[Path(__file__),Path(__file__).with_name('verified_recovery_cpu_policy.py'),
           Path(__file__).with_name('verified_recovery_feasibility.py'),Path(__file__).with_name('verified_recovery_metrics.py')]
    value={'schema':'verified_recovery_cpu_paired_v1','subset_plan_sha256':sha256(RUN/'subset_plan.json'),
        'subset_preflight_sha256':sha256(RUN/'subset_preflight_summary.json'),
        'model_sha256':plan['model_sha256'],'config':plan['config'],'max_interventions':3,
        'code_sha256':{p.name:sha256(p) for p in paths},
        'model_input_hashes':{file:sha256(model/file) for file in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json')},
        'engine':'transformers greedy CPU float32 no quantization; identical engine/template/parser for both arms; NOT guaranteed equal to historical GPU',
        'cost_definition':'all actual IsolatedEpisode.call invocations (student/error/gold correction/target verification/verifier replay); separately actual MCP-dispatched domain calls and admin calls. No blocked synthetic calls counted.',
        'depth_definition':'verified gold tool-action pointer, NOT dependency graph depth',
        'state_definition':'unique canonical physical environment state per task, NOT pointer-only identity; teacher-forced and student-reached states separate',
        'span_definition':'verified autonomous gold-progress actions plus correct student EOS; tools-only span also reported; budget/uncertain censored intervals not zero failures',
        'gate_predeclared':{'A':'paired task-resampled pooled yield difference lower95CI>0',
            'B':'depth>=2 paired task-resampled pooled yield difference lower95CI>0',
            'C':'among completed (not budget-censored) spans, tools-only zero-step fraction<0.5; report EOS-inclusive sensitivity. A repaired final gold action followed only by student EOS is not autonomous tool-suffix evidence.',
            'D':'positive paired new-student-reached deeper physical states mean lower95CI>0; no teacher-only states counted for D',
            'DATA_GO':'A and B and (C or D); otherwise NO-GO; incomplete/infrastructure not a method verdict'},
        'paired_sampling':'identical task order and request seed formula; 2 CPU workers, each task FD then Recovery; compare no-intervention prefixes',
        'memoization':'Only identical FRESH CPU input token IDs + seed + length within SAME paired task may reuse deterministic output; intervened contexts generated anew. Actual CPU seconds/cache hits reported; cached wall-clock advantage is NOT a method claim.',
        'kv_cache':'CPU-only incremental prefill, cropped to longest ACTUAL input token-ID prefix; uncached/cached greedy outputs must match in independent probe. Thread count4 per worker; no quantization.',
        'task_bootstrap_seed':20261009,'bootstrap_replicates':10000,'no_gpu':True,'no_training':True}
    feasibility.immutable(RUN/'collection_protocol.json',value);return value


def verify_protocol():
    p,parent=feasibility.checked();lock=feasibility.read(RUN/'collection_protocol.json')
    assert lock['subset_plan_sha256']==sha256(RUN/'subset_plan.json')
    assert lock['subset_preflight_sha256']==sha256(RUN/'subset_preflight_summary.json')
    assert sha256(Path(p['model'])/'model.safetensors')==lock['model_sha256'],'worker checkpoint identity drift'
    for file,digest in lock['model_input_hashes'].items():assert sha256(Path(p['model'])/file)==digest
    for file,digest in lock['code_sha256'].items():assert sha256(Path(__file__).with_name(file))==digest
    return p,parent,lock


def worker(shard,limit=None,threads=4):
    p,parent,lock=verify_protocol();raw=feasibility.read(SOURCE)
    entries={e['audit_id']:e for e in p['entries']};ids=p['task_ids'][shard::2]
    if limit is not None:ids=ids[:limit]
    policy=CPUPolicy(p['model'],threads);install_accounting()
    for i,task in enumerate(ids):
        policy.cache.clear()
        path=RUN/'paired_episodes'/f'{task}.json'
        if path.exists():assert feasibility.read(path)['collection_protocol_sha256']==sha256(RUN/'collection_protocol.json');continue
        e=entries[task];reference=feasibility.read(feasibility.baseline(task,parent))['replay']
        item={'task_id':task,'source_row_index':e['source_row_index'],'collection_protocol_sha256':sha256(RUN/'collection_protocol.json')}
        try:
            arms={arm:asyncio.run(collect(task,e,raw[e['source_row_index']],reference['gold'],reference['schemas'],policy,p['config'],arm)) for arm in ('FD','RECOVERY')}
            # Every pre-intervention request must really be a matched fresh policy decision.
            for a,b in zip(arms['FD']['steps'],arms['RECOVERY']['steps']):
                if b['prior_interventions']:break
                assert a['model_audit']['input_token_ids_sha256']==b['model_audit']['input_token_ids_sha256'],'paired input drift'
                assert typed_equal(a['model_message'],b['model_message']),'paired greedy output drift'
                assert typed_equal(a['state_before_action'],b['state_before_action']),'paired state drift'
                assert typed_equal(a['state_after_action'],b['state_after_action']),'paired post-state drift'
                assert typed_equal(a['typed_events'],b['typed_events']),'paired response drift'
                assert typed_equal(a['verifier'],b['verifier']),'paired decision drift'
            item.update(status='COMPLETE',arms=arms)
        except Exception as exc:
            item.update(status='FAILED',exception=f'{type(exc).__name__}:{exc}')
            atomic_json(path,item);raise
        atomic_json(path,item)
        print(stable({'shard':shard,'completed':i+1,'task_id':task,'FD_targets':len(arms['FD']['targets']),
            'Recovery_targets':len(arms['RECOVERY']['targets']),'FD_calls':arms['FD']['environment_calls'],
            'Recovery_calls':arms['RECOVERY']['environment_calls'],'Recovery_termination':arms['RECOVERY']['termination']}),flush=True)


def probe():
    p,parent=feasibility.checked();task=p['task_ids'][0];e=p['entries'][0]
    reference=feasibility.read(feasibility.baseline(task,parent))['replay'];row=feasibility.read(SOURCE)[e['source_row_index']]
    policy=CPUPolicy(p['model'],4)
    responses=[]
    for _ in range(2):
        policy.cache.clear()
        responses.append(policy.generate([{'role':'user','content':query_from_row(row)}],reference['schemas'],p['config']['seed']+e['source_row_index']*1009,p['config']))
    assert typed_equal(responses[0][0],responses[1][0]) and responses[0][2]['generated_token_ids']==responses[1][2]['generated_token_ids'],'CPU repeated greedy probe differs'
    old_probe=RUN/'cpu_policy_probe.json'
    if old_probe.exists():assert responses[0][2]['generated_token_ids']==feasibility.read(old_probe)['samples'][0]['audit']['generated_token_ids'],'optimized CPU probe differs from frozen unoptimized output'
    result={'status':'CPU_POLICY_PROBE_PASS','task_id':task,'samples':[{'message':m,'finish':f,'audit':a} for m,f,a in responses]}
    feasibility.immutable(RUN/'cpu_policy_kv_probe_v2.json',result)
    print(stable({'status':result['status'],'samples':[{'seconds':a['model_seconds'],'prompt_tokens':a['prompt_tokens'],'completion_tokens':a['completion_tokens']} for _,_,a in responses]}))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['protocol','worker','probe']);a.add_argument('--shard',type=int);a.add_argument('--limit',type=int);a.add_argument('--threads',type=int,default=4);v=a.parse_args()
    if v.command=='protocol':print(stable(make_collection_protocol()))
    elif v.command=='probe':probe()
    else:worker(v.shard,v.limit,v.threads)
