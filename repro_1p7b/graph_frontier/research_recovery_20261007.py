"""Bounded Verified Rollback-and-Correct prototype; NEVER trains a model.

Post-intervention states are explicitly hybrid, NOT pure on-policy states.
Actual erroneous proposals are kept separately from the corrected history.
Rollback restores the pre-error snapshot, never applies gold onto corrupted state.
"""
import argparse
import asyncio
import collections
import copy
import json
import random
import time
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import IsolatedEpisode, atomic_json, typed_equal
from repro_1p7b.graph_frontier.collect_preference_rollouts import generate_action
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, OUT, decode, query_from_row
from repro_1p7b.graph_frontier.recursive_opd_v1.mine import classify
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import RUN as V1, PROTOCOL, CONFIG
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import parse_calls

RUN=V1.parent/'research_recovery_20261007_run'
SEED=20261007


def read(path):
    return json.loads(path.read_text())


def immutable(path,value):
    if path.exists():
        assert read(path)==value, 'Immutable research plan drift'
    else:
        atomic_json(path,value)


def make_plan():
    protocol=read(PROTOCOL)
    metrics=[json.loads(x) for x in (V1/'cycle0/metrics/per_task.jsonl').read_text().splitlines() if x]
    train=set(protocol['train_task_ids'])
    eligible=[r for r in metrics if r['task_id'] in train and r['status']=='FIRST_ERROR']
    rng=random.Random(SEED);chosen=[]
    for regime,quota in [('START',40),('ADVANCE',40),('TERMINATE',20)]:
        group=sorted((r for r in eligible if r['error_regime']==regime),key=lambda x:x['task_id'])
        rng.shuffle(group);chosen+=group[:quota]
    remaining=sorted((r for r in eligible if r not in chosen),key=lambda x:x['task_id'])
    rng.shuffle(remaining);chosen+=remaining[:100-len(chosen)]
    rng.shuffle(chosen)
    model=Path(protocol['pi0_model'])
    assert sha256(model/'model.safetensors')==protocol['pi0_model_sha256']
    plan={'schema':'verified_rollback_recovery_prototype_1','seed':SEED,'model':str(model),
          'model_sha256':protocol['pi0_model_sha256'],'protocol_sha256':sha256(PROTOCOL),
          'pi0_metrics_sha256':sha256(V1/'cycle0/metrics/per_task.jsonl'),
          'config':CONFIG,'max_student_requests':CONFIG['max_turns'],'max_interventions':8,
          'selected_tasks':[{**{k:r[k] for k in ('task_id','environment','error_regime','frontier_depth','student_step')},
                              'episode_sha256':sha256(V1/'cycle0/rollouts'/f"{r['task_id']}.json")} for r in chosen],
          'selection':'100 train-only previously verified FIRST_ERROR, predeclared 40/40/20 quotas with deficit fill; no heldout selection',
          'semantics':'rollback to PRE-error saved state and truncate erroneous history, inject one gold action; post-intervention state distribution is hybrid'}
    assert len(chosen)==100 and not {r['task_id'] for r in chosen}&set(protocol['heldout_task_ids'])
    immutable(RUN/'plan.json',plan)
    return plan


async def cpu_one(item):
    assert sha256(V1/'cycle0/rollouts'/f"{item['task_id']}.json")==item['episode_sha256']
    episode=read(V1/'cycle0/rollouts'/f"{item['task_id']}.json")
    gold=episode['gold'];p=item['frontier_depth']
    expected=gold['initial_state'] if p==0 else gold['states_after'][p-1]
    step=next(s for s in episode['student']['steps'] if s['step_index']==item['student_step'])
    assert typed_equal(step['state_before_action'],expected), 'historical pre-error state differs from gold state'
    env=IsolatedEpisode(item['task_id'],list(expected),expected)
    try:
        await env.open();assert typed_equal(await env.save(),expected), 'pre-error snapshot roundtrip mismatch'
        if p<len(gold['actions']):
            action=gold['actions'][p]
            response,ok,error=await env.call(action['name'],action['arguments'])
            assert ok is True and error is None, f'corrective action failed: {error}'
            assert typed_equal(response,gold['observations'][p]), f'corrective response mismatch: {stable(response)} vs {stable(gold["observations"][p])}'
            after=await env.save();assert typed_equal(after,gold['states_after'][p]), 'corrective post-state mismatch'
        else:
            after=await env.save();assert typed_equal(after,gold['final_state']), 'terminal snapshot mismatch'
    finally:
        await env.close()
    restored=IsolatedEpisode(item['task_id'],list(after),after)
    try:
        await restored.open();assert typed_equal(await restored.save(),after), 'post-corrective snapshot roundtrip mismatch'
    finally:
        await restored.close()
    return {'task_id':item['task_id'],'status':'RESTORE_AND_CORRECT_PASS','pointer':p}


def preflight():
    plan=read(RUN/'plan.json');results=[]
    for index,item in enumerate(plan['selected_tasks']):
        path=RUN/'cpu_preflight'/f"{item['task_id']}.json"
        if path.exists():
            result=read(path)
            assert result['plan_sha256']==sha256(RUN/'plan.json')
        else:
            try:
                result=asyncio.run(cpu_one(item))
            except Exception as exc:
                result={'task_id':item['task_id'],'status':'FAILED','exception':f'{type(exc).__name__}:{exc}'}
            result['plan_sha256']=sha256(RUN/'plan.json')
            atomic_json(path,result)
        results.append(result)
        print(stable({'cpu_preflight':index+1,'status':result['status']}),flush=True)
    summary={'status':'CPU_READY' if all(r['status']=='RESTORE_AND_CORRECT_PASS' for r in results) else 'CPU_GATE_FAILED',
             'passed':sum(r['status']=='RESTORE_AND_CORRECT_PASS' for r in results),'tasks':len(results),
             'plan_sha256':sha256(RUN/'plan.json'),'failures':[r for r in results if r['status']!='RESTORE_AND_CORRECT_PASS']}
    atomic_json(RUN/'cpu_preflight_summary.json',summary)
    assert summary['status']=='CPU_READY', stable(summary)


def diagnose():
    # Preserve the original failed gate and original task-level preflight files.
    plan=read(RUN/'plan.json');failed={r['task_id'] for r in read(RUN/'cpu_preflight_summary.json')['failures']}
    results=[]
    for item in plan['selected_tasks']:
        if item['task_id'] not in failed: continue
        path=RUN/'cpu_diagnostics'/f"{item['task_id']}.json"
        if path.exists(): results.append(read(path));continue
        try:
            result=asyncio.run(cpu_one(item))
        except Exception as exc:
            result={'task_id':item['task_id'],'status':'FAILED','exception':f'{type(exc).__name__}:{exc}'}
        result['plan_sha256']=sha256(RUN/'plan.json');atomic_json(path,result);results.append(result)
        print(stable(result),flush=True)
    atomic_json(RUN/'cpu_diagnostics_summary.json',{'original_gate':'CPU_GATE_FAILED','results':results,'plan_sha256':sha256(RUN/'plan.json')})


async def collect(item,entry,row,endpoint,served):
    assert sha256(V1/'cycle0/rollouts'/f"{item['task_id']}.json")==item['episode_sha256']
    original=read(V1/'cycle0/rollouts'/f"{item['task_id']}.json")
    gold=original['gold'];env=IsolatedEpisode(item['task_id'],list(gold['initial_state']),gold['initial_state'])
    await env.open();schema=copy.deepcopy(env.schemas)
    messages=[{'role':'user','content':query_from_row(row)}]
    steps,proposals,corrections=[],[],[];termination='MAX_TURNS';usage_total=collections.Counter()
    started=time.monotonic();state_mismatches=0;executed=0;attempted=0
    try:
        for request_index in range(CONFIG['max_turns']):
            before=await env.save();count=len(messages)
            message,parsed,usage,finish=generate_action(endpoint,served,messages,schema,
                seed=CONFIG['seed']+entry['source_row_index']*1009+request_index,
                temperature=CONFIG['temperature'],max_tokens=CONFIG['max_tokens_per_turn'])
            for k in ('prompt_tokens','completion_tokens','total_tokens'):
                usage_total[k]+=usage.get(k,0)
            messages.append(message);calls,errors=parse_calls(message);events=[]
            async def execute(call):
                nonlocal executed
                if attempted+call['position']>=CONFIG['tool_call_limit']:
                    response,ok,error='unknown',False,{'type':'ToolCallLimit','message':'fixed call budget reached'}
                else:
                    response,ok,error=await env.call(call['name'],call['arguments']);executed+=1
                return {'tool_name':call['name'],'tool_arguments':call['arguments'],
                               'tool_response':response,'execution_success':ok,'exception':error}
            events=await asyncio.gather(*(execute(call) for call in calls))
            attempted+=len(calls)
            by_position={call['position']:event for call,event in zip(calls,events)}
            bad={error['position']:error for error in errors}
            for position,raw_call in enumerate(message.get('tool_calls') or []):
                if position in by_position:
                    event=by_position[position]
                    value=event['tool_response'] if event['execution_success'] else {'error':event['exception']}
                else:
                    value={'error':bad[position]}
                messages.append({'role':'tool','name':str((raw_call.get('function') or {}).get('name') or 'unknown'),
                    'tool_call_id':str(raw_call.get('id') or f'call_{position}'),'content':stable(value)})
            after=await env.save()
            step={'step_index':len(steps),'prompt_message_count':count,'model_message':message,
                  'structured_calls':calls,'parse_errors':errors,'typed_events':events,
                  'state_before_action':before,'state_after_action':after,'source':'STUDENT',
                  'usage':usage,'finish_reason':finish}
            steps.append(step)
            # A blocked call is infrastructure budget exhaustion, not a learnable
            # semantic error. Preserve its proposal, never create gold supervision.
            if any((e.get('exception') or {}).get('type')=='ToolCallLimit' for e in events):
                proposals.append({'request_index':request_index,'step':copy.deepcopy(step),
                    'verifier':{'status':'BUDGET_EXHAUSTED'},'interventions_before':len(corrections)})
                termination='STUDENT_TOOL_BUDGET';break
            proposed_end=('MODEL_FINAL' if finish!='length' else 'GENERATION_LENGTH') if not message.get('tool_calls') else 'MAX_TURNS'
            student={'steps':steps,'messages':messages,'final_state':after,'termination_reason':proposed_end}
            verdict=await classify({'gold':gold,'student':student},entry,row,schema)
            proposals.append({'request_index':request_index,'step':copy.deepcopy(step),'verifier':verdict,
                              'interventions_before':len(corrections)})
            if verdict['status']=='FIRST_ERROR':
                if verdict['student_step']!=step['step_index']:
                    raise RuntimeError('Verifier points to an earlier unrepaired step')
                if len(corrections)>=8:
                    termination='INTERVENTION_BUDGET';break
                p=verdict['frontier_depth'];expected=gold['initial_state'] if p==0 else gold['states_after'][p-1]
                if not typed_equal(before,expected):
                    state_mismatches+=1;raise RuntimeError('Pre-correction state mismatch')
                await env.close();env=IsolatedEpisode(item['task_id'],list(before),before);await env.open()
                if not typed_equal(await env.save(),before):
                    state_mismatches+=1;raise RuntimeError('Rollback state mismatch')
                messages=messages[:count];steps.pop()
                correction={'pointer':p,'type':verdict['error_regime'],'request_index':request_index,
                            'original_state_provenance':'STUDENT_INITIAL_PATH' if not corrections else 'POST_INTERVENTION_HYBRID',
                            'gold_action':gold['actions'][p] if p<len(gold['actions']) else 'EOS_ONLY',
                            'student_prefix':copy.deepcopy(messages),'state_before':before,
                            'wrong_proposal':message,'wrong_observed_state_after':after,'rollback_verified':True}
                if p==len(gold['actions']):
                    assert typed_equal(before,gold['final_state'])
                    corrections.append(correction);termination='GOLD_CORRECTED_TERMINAL';break
                action=gold['actions'][p];response,ok,error=await env.call(action['name'],action['arguments'])
                newstate=await env.save()
                if not ok or error or not typed_equal(response,gold['observations'][p]) or not typed_equal(newstate,gold['states_after'][p]):
                    state_mismatches+=1;raise RuntimeError('Corrective execution/state mismatch')
                repair_id=f'repair_{len(corrections)}'
                repaired={'role':'assistant','content':'','tool_calls':[{'id':repair_id,'type':'function','function':action}]}
                messages.append(repaired);messages.append({'role':'tool','name':action['name'],'tool_call_id':repair_id,'content':stable(response)})
                steps.append({'step_index':len(steps),'prompt_message_count':count,'model_message':repaired,
                    'structured_calls':[action],'parse_errors':[],
                    'typed_events':[{'tool_name':action['name'],'tool_arguments':action['arguments'],'tool_response':response,'execution_success':True,'exception':None}],
                    'state_before_action':before,'state_after_action':newstate,'source':'GOLD_LOCAL_INTERVENTION'})
                correction.update(corrective_execution_verified=True,state_after=newstate);corrections.append(correction)
            elif verdict['status']=='SUCCESS':
                termination='STUDENT_SUCCESS';break
            elif verdict['status']=='UNCERTAIN' and verdict['reason']!='max-turn exhaustion or no verifiable next decision':
                termination='UNCERTAIN_VERIFIER';break
        final=await env.save()
        return {'task_id':item['task_id'],'status':'PROTOTYPE_COMPLETE','termination':termination,
                'baseline_regime':item['error_regime'],'baseline_pointer':item['frontier_depth'],
                'correction_count':len(corrections),'corrections':corrections,'student_proposals':proposals,
                'corrected_history_steps':steps,'state_mismatches':state_mismatches,
                'student_requests':len(proposals),'student_executed_tool_calls':executed,'student_attempted_calls':attempted,
                'usage':dict(usage_total),'final_state':final,
                'exact_gold_final_state':typed_equal(final,gold['final_state']),
                'success_with_intervention':termination in {'STUDENT_SUCCESS','GOLD_CORRECTED_TERMINAL'},
                'runtime_seconds':time.monotonic()-started,'sampling_semantics':'HYBRID_POST_INTERVENTION',
                'plan_sha256':sha256(RUN/'plan.json')}
    except Exception as exc:
        return {'task_id':item['task_id'],'status':'FAILED','exception':f'{type(exc).__name__}:{exc}',
            'state_mismatches':state_mismatches,'student_proposals':proposals,'corrections':corrections,
            'usage':dict(usage_total),'runtime_seconds':time.monotonic()-started,
            'student_requests':len(proposals),'plan_sha256':sha256(RUN/'plan.json')}
    finally:
        await env.close()


def worker(shard,endpoint,limit=None):
    plan=read(RUN/'plan.json');assert read(RUN/'cpu_preflight_summary.json')['status']=='CPU_READY'
    assert read(RUN/'cpu_preflight_summary.json')['plan_sha256']==sha256(RUN/'plan.json')
    raw=read(SOURCE);entries={e['audit_id']:e for e in read(OUT/'opd_source_manifest.json')['rows']}
    tasks=plan['selected_tasks'][shard::2]
    if limit is not None:tasks=tasks[:limit]
    for n,item in enumerate(tasks):
        path=RUN/'episodes'/f"{item['task_id']}.json"
        if path.exists():
            assert read(path)['plan_sha256']==sha256(RUN/'plan.json');continue
        entry=entries[item['task_id']];row=raw[entry['source_row_index']]
        try:
            result=asyncio.run(collect(item,entry,row,endpoint,'research-recovery-pi0'))
        except Exception as exc:
            result={'task_id':item['task_id'],'status':'FAILED','exception':f'{type(exc).__name__}:{exc}',
                    'plan_sha256':sha256(RUN/'plan.json')}
        atomic_json(path,result);print(stable({'shard':shard,'completed':n+1,'task_id':item['task_id'],
                                             'status':result['status'],'corrections':result.get('correction_count')}),flush=True)


def summarize():
    plan=read(RUN/'plan.json');results=[read(p) for p in sorted((RUN/'episodes').glob('*.json'))]
    good=[r for r in results if r['status']=='PROTOTYPE_COMPLETE']
    points=[c for r in good for c in r['corrections']]
    value={'planned':100,'persisted':len(results),'completed':len(good),'failed':len(results)-len(good),
           'termination_counts':dict(collections.Counter(r['termination'] for r in good)),
           'correction_points':len(points),'points_per_completed_task':len(points)/len(good) if good else None,
           'tasks_with_multiple_corrections':sum(r['correction_count']>1 for r in good),
           'correction_depth_distribution':dict(collections.Counter(c['pointer'] for c in points)),
           'correction_type_distribution':dict(collections.Counter(c['type'] for c in points)),
           'corrections_deeper_than_ordinary_first_error':sum(c['pointer']>r['baseline_pointer'] for r in good for c in r['corrections']),
           'state_mismatches':sum(r.get('state_mismatches',0) for r in results),
           'success_with_intervention':sum(r['success_with_intervention'] for r in good),
           'student_requests':sum(r['student_requests'] for r in good),
           'usage':{k:sum(r['usage'].get(k,0) for r in good) for k in ('prompt_tokens','completion_tokens','total_tokens')},
           'runtime_seconds_sum':sum(r['runtime_seconds'] for r in good),
           'failed_records':[r for r in results if r['status']=='FAILED'],
           'plan_sha256':sha256(RUN/'plan.json'),'no_training':True,
           'caveat':'Conditioned on previous pi0 FIRST_ERROR; intervened success is NOT autonomous student success; subsequent states are hybrid.'}
    atomic_json(RUN/'summary.json',value);print(stable(value),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','preflight','diagnose','worker','summary']);p.add_argument('--shard',type=int,choices=[0,1]);p.add_argument('--endpoint');p.add_argument('--limit',type=int);args=p.parse_args()
    if args.command=='plan':print(stable(make_plan()))
    elif args.command=='preflight':preflight()
    elif args.command=='diagnose':diagnose()
    elif args.command=='worker':worker(args.shard,args.endpoint,args.limit)
    else:summarize()
