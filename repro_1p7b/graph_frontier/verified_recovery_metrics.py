"""Task-paired, call-accounted feasibility metrics; never runs a model.

Bootstrap draws paired tasks, recomputing pooled call-normalized yields, not
independent arms or preselected corrected tasks. Gate cannot pass incomplete.
"""
import argparse
import collections
import json
import math
import statistics
from pathlib import Path

import numpy as np
from repro_1p7b.graph_frontier import verified_recovery_feasibility as f
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import atomic_json,typed_equal
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256,stable

RUN=f.RUN


def avg(xs):return statistics.mean(xs) if xs else None
def median(xs):return statistics.median(xs) if xs else None
def rate(n,d):return 100*n/d if d else 0.0
def interval(xs):return [float(x) for x in np.quantile(xs,[0.025,0.975])]


def state_difference(a,b):
    before={s['state_hash'] for s in a['verified_states']};new={}
    for s in b['verified_states']:
        if s['state_hash'] not in before and s['state_hash'] not in new:new[s['state_hash']]=s
    student=[s for s in new.values() if s['origin']=='STUDENT']
    deeper=[s for s in student if s['pointer']>a['max_verified_frontier_depth']]
    return len(new),len(student),len(deeper),list(new.values())


def task_metrics(episode):
    a,b=episode['arms']['FD'],episode['arms']['RECOVERY']
    n,s,d,states=state_difference(a,b)
    values={}
    for name,arm in (('FD',a),('RECOVERY',b)):
        depths=[t['pointer'] for t in arm['targets']]
        values[name]={'corrections':len(depths),'deep_ge2':sum(x>=2 for x in depths),'deep_ge3':sum(x>=3 for x in depths),
            'calls':arm['environment_calls'],'actual_mcp_calls':arm['actual_mcp_domain_calls'],'admin_calls':arm['admin_mcp_calls'],
            'max_frontier':arm['max_verified_frontier_depth'],'termination':arm['termination'],
            'successful_inline_interventions':arm['successful_interventions'],
            'new_verified_states':n if name=='RECOVERY' else 0,
            'new_student_reached_states':s if name=='RECOVERY' else 0,
            'new_student_reached_deeper_states':d if name=='RECOVERY' else 0,
            'corrections_per100calls':rate(len(depths),arm['environment_calls']),
            'deep_per100calls':rate(sum(x>=2 for x in depths),arm['environment_calls']),
            'new_states_per100calls':rate(n if name=='RECOVERY' else 0,arm['environment_calls']),
            'wall_seconds':arm['wall_seconds'],'source_row_index':episode['source_row_index']}
    return {'task_id':episode['task_id'],'arms':values,'new_states':states,
        'completed_spans':[s for s in b['spans'] if not s['censored']],
        'censored_spans':[s for s in b['spans'] if s['censored']]}


def aggregate(episodes,rows):
    value={}
    for name in ('FD','RECOVERY'):
        arms=[e['arms'][name] for e in episodes];targets=[t for a in arms for t in a['targets']]
        depths=[t['pointer'] for t in targets];calls=sum(a['environment_calls'] for a in arms)
        cpu=sum(s['model_audit']['model_seconds'] for a in arms for s in a['steps'])+sum(s['model_audit']['model_seconds'] for a in arms for s in a['unexecuted_proposals'])
        spans=[s for a in arms for s in a['spans'] if not s['censored']]
        lengths=[s['actions'] for s in spans];tools=[s['tool_progress_actions'] for s in spans]
        new=sum(r['arms'][name]['new_verified_states'] for r in rows)
        student_new=sum(r['arms'][name]['new_student_reached_states'] for r in rows)
        student_deep=sum(r['arms'][name]['new_student_reached_deeper_states'] for r in rows)
        value[name]={'tasks':len(arms),'verified_correction_targets':len(targets),'mean_corrections_per_task':len(targets)/len(arms),
            'mean_correction_depth':avg(depths),'median_correction_depth':median(depths),
            'depth_ge2_corrections':sum(x>=2 for x in depths),'depth_ge3_corrections':sum(x>=3 for x in depths),
            'corrections_per100_environment_calls':rate(len(targets),calls),
            'deep_ge2_per100_environment_calls':rate(sum(x>=2 for x in depths),calls),
            'deep_ge3_per100_environment_calls':rate(sum(x>=3 for x in depths),calls),
            'environment_calls':calls,'actual_mcp_domain_calls':sum(a['actual_mcp_domain_calls'] for a in arms),
            'admin_mcp_calls':sum(a['admin_mcp_calls'] for a in arms),
            'call_roles':dict(collections.Counter(x['role'] for a in arms for x in a['environment_call_ledger'])),
            'successful_inline_interventions':sum(a['successful_interventions'] for a in arms),
            'new_verified_states':new,'new_student_reached_states':student_new,'new_student_reached_deeper_states':student_deep,
            'mean_new_states_per_task':new/len(arms),'median_new_states_per_task':median([r['arms'][name]['new_verified_states'] for r in rows]),
            'tasks_with_new_state':sum(r['arms'][name]['new_verified_states']>0 for r in rows),
            'fraction_tasks_with_new_state':sum(r['arms'][name]['new_verified_states']>0 for r in rows)/len(arms),
            'new_verified_states_per100_environment_calls':rate(new,calls),
            'termination_counts':dict(collections.Counter(a['termination'] for a in arms)),
            'uncertain_verifier_rate':sum(a['termination']=='UNCERTAIN_VERIFIER' for a in arms)/len(arms),
            'unrecoverable_rate':sum(a['termination'].startswith('UNRECOVERABLE') for a in arms)/len(arms),
            'mean_max_verified_frontier':avg([a['max_verified_frontier_depth'] for a in arms]),
            'model_cpu_seconds_sum':cpu,'environment_and_model_seconds_sum':sum(a['wall_seconds'] for a in arms),
            'fresh_input_cache_hits':sum(bool(s['model_audit'].get('cached_identical_fresh_input')) for a in arms for s in a['steps']),
            'post_recovery_span':{'completed':len(lengths),'censored':sum(s['censored'] for a in arms for s in a['spans']),
                'mean':avg(lengths),'median':median(lengths),
                'P75':float(np.quantile(lengths,.75)) if lengths else None,'P90':float(np.quantile(lengths,.90)) if lengths else None,
                'zero_step_ratio':sum(x==0 for x in lengths)/len(lengths) if lengths else None,
                'tools_only_mean':avg(tools),'tools_only_zero_ratio':sum(x==0 for x in tools)/len(tools) if tools else None,
                'all_span_lower_bound_mean':avg([s['actions'] for a in arms for s in a['spans']])}}
    a=value['FD'];b=value['RECOVERY']
    value['corrections_per_task_ratio']=b['mean_corrections_per_task']/a['mean_corrections_per_task'] if a['mean_corrections_per_task'] else None
    value['pooled_yield_ratio']=b['corrections_per100_environment_calls']/a['corrections_per100_environment_calls'] if a['corrections_per100_environment_calls'] else None
    return value


def paired_statistics(rows,seed=20261009,repeats=10000):
    metrics={};rng=np.random.default_rng(seed);n=len(rows)
    indices=rng.integers(0,n,(repeats,n))
    def arrays(name,key):return np.array([r['arms'][name][key] for r in rows],dtype=float)
    ac,bc=arrays('FD','calls'),arrays('RECOVERY','calls')
    for title,key,per_task in [('corrections_per100calls','corrections','corrections_per100calls'),
                              ('deep_per100calls','deep_ge2','deep_per100calls'),
                              ('new_states_per100calls','new_verified_states','new_states_per100calls')]:
        av,bv=arrays('FD',key),arrays('RECOVERY',key)
        diff=arrays('RECOVERY',per_task)-arrays('FD',per_task)
        # paired task resampling, pooled rates recomputed with sampled costs
        ad,bd=ac[indices].sum(1),bc[indices].sum(1)
        boot=100*np.divide(bv[indices].sum(1),bd,out=np.zeros(repeats),where=bd>0)-100*np.divide(av[indices].sum(1),ad,out=np.zeros(repeats),where=ad>0)
        metrics[title]={'paired_mean_difference':float(diff.mean()),'paired_mean_difference_CI95':interval(diff[indices].mean(1)),
            'median_difference':float(np.median(diff)),'win':int((diff>1e-12).sum()),'tie':int((abs(diff)<=1e-12).sum()),'loss':int((diff<-1e-12).sum()),
            'pooled_rate_difference':rate(float(bv.sum()),float(bc.sum()))-rate(float(av.sum()),float(ac.sum())),
            'pooled_rate_difference_CI95':interval(boot),'zero_call_tasks_FD':int((ac==0).sum()),'zero_call_tasks_Recovery':int((bc==0).sum())}
    for key in ('max_frontier','new_student_reached_deeper_states'):
        diff=arrays('RECOVERY',key)-arrays('FD',key)
        metrics[key]={'paired_mean_difference':float(diff.mean()),'paired_mean_difference_CI95':interval(diff[indices].mean(1)),
            'median_difference':float(np.median(diff)),'win':int((diff>0).sum()),'tie':int((diff==0).sum()),'loss':int((diff<0).sum())}
    return metrics


def summarize(expected=None):
    p,_=f.checked();lock=f.read(RUN/'collection_protocol.json');ids=p['task_ids']
    if expected is not None:ids=ids[0::2][:expected//2]+ids[1::2][:expected//2]
    episodes=[f.read(RUN/'paired_episodes'/f'{t}.json') for t in ids if (RUN/'paired_episodes'/f'{t}.json').exists()]
    assert all(e['collection_protocol_sha256']==sha256(RUN/'collection_protocol.json') for e in episodes)
    failures=[{k:e[k] for k in ('task_id','status','exception')} for e in episodes if e['status']!='COMPLETE']
    status='COMPLETE' if len(episodes)==len(ids) and not failures else 'INCOMPLETE_OR_FAILED'
    good=[e for e in episodes if e['status']=='COMPLETE'];rows=[task_metrics(e) for e in good]
    value={'status':status,'planned':len(ids),'persisted':len(episodes),'completed':len(good),'failures':failures,
        'subset_scope':p['scope'],'excluded':p['excluded'],'original_full_gate':'DETERMINISM_FAILED',
        'collection_protocol_sha256':sha256(RUN/'collection_protocol.json'),'no_gpu':True,'no_training':True}
    if good:
        value['aggregate']=aggregate(good,rows)
        if expected is None and status=='COMPLETE':
            statistics_=paired_statistics(rows);value['paired_statistics']=statistics_
            complete_spans=value['aggregate']['RECOVERY']['post_recovery_span']
            criteria={'A':statistics_['corrections_per100calls']['pooled_rate_difference_CI95'][0]>0,
                'B':statistics_['deep_per100calls']['pooled_rate_difference_CI95'][0]>0,
                'C':complete_spans['tools_only_zero_ratio'] is not None and complete_spans['tools_only_zero_ratio']<.5,
                'D':statistics_['new_student_reached_deeper_states']['paired_mean_difference_CI95'][0]>0}
            value['criteria']=criteria;value['verdict']='DATA-GO' if criteria['A'] and criteria['B'] and (criteria['C'] or criteria['D']) else 'NO-GO'
            random_rows=[r for r in rows if r['task_id'] not in set(p['task_ids'][:100])]
            value['random299_sensitivity']=paired_statistics(random_rows)
            value['physical_state_definition']='canonical environment snapshot identity; pointer-only novelty not counted; new states first reached by teacher are not student-reached D evidence'
            # Full selected trajectories are runtime artifacts, excluded from Git.
            ordered=sorted(zip(rows,good),key=lambda pair:(pair[0]['arms']['RECOVERY']['max_frontier']-pair[0]['arms']['FD']['max_frontier'],
                pair[0]['arms']['RECOVERY']['new_student_reached_states'],pair[0]['arms']['RECOVERY']['corrections']-pair[0]['arms']['FD']['corrections']),reverse=True)
            better=[pair for pair in ordered if pair[0]['arms']['RECOVERY']['max_frontier']>pair[0]['arms']['FD']['max_frontier']]
            best=better[:10];used={r['task_id'] for r,e in best}
            worse=[pair for pair in reversed(ordered) if pair[0]['task_id'] not in used][:10]
            case_manifest=[]
            for group,pairs in [('frontier_improved',best),('no_help_or_worse',worse)]:
                for r,e in pairs:
                    path=RUN/'cases'/f"{group}_{r['task_id']}.json"
                    atomic_json(path,{'case_group':group,'metrics':r,'full_paired_episode':e,'manual_review':'PENDING'})
                    case_manifest.append({'task_id':r['task_id'],'group':group,'path':str(path),'sha256':sha256(path)})
            value['case_manifest']=case_manifest
            value['case_shortfall']=max(0,10-len(best))
            (RUN/'paired_task_results.jsonl').write_text(''.join(stable(r)+'\n' for r in rows))
            (RUN/'recovery_examples.jsonl').write_text(''.join(stable(e)+'\n' for r,e in best+worse))
    path=RUN/('paired_metrics.json' if expected is None else f'smoke{expected}_metrics.json')
    atomic_json(path,value);print(stable({k:v for k,v in value.items() if k not in ('excluded','aggregate','paired_statistics','case_manifest','random299_sensitivity')}))
    if status!='COMPLETE':raise SystemExit(3)
    return value


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--expected',type=int);v=a.parse_args();summarize(v.expected)
