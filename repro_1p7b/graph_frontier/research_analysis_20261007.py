"""Exposure-conditioned analysis. No model/optimizer/environment execution."""
import base64
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

HERE=Path(__file__).parent
REPORTS=HERE/'reports'
SEED=20261007
BOOTSTRAPS=5000
PERMUTATIONS=5000


def unpack():
    envelope=json.loads((REPORTS/'research_assets_envelope_v2_20261007.json').read_text())
    assert hashlib.sha256(envelope['document_gzip_base64'].encode()).hexdigest()==envelope['sha256']
    return json.loads(gzip.decompress(base64.b64decode(envelope['document_gzip_base64'])))


def summarize(rows, outcome):
    valid=[r for r in rows if r[outcome] is not None]
    yes=sum(r[outcome] for r in valid)
    return {'positive':yes,'valid':len(valid),'unknown':len(rows)-len(valid),
            'rate':yes/len(valid) if valid else None}


def bootstrap_difference(a,b,outcome):
    # Resample TASK clusters, keeping all their repeated cycle observations.
    task_ids=sorted({r['task_id'] for r in a+b})
    index={t:i for i,t in enumerate(task_ids)}
    stats=np.zeros((len(task_ids),4))
    for group,rows in enumerate((a,b)):
        for r in rows:
            if r[outcome] is not None:
                stats[index[r['task_id']],group*2]+=int(r[outcome])
                stats[index[r['task_id']],group*2+1]+=1
    rng=np.random.default_rng(SEED)
    diffs=[]
    for _ in range(BOOTSTRAPS):
        total=stats[rng.integers(0,len(task_ids),len(task_ids))].sum(axis=0)
        if total[1] and total[3]:
            diffs.append(total[0]/total[1]-total[2]/total[3])
    return np.quantile(diffs,[.025,.975]).tolist() if diffs else None


def comparison(a,b,outcome,bootstrap=True):
    x,y=summarize(a,outcome),summarize(b,outcome)
    if x['rate'] is None or y['rate'] is None:
        return {'exposed':x,'unexposed':y,'difference':None}
    return {'exposed':x,'unexposed':y,'difference':x['rate']-y['rate'],
            'relative_difference':x['rate']/y['rate']-1 if y['rate'] else None,
            'risk_ratio':x['rate']/y['rate'] if y['rate'] else None,
            'task_cluster_bootstrap_95ci':bootstrap_difference(a,b,outcome) if bootstrap else None,
            'fisher_two_sided_p_descriptive':float(fisher_exact([[x['positive'],x['valid']-x['positive']],
                    [y['positive'],y['valid']-y['positive']]]).pvalue)}


def stratum(r):
    # Action count is NOT dependency graph depth. No missing graph reconstruction.
    return (r['cycle'],r['regime'],r['prev_position'],r['gold_action_count'])


def adjusted(a,b,outcome,environment=False):
    a=[r for r in a if r[outcome] is not None]
    b=[r for r in b if r[outcome] is not None]
    groups=collections.defaultdict(lambda:[[],[]])
    for g,rs in enumerate((a,b)):
        for r in rs:
            key=stratum(r)+(r['environment'],) if environment else stratum(r)
            groups[key][g].append(r)
    groups=[g for g in groups.values() if g[0] and g[1]]
    n=sum(len(g[0]) for g in groups)
    if not n:
        return {'matched_exposed':0,'difference':None}
    d=sum(len(x)*(np.mean([r[outcome] for r in x])-np.mean([r[outcome] for r in y])) for x,y in groups)/n
    # Conditional label randomization: descriptive association only; not causal
    # under unmeasured confounding or other-exposure interference.
    rng=np.random.default_rng(SEED)
    null=np.zeros(PERMUTATIONS)
    for x,y in groups:
        vals=np.array([r[outcome] for r in x+y],dtype=float)
        na=len(x)
        for k in range(PERMUTATIONS):
            shuffled=rng.permutation(vals)
            null[k]+=na*(shuffled[:na].mean()-shuffled[na:].mean())/n
    p=(1+sum(abs(null)>=abs(d)))/(PERMUTATIONS+1)
    return {'matched_exposed':n,'all_exposed_valid':len(a),'discarded_exposed_no_common_support':len(a)-n,
            'matched_control_count':sum(len(y) for x,y in groups),'strata':len(groups),
            'difference':float(d),'conditional_permutation_p_descriptive':float(p)}


def run():
    assets=unpack()
    paired=json.loads((REPORTS/'paired_fd_correction_audit_20261007.json').read_text())
    policies=[{r['task_id']:r for r in p['records']} for p in assets['policies']]
    features={(r['cycle'],r['task_id']):r for r in assets['features']}
    exposure=collections.defaultdict(list)
    for e in assets['exposures']:
        exposure[e['cycle'],e['task_id']].append(e)
    eligible=[]
    for m in assets['metadata']:
        eligible.append({r['task_id'] for r in m['natural_population_metadata']
                         if r.get('complete_c_source_label')=='CURRENT_FRONTIER_CORRECTION'})
    rows=[]
    for r in paired['rows']:
        cycle=0 if r['transition']=='pi0->Cpi1' else int(r['transition'][3])
        task=r['task_id'];old=policies[cycle][task];new=policies[cycle+1][task]
        actual=exposure[cycle,task]
        main=[e for e in actual if e.get('complete_c_source_label')=='CURRENT_FRONTIER_CORRECTION']
        other=[e for e in actual if e not in main]
        assert not main or r['pool']=='train'
        for e in main:
            assert e['source_verdict']=='FIRST_ERROR'
            assert e['frontier_depth']==old['frontier_depth']
            assert e['correction_kind']==old['error_regime']
            assert e['query_hash']==old['query_hash']
        verified=(old['status']!='UNCERTAIN' and new['status']!='UNCERTAIN')
        delta=new['frontier_depth']-old['frontier_depth'] if verified else None
        target_index=min(r['prev_position'],len(features[cycle,task]['gold_tools'])-1)
        f=features[cycle,task]
        motif=json.dumps((f['gold_tools'][max(0,target_index-1):target_index+1],
                          f['gold_argument_roles'][target_index],r['regime']),separators=(',',':'))
        rows.append({**r,'cycle':cycle,'main_exposures':len(main),'other_exposures':len(other),
                     'all_exposure_sources':json.dumps([(e['tag'],e['source'],e.get('complete_c_source_label')) for e in actual]),
                     'exposed_main':bool(main),'candidate_main_eligible':task in eligible[cycle],
                     'gold_action_count':f['gold_action_count'],'graph_depth':'unknown','environment':old['environment'],
                     'motif_proxy':motif,'exact_crossed':r['outcome']=='CORRECTED' if r['outcome']!='UNCERTAIN' else None,
                     'progressed':delta>0 if delta is not None else None,
                     'same_frontier':delta==0 if delta is not None else None,
                     'regressed':delta<0 if delta is not None else None,
                     'final_success':new['status']=='SUCCESS' if new['status']!='UNCERTAIN' else None,
                     'next_state':new['error_regime'] or new['status'],
                     'old_state':old['error_regime'] or old['status']})
    metrics=['exact_crossed','progressed','same_frontier','regressed','final_success']
    train=[r for r in rows if r['pool']=='train']
    a=[r for r in train if r['exposed_main']];b=[r for r in train if not r['exposed_main']]
    result={'seed':SEED,'bootstrap_replicates':BOOTSTRAPS,'permutation_replicates':PERMUTATIONS,
            'scope':'MAIN correction exposure; primary controls=train only; heldout separate',
            'exposure_proof':'successful checkpoint trainer_state + matching attempt result, contiguous optimizer steps, per-rank exposure_tags + immutable plan/source row identity; log does not include per-example task IDs',
            'selection':'128 natural rows sampled uniformly without replacement; possible one parity filler; every previous crossed anchor exposed once, not randomized',
            'main_exposed_task_transitions':len(a),'unexposed_train_task_transitions':len(b),
            'exposed_main_count_distribution':dict(collections.Counter(r['main_exposures'] for r in a)),
            'unexposed_with_other_exposure':sum(r['other_exposures']>0 for r in b),
            'comparison':{m:comparison(a,b,m) for m in metrics},
            'per_cycle':{},'by_regime':{},'by_frontier_depth':{},
            'matched_eligible':{m:adjusted(a,[r for r in b if r['candidate_main_eligible']],m) for m in metrics},
            'clean_unexposed_sensitivity':{m:comparison(a,[r for r in b if not r['other_exposures'] and r['candidate_main_eligible']],m) for m in metrics},
            'cycle_summaries':[],'transition_matrices':{},'motif_transfer_proxy':{}}
    for cycle in range(4):
        x=[r for r in a if r['cycle']==cycle];y=[r for r in b if r['cycle']==cycle]
        result['per_cycle'][str(cycle)]={'exposed_raw':len(x),'unexposed_raw':len(y),
                'comparison':{m:comparison(x,y,m,False) for m in metrics},
                'heldout':{m:summarize([r for r in rows if r['cycle']==cycle and r['pool']=='heldout'],m) for m in metrics}}
        matrix=collections.Counter()
        allmatrix=collections.Counter()
        for task in set(policies[cycle])&set(policies[cycle+1]):
            old=policies[cycle][task];new=policies[cycle+1][task]
            allmatrix[(old['error_regime'] or old['status'],new['error_regime'] or new['status'])]+=1
        result['transition_matrices'][str(cycle)]={f'{a}->{b}':n for (a,b),n in allmatrix.items()}
        # Exploratory exact tool/argument-role proxy; not an actual graph motif.
        trained_motifs={r['motif_proxy'] for r in x}
        groups={'same_task_exposed':x,
                'same_motif_unexposed_train':[r for r in y if r['motif_proxy'] in trained_motifs],
                'unrelated_unexposed_train':[r for r in y if r['motif_proxy'] not in trained_motifs],
                'same_motif_heldout':[r for r in rows if r['cycle']==cycle and r['pool']=='heldout' and r['motif_proxy'] in trained_motifs],
                'unrelated_heldout':[r for r in rows if r['cycle']==cycle and r['pool']=='heldout' and r['motif_proxy'] not in trained_motifs]}
        result['motif_transfer_proxy'][str(cycle)]={g:{'raw':len(rs),**{m:summarize(rs,m) for m in metrics}} for g,rs in groups.items()}
        result['motif_transfer_proxy'][str(cycle)]['heldout_matched_progression']=adjusted(groups['same_motif_heldout'],groups['unrelated_heldout'],'progressed')
        result['motif_transfer_proxy'][str(cycle)]['heldout_environment_matched_progression']=adjusted(groups['same_motif_heldout'],groups['unrelated_heldout'],'progressed',environment=True)
        result['motif_transfer_proxy'][str(cycle)]['train_environment_matched_progression']=adjusted(groups['same_motif_unexposed_train'],groups['unrelated_unexposed_train'],'progressed',environment=True)
    for regime in ('START','ADVANCE','TERMINATE'):
        result['by_regime'][regime]={m:comparison([r for r in a if r['regime']==regime],[r for r in b if r['regime']==regime],m,False) for m in metrics}
    for depth in sorted({r['prev_position'] for r in train}):
        result['by_frontier_depth'][str(depth)]={m:comparison([r for r in a if r['prev_position']==depth],[r for r in b if r['prev_position']==depth],m,False) for m in metrics}
    for cycle,p in enumerate(policies):
        valid=[r for r in p.values() if r['status']!='UNCERTAIN']
        fs=[features[cycle,t] for t in p]
        calls=[f['tool_events'] for f in fs if f['tool_events'] is not None]
        result['cycle_summaries'].append({'cycle':cycle,'state_distribution':dict(collections.Counter(r['error_regime'] or r['status'] for r in p.values())),
            'frontier_depth_all_saved_distribution':dict(collections.Counter(r['frontier_depth'] for r in p.values())),
            'verified_frontier_depth_distribution':dict(collections.Counter(r['frontier_depth'] for r in valid)),
            'verified_frontier_mean':float(np.mean([r['frontier_depth'] for r in valid])),
            'verified_first_error_depth_distribution':dict(collections.Counter(r['frontier_depth'] for r in valid if r['status']=='FIRST_ERROR')),
            'gold_progress_fraction_mean':float(np.mean([r['frontier_depth']/r['gold_tool_count'] for r in valid])),
            'dependency_completion':'unknown: no authoritative dependency graph on these Official-RL tasks',
            'tool_events_observed_tasks':len(calls),'tool_events_mean':float(np.mean(calls)),
            'tool_events_distribution':dict(collections.Counter(calls)),
            'premature_stop':sum(f['premature_stop'] for f in fs),'over_continuation':sum(f['over_continuation'] for f in fs)})
    result['global_repeated_task_fisher_warning']='Fisher is descriptive; repeated tasks violate independence. Cluster bootstrap is primary CI. Multiple exploratory tests are not confirmatory.'
    (REPORTS/'research_exploration_20261007.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (REPORTS/'research_assets_20261007.json').write_text(json.dumps(assets,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    with (REPORTS/'research_exposure_pairs_20261007.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (REPORTS/'research_optimizer_exposures_20261007.csv').open('w',newline='',encoding='utf-8') as f:
        fields=sorted(set().union(*(e.keys() for e in assets['exposures'])))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(assets['exposures'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    states=['START','ADVANCE','TERMINATE','SUCCESS','UNCERTAIN']
    bottoms=np.zeros(5)
    for state in states:
        vals=[c['state_distribution'].get(state,0) for c in result['cycle_summaries']]
        axes[0].bar(range(5),vals,bottom=bottoms,label=state);bottoms+=vals
    axes[0].set_xticks(range(5),['pi0','Cpi1','Cpi2','Cpi3','Cpi4']);axes[0].legend(fontsize=7);axes[0].set_title('All-task outcome distribution')
    for c in result['cycle_summaries']:
        distribution=c['verified_frontier_depth_distribution'];total=sum(distribution.values())
        xs=sorted(int(x) for x in distribution)
        ys=[sum(n for k,n in distribution.items() if int(k)<=x)/total for x in xs]
        axes[1].step(xs,ys,where='post',label=f"cycle{c['cycle']}")
    axes[1].legend(fontsize=7);axes[1].set_title('Verified frontier CDF (UNCERTAIN excluded)');axes[1].set_xlabel('Gold action pointer; NOT graph depth')
    fig.tight_layout();fig.savefig(REPORTS/'research_frontier_distributions_20261007.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('main_exposed_task_transitions','unexposed_train_task_transitions','unexposed_with_other_exposure','comparison','matched_eligible','clean_unexposed_sensitivity')},indent=2))


if __name__=='__main__':
    run()
