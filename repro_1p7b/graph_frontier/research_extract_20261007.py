"""Read-only extraction of successful optimizer exposure and rollout features."""
import base64
import collections
import gzip
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def extract():
    base = Path.cwd() / 'repro_1p7b/graph_frontier'
    run = base / 'recursive_opd_v3_experiment_run'
    cpu = base / 'recursive_opd_v3_complete_c_run'
    # Resolve cycle-zero D0 from the same actual schedule source constant.
    source = (base / 'recursive_opd_v3_complete_c/prepare.py').read_text()
    print('SOURCE_CPU_RUN_LINES=' + repr([x for x in source.splitlines() if 'RUN =' in x]), flush=True)
    dataset0 = next(p for p in base.glob('recursive_opd_v3*/*/dataset/dataset.jsonl')
                    if p.parent.parent.name == 'cycle0' and sha(p) == read(run/'cycle0/train_plan.json')['natural_dataset_sha256'])
    protocol = read(base/'recursive_opd_v1_run/protocol.json')
    heldout = set(protocol['heldout_task_ids'])
    metadata, exposures, policies, features = [], [], [], []
    keep = ('task_id','pool','query_hash','complete_c_source_label','complete_c_sample_identity',
            'anchor_sample_identity','target_type','source_verdict','correction_kind',
            'source_frontier_type','source_frontier_depth','source_error_cycle',
            'crossed_cycle','frontier_depth','turn_index','gold_pointer')
    for cycle in range(4):
        directory = run/f'cycle{cycle}'
        plan_path = directory/'train_plan.json'
        plan = read(plan_path)
        checkpoint = run/f'cycle{cycle+1}/checkpoint/trainer_state.json'
        state = read(checkpoint)
        assert state['status']=='TRAINED' and state['train_plan_sha256']==sha(plan_path)
        npath = dataset0 if cycle==0 else directory/'natural/dataset.jsonl'
        apath = directory/'anchors/anchors.jsonl'
        natural = rows(npath)
        anchors = rows(apath) if cycle else []
        assert sha(npath)==plan['natural_dataset_sha256']
        if cycle:
            assert sha(apath)==plan['anchor_dataset_sha256']
        successful=[]
        attempts=[]
        for logdir in sorted(directory.glob('train_metrics_*')):
            result_path=logdir/'result.json'
            log=rows(logdir/'training_metrics.jsonl')
            attempts.append({'path':str(logdir),'steps':len(log),'has_result':result_path.exists()})
            if result_path.exists() and read(result_path)==state:
                assert (logdir/'plan_sha256.txt').read_text().strip()==sha(plan_path)
                assert [r['step'] for r in log]==list(range(1,plan['optimizer_steps']+1))
                assert len(log)*2==len(plan['stream'])
                for step,r in enumerate(log):
                    assert r['exposure_tags']==[plan['stream'][step*2+i]['tag'] for i in range(2)]
                successful.append(logdir)
        assert len(successful)==1, 'Unidentified successful optimizer attempt'
        for index,item in enumerate(plan['stream']):
            row=(natural if item['source']=='natural' else anchors)[item['index']]
            identity=row.get('complete_c_sample_identity') if item['source']=='natural' else row.get('anchor_sample_identity')
            assert identity==item['identity'] and row['task_id'] not in heldout
            exposure={k:row.get(k) for k in keep if k in row}
            exposure.update(cycle=cycle,optimizer_step=index//2+1,rank=index%2,
                            source=item['source'],tag=item['tag'],identity=identity,
                            input_tokens=len(row['input_ids']),target_tokens=sum(x!=-100 for x in row['labels']))
            exposures.append(exposure)
        metadata.append({'cycle':cycle,'plan':plan,'plan_path':str(plan_path),
                         'natural_path':str(npath),'anchor_path':str(apath) if cycle else None,
                         'natural_population_metadata':[{k:r[k] for k in keep if k in r} for r in natural],
                         'successful_attempt':str(successful[0]),'attempts':attempts,
                         'trainer_state':state,'checkpoint_state_path':str(checkpoint)})
    roots=[base/'recursive_opd_v1_run/cycle0']+[run/f'cycle{i}' for i in range(1,5)]
    for cycle,directory in enumerate(roots):
        metrics=rows(directory/'metrics/per_task.jsonl')
        policies.append({'cycle':cycle,'records':[{k:v for k,v in r.items() if k!='prompt_messages'} for r in metrics]})
        for r in metrics:
            task=r['task_id'];episode=read(directory/'rollouts'/f'{task}.json')
            gold=episode['gold'];student=episode.get('student') or {}
            assert episode['task_id']==task
            feature={'cycle':cycle,'task_id':task,'tool_events':sum(len(s.get('typed_events',[])) for s in student.get('steps',[])) if student.get('steps') else None,
                     'gold_action_count':len(gold['actions']),'gold_tools':[a['name'] for a in gold['actions']],
                     'gold_argument_roles':[sorted(a['arguments']) for a in gold['actions']],
                     'observed_graph_depth':'unknown','dependency_completion':'unknown',
                     'premature_stop':r['status']=='FIRST_ERROR' and r.get('error_regime') in {'START','ADVANCE'} and r.get('reason')=='premature model stop',
                     'over_continuation':r['status']=='FIRST_ERROR' and r.get('error_regime')=='TERMINATE'}
            features.append(feature)
        print('EXTRACTED_CYCLE='+str(cycle),flush=True)
    return {'metadata':metadata,'exposures':exposures,'policies':policies,'features':features,
            'train_task_ids':protocol['train_task_ids'],'heldout_task_ids':protocol['heldout_task_ids'],
            'training_source_sha256':sha(base/'recursive_opd_v3_experiment/train.py'),
            'schedule_source_sha256':sha(base/'recursive_opd_v3_experiment/schedule.py')}


if __name__=='__main__':
    data=extract()
    body=base64.b64encode(gzip.compress(json.dumps(data,separators=(',',':')).encode())).decode()
    for index in range(0,len(body),90000):
        print('RESEARCH_CHUNK='+body[index:index+90000],flush=True)
        time.sleep(1)
    print('RESEARCH_DONE='+hashlib.sha256(body.encode()).hexdigest(),flush=True)
