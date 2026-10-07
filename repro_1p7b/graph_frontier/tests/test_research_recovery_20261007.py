"""CPU mocks test rollback bookkeeping, not replace the real executor preflight."""
import ast
import asyncio
import collections
import copy
import json
import time
import unittest
from pathlib import Path


def load_functions():
    source=Path(__file__).parents[1]/'research_recovery_20261007.py'
    tree=ast.parse(source.read_text())
    tree.body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))]
    namespace={'asyncio':asyncio,'collections':collections,'copy':copy,'time':time,
               'typed_equal':lambda a,b:type(a)==type(b) and a==b,'stable':json.dumps,
               'sha256':lambda p:'frozen','RUN':Path('/mock/run'),'V1':Path('/mock/v1'),
               'CONFIG':{'max_turns':12,'tool_call_limit':11,'seed':0,'temperature':0,'max_tokens_per_turn':100},
               'query_from_row':lambda r:'query'}
    exec(compile(tree,str(source),'exec'),namespace)
    return namespace


class RecoveryMockTest(unittest.TestCase):
    def setup_case(self,mismatch=False,budget=11):
        n=load_functions();n['CONFIG']['tool_call_limit']=budget
        gold={'initial_state':{'S':{'p':0}},'actions':[{'name':'A','arguments':{}},{'name':'B','arguments':{}}],
              'observations':[{'p':1},{'p':2}],'states_after':[{'S':{'p':1}},{'S':{'p':2}}],
              'final_state':{'S':{'p':2}}}
        instances=[]
        class Env:
            def __init__(self,task,servers,state):
                self.state=copy.deepcopy(state);self.schemas=[];instances.append(self)
            async def open(self): pass
            async def close(self): pass
            async def save(self): return copy.deepcopy(self.state)
            async def call(self,name,args):
                self.state['S']['p']={'A':1,'B':2}.get(name,99)
                return {'p':-1 if mismatch and name=='A' else self.state['S']['p']},True,None
        replies=iter(['X','X',None])
        def generate(*a,**kw):
            name=next(replies)
            return ({'role':'assistant','tool_calls':[{'id':'x','function':{'name':name,'arguments':{}}}]} if name else {'role':'assistant'},None,{'total_tokens':5},'stop')
        def parse(message):
            return ([{'position':i,'id':'x',**r['function']} for i,r in enumerate(message.get('tool_calls') or [])],[])
        async def classify(episode,*args):
            last=episode['student']['steps'][-1];p=len(episode['student']['steps'])-1
            if last['structured_calls'] and last['structured_calls'][0]['name']=='X':
                return {'status':'FIRST_ERROR','student_step':last['step_index'],'frontier_depth':p,
                        'error_regime':'START' if p==0 else 'ADVANCE'}
            return {'status':'SUCCESS'}
        n.update(IsolatedEpisode=Env,generate_action=generate,parse_calls=parse,classify=classify,
                 read=lambda p:{'gold':gold})
        item={'task_id':'t','episode_sha256':'frozen','frontier_depth':0,'error_regime':'START'}
        return n,item,instances

    def test_two_errors_rollback_before_gold_and_keep_wrong_proposals(self):
        n,item,instances=self.setup_case()
        result=asyncio.run(n['collect'](item,{'source_row_index':0},{},'endpoint','model'))
        self.assertEqual(result['status'],'PROTOTYPE_COMPLETE')
        self.assertEqual(result['correction_count'],2)
        self.assertEqual([c['pointer'] for c in result['corrections']],[0,1])
        self.assertEqual([c['original_state_provenance'] for c in result['corrections']],
                         ['STUDENT_INITIAL_PATH','POST_INTERVENTION_HYBRID'])
        self.assertEqual(result['corrected_history_steps'][0]['state_before_action'],{'S':{'p':0}})
        self.assertEqual(result['corrected_history_steps'][1]['state_before_action'],{'S':{'p':1}})
        self.assertEqual(result['student_proposals'][0]['step']['state_after_action'],{'S':{'p':99}})
        self.assertEqual(result['state_mismatches'],0)
        self.assertEqual(len(instances),3)

    def test_real_corrective_response_mismatch_is_failed_not_hidden(self):
        n,item,_=self.setup_case(mismatch=True)
        result=asyncio.run(n['collect'](item,{'source_row_index':0},{},'endpoint','model'))
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['state_mismatches'],1)
        self.assertEqual(len(result['student_proposals']),1)

    def test_budget_exhaustion_never_becomes_correction(self):
        n,item,_=self.setup_case(budget=0)
        result=asyncio.run(n['collect'](item,{'source_row_index':0},{},'endpoint','model'))
        self.assertEqual(result['termination'],'STUDENT_TOOL_BUDGET')
        self.assertEqual(result['correction_count'],0)
        self.assertEqual(result['student_executed_tool_calls'],0)

    def test_exposure_analysis_does_not_call_action_count_graph_depth(self):
        from repro_1p7b.graph_frontier.research_analysis_20261007 import stratum
        r={'cycle':0,'regime':'ADVANCE','prev_position':1,'gold_action_count':3,'graph_depth':'unknown'}
        self.assertEqual(stratum(r),(0,'ADVANCE',1,3))


if __name__=='__main__': unittest.main()
