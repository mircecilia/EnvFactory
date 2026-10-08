"""CPU bookkeeping mocks: not substitutes for real MCP/policy smoke."""
import ast
import asyncio
import collections
import contextvars
import copy
import hashlib
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


def namespace():
    p=Path(__file__).parents[1]/'verified_recovery_collect.py'
    tree=ast.parse(p.read_text());tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
    n={'asyncio':asyncio,'collections':collections,'contextvars':contextvars,'copy':copy,'time':time,
        'IsolatedEpisode':object,'typed_equal':lambda a,b:type(a)==type(b) and a==b,'stable':json.dumps,
        'COST':contextvars.ContextVar('cost',default=None),'ROLE':contextvars.ContextVar('role',default='VERIFIER'),
        'det':SimpleNamespace(digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()),
        'query_from_row':lambda r:'query'}
    exec(compile(tree,str(p),'exec'),n);return n


class RecoveryFeasibilityTest(unittest.TestCase):
    def case(self,actions,repair_fails=False,budget=11):
        n=namespace();instances=[];gold={'initial_state':{'S':{'p':0}},
            'actions':[{'name':str(i),'arguments':{}} for i in range(1,5)],
            'observations':[{'p':i} for i in range(1,5)],'states_after':[{'S':{'p':i}} for i in range(1,5)],
            'final_state':{'S':{'p':4}}}
        class Env:
            def __init__(self,task,servers,state):self.state=copy.deepcopy(state);self.schemas=[];self.calls=[];instances.append(self)
            async def open(self):pass
            async def close(self):pass
            async def save(self):return copy.deepcopy(self.state)
            async def call(self,name,args):
                n['COST'].get()['ledger'].append({'role':n['ROLE'].get(),'mcp_dispatched':True})
                self.calls.append((name,copy.deepcopy(self.state)))
                self.state['S']['p']=99 if name=='X' or repair_fails else int(name)
                return {'p':self.state['S']['p']},True,None
        class Policy:
            def __init__(self):self.actions=iter(actions)
            def generate(self,*args):
                name=next(self.actions)
                msg={'role':'assistant'}
                if name:msg['tool_calls']=[{'id':'student','function':{'name':name,'arguments':{}}}]
                return msg,'stop',{}
        def parse(msg):return ([{'id':'student','position':i,**x['function']} for i,x in enumerate(msg.get('tool_calls') or [])],[])
        async def check(g,p,step,*args):
            calls=step['structured_calls']
            if calls and calls[0]['name']=='X':return {'status':'FIRST_ERROR','error_regime':'START' if p==0 else 'ADVANCE','reason':'mock verified wrong','student_step':p}
            if not calls:return {'status':'SUCCESS'}
            return {'status':'UNCERTAIN','reason':'max-turn exhaustion or no verifiable next decision','frontier_depth':p+1}
        async def valid(*args):return True
        n.update(CountedEpisode=Env,parse_calls=parse,decision=check,validate_target=valid)
        cfg={'max_turns':12,'tool_call_limit':budget,'seed':0}
        return n,gold,Policy(),cfg,instances

    def run_case(self,actions,**kw):
        n,g,p,c,envs=self.case(actions,**kw)
        result=asyncio.run(n['collect']('t',{'source_row_index':0},{},g,[],p,c,'RECOVERY'))
        return result,envs

    def test_repair_same_post_error_state_history_not_truncated(self):
        r,e=self.run_case(['X','2',None])
        self.assertEqual(len(e),1)
        self.assertEqual(e[0].calls[1],('1',{'S':{'p':99}}))
        self.assertEqual(r['messages'][1]['tool_calls'][0]['function']['name'],'X')
        self.assertEqual(r['successful_interventions'],1)
        self.assertEqual(r['spans'][0]['tool_progress_actions'],1)
        self.assertEqual(r['spans'][0]['actions'],2) # tool + independently correct EOS

    def test_failed_repair_is_unrecoverable_not_free_rollback(self):
        r,e=self.run_case(['X'],repair_fails=True)
        self.assertEqual(r['termination'],'UNRECOVERABLE_LOCAL_CORRECTION')
        self.assertEqual(r['successful_interventions'],0)
        self.assertEqual(r['targets'][0]['inline_repair_success'],False)
        self.assertEqual(len(e),1)

    def test_three_interventions_no_fourth_gold_action(self):
        r,e=self.run_case(['X','X','X','X'])
        self.assertEqual(r['termination'],'INTERVENTION_LIMIT')
        self.assertEqual(r['successful_interventions'],3)
        self.assertEqual(len(r['targets']),3)
        self.assertEqual([x[0] for x in e[0].calls],['X','1','X','2','X','3','X'])
        self.assertTrue(all(not s['censored'] and s['actions']==0 for s in r['spans']))

    def test_no_budget_no_synthetic_tool_call_or_intervention(self):
        r,e=self.run_case(['X'],budget=1)
        self.assertEqual(r['termination'],'NO_BUDGET_FOR_CORRECTION')
        self.assertEqual(r['environment_calls'],1)
        self.assertEqual(r['successful_interventions'],0)

    def test_budget_censored_span_is_not_immediate_failure(self):
        r,e=self.run_case(['X','2'],budget=2)
        self.assertEqual(r['termination'],'TOOL_BUDGET')
        self.assertEqual(r['environment_calls'],2)
        self.assertTrue(r['spans'][0]['censored'])
        self.assertEqual(r['spans'][0]['ended_by'],'TOOL_BUDGET')

    def test_state_identity_does_not_fake_new_states_from_pointer(self):
        n=namespace();states=[]
        n['add_state'](states,{'S':{'p':0}},0,'INITIAL')
        n['add_state'](states,{'S':{'p':0}},3,'GOLD_LOCAL_CORRECTION')
        self.assertEqual(states[0]['state_hash'],states[1]['state_hash'])


if __name__=='__main__':unittest.main()
