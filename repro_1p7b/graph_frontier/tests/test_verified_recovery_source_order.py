"""Mock verifies prefix is physically replayed/charged and live env is untouched."""
import ast
import asyncio
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace


class SourceOrderTest(unittest.TestCase):
    def test_student_initial_order_comes_from_authoritative_source(self):
        class Base:
            def __init__(self,task,servers,initial):self.initial=initial
        source={'S':{'issues':{'342':{'x':1},'330':{'x':2}}}}
        canonical={'S':{'issues':{'330':{'x':2},'342':{'x':1}}}}
        ns={'BaseEpisode':Base,'copy':copy,'c':SimpleNamespace(typed_equal=lambda a,b:a==b),
            'ACTIVE':{'task_id':'t','initial':source,'gold':{'initial_state':canonical}}}
        path=Path(__file__).resolve().parents[1]/'verified_recovery_gpu_v2.py'
        tree=ast.parse(path.read_text())
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SourceOrderedEpisode'],type_ignores=[]),str(path),'exec'),ns)
        episode=ns['SourceOrderedEpisode']('t',['S'],canonical)
        self.assertEqual(list(episode.initial['S']['issues']),['342','330'])
        episode.initial['S']['issues']['342']['x']=9
        self.assertEqual(source['S']['issues']['342']['x'],1)

    def test_independent_prefix_real_calls_and_exact_observation(self):
        instances=[];calls=[]
        class Env:
            def __init__(self,*args):self.p=0;instances.append(self)
            async def open(self):pass
            async def save(self):return {'p':self.p}
            async def close(self):self.closed=True
        async def call(env,role,action):
            calls.append((role,action));env.p+=1
            return {'p':env.p},True,None
        c=SimpleNamespace(typed_equal=lambda a,b:a==b,call_as=call,state_at=lambda g,p:{'p':p})
        ns={'c':c,'copy':copy,'SourceOrderedEpisode':Env,'ACTIVE':{'task_id':'t','initial':{'p':0}}}
        path=Path(__file__).resolve().parents[1]/'verified_recovery_gpu_v2.py'
        tree=ast.parse(path.read_text())
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='prefix_validate_target'],type_ignores=[]),str(path),'exec'),ns)
        gold={'initial_state':{'p':0},'actions':[{'name':'A'},{'name':'B'}],
              'observations':[{'p':1},{'p':2}],'states_after':[{'p':1},{'p':2}],'final_state':{'p':2}}
        self.assertTrue(asyncio.run(ns['prefix_validate_target']('t',gold,1)))
        self.assertEqual([r for r,a in calls],['TARGET_PREFIX_REPLAY','TARGET_VALIDATION'])
        self.assertEqual(len(instances),1);self.assertTrue(instances[0].closed)
        gold['observations'][0]={'wrong':True}
        self.assertFalse(asyncio.run(ns['prefix_validate_target']('t',gold,1)))

if __name__=='__main__':unittest.main()
