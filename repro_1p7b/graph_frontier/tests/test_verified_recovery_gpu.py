"""CPU-only adapter tests; no model/network import or verifier substitute."""
import ast
import copy
import json
import unittest
from pathlib import Path

path=Path(__file__).resolve().parents[1]/'verified_recovery_gpu.py'
tree=ast.parse(path.read_text())
ns={'copy':copy,'json':json}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='canonical_message'],type_ignores=[]),str(path),'exec'),ns)

class GPUAdapterTest(unittest.TestCase):
    def test_random_server_ids_do_not_break_paired_prefix(self):
        a={'role':'assistant','content':'','tool_calls':[{'id':'randomA','function':{'name':'A','arguments':'{"x":123}'}}]}
        b=copy.deepcopy(a);b['tool_calls'][0]['id']='randomB'
        self.assertEqual(ns['canonical_message'](a,7),ns['canonical_message'](b,7))
        self.assertEqual(a['tool_calls'][0]['id'],'randomA')
        self.assertEqual(ns['canonical_message'](a,7)['tool_calls'][0]['function']['arguments'],{'x':123})

    def test_non_object_arguments_rejected(self):
        with self.assertRaises(ValueError):
            ns['canonical_message']({'tool_calls':[{'function':{'arguments':'[1]'}}]},1)

if __name__=='__main__':unittest.main()
