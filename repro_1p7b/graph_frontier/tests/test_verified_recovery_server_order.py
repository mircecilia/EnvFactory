import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace


class ServerOrderTest(unittest.TestCase):
    def test_source_server_list_not_snapshot_keys(self):
        class Base:
            def __init__(self,task,servers,initial):self.servers=servers;self.initial=initial
        ns={'v':SimpleNamespace(SourceOrderedEpisode=Base),'copy':copy,
            'SERVERS':{'task_id':'t','servers':['TradingBot','ShopifyEcommerce']}}
        path=Path(__file__).resolve().parents[1]/'verified_recovery_gpu_v3.py'
        tree=ast.parse(path.read_text())
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ServerOrderedEpisode'],type_ignores=[]),str(path),'exec'),ns)
        initial={'ShopifyEcommerce':{},'TradingBot':{}}
        actual=ns['ServerOrderedEpisode']('t',list(initial),initial)
        self.assertEqual(actual.servers,['TradingBot','ShopifyEcommerce'])
        self.assertIs(actual.initial,initial)
        other=ns['ServerOrderedEpisode']('other',['Other'],{})
        self.assertEqual(other.servers,['Other'])

if __name__=='__main__':unittest.main()
