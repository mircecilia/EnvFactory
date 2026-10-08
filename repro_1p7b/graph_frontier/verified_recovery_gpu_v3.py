"""Preserve BOTH original scenario object order and original MCP server order.

Separate v3 epoch: v1/v2 failures and source files remain unchanged. Strict
schema/state/response equality and paid target-prefix verification are retained.
"""
import argparse
import copy
from pathlib import Path

from repro_1p7b.graph_frontier import verified_recovery_gpu_v2 as v

RUN=v.c.feasibility.RUN/'gpu_sglang_v3'
SERVERS=None


class ServerOrderedEpisode(v.SourceOrderedEpisode):
    def __init__(self,task_id,servers,initial):
        if SERVERS is not None and task_id==SERVERS['task_id']:
            servers=copy.deepcopy(SERVERS['servers'])
        super().__init__(task_id,servers,initial)


def install():
    v.RUN=RUN
    v.install()
    v.SourceOrderedEpisode=ServerOrderedEpisode
    v.c.CountedEpisode=ServerOrderedEpisode
    original=v.c.collect
    async def collect(task,entry,row,gold,schemas,policy,config,arm):
        global SERVERS
        assert SERVERS is None,'overlapping task in worker'
        SERVERS={'task_id':task,'servers':v.decode(row['extra_info']['mcp_factory_kwargs']['mcp_servers'])}
        try:
            result=await original(task,entry,row,gold,schemas,policy,config,arm)
            result['mcp_server_order_source']='authoritative original mcp_servers list, never sorted snapshot keys'
            return result
        finally:SERVERS=None
    v.c.collect=collect


def protocol():
    f=v.c.feasibility
    original=f.immutable
    def write(path,value):
        if path==RUN/'collection_protocol.json':
            value=copy.deepcopy(value)
            value['schema']='verified_recovery_gpu_sglang_paired_v3_source_and_server_order'
            value['code_sha256'][Path(__file__).name]=v.g.sha256(Path(__file__))
            value['mcp_server_order_source']='original authoritative mcp_servers list; exact schemas/order checked, never loosened'
            value['previous_epoch']='gpu_sglang_v2 preserved, incomplete due canonical snapshot server key order mismatch; no episodes migrated'
            value['smoke_tasks']=12
            value['mandatory_smoke_regressions']=['official-rl-1038-86717bd5193b','official-rl-0656-b3222619f7d8']
        original(path,value)
    f.immutable=write
    try:return v.protocol()
    finally:f.immutable=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['protocol','worker','metrics'])
    p.add_argument('--shard',type=int);p.add_argument('--limit',type=int);p.add_argument('--expected',type=int)
    a=p.parse_args();install()
    if a.command=='protocol':print(v.g.stable(protocol()))
    elif a.command=='worker':v.c.worker(a.shard,a.limit,4)
    else:v.g.m.summarize(a.expected)
