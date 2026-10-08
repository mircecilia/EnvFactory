"""Secondary official no-call constraints and duplicate-record consistency.
CPU read-only. A constraint violation is not a unique causal diagnosis.
Failure cohort only: successful score rows do not retain gold constraints.
"""
import argparse
import collections
import json
from pathlib import Path
from repro_1p7b.graph_frontier.bfcl_failure_driven_audit import call_ast, sha, stable

def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)

def constraints(record):
    gold=record.get("possible_answer")
    decoded=record.get("model_result_decoded")
    if not isinstance(gold,list) or not isinstance(decoded,list):
        return {"unknown":True,"eligible_turns":[],"violated_turns":[]}
    eligible=[]; violated=[]; unobserved=[]
    for turn, actions in enumerate(gold):
        if actions != []:
            continue
        eligible.append(turn)
        if turn>=len(decoded):
            unobserved.append(turn)
            continue
        if any(call_ast(x) is not None for x in strings(decoded[turn])):
            violated.append(turn)
    return {"unknown":False,"eligible_turns":eligible,
            "violated_turns":violated,"unobserved_turns":unobserved}

def run(summary_file, output):
    output=Path(output)
    assert not output.exists(),"new namespace required"
    summary=json.loads(Path(summary_file).read_text())
    result={}
    for model, inventory in summary["inventory"].items():
        categories={}
        for source in inventory["score_inputs"]:
            p=Path(source["path"]); assert sha(p)==source["sha256"]
            rows={}; fingerprints=collections.defaultdict(set); gold_decoded=collections.defaultdict(set); error_types=collections.defaultdict(set); duplicate=0
            with p.open() as stream:
                next(stream)
                for line in stream:
                    r=json.loads(line); task=r["id"]
                    duplicate+=task in rows
                    fingerprints[task].add(stable({k:r.get(k) for k in
                        ["error","possible_answer","model_result_decoded"]}))
                    gold_decoded[task].add(stable({k:r.get(k) for k in ["possible_answer","model_result_decoded"]}))
                    error_types[task].add(r.get("error",{}).get("error_type","UNKNOWN"))
                    rows.setdefault(task,r)
            classified={task:constraints(r) for task,r in rows.items()}
            categories[p.name]={
                "failed_tasks":len(rows),"duplicate_rows":duplicate,
                "conflicting_duplicate_task_ids":[t for t,v in fingerprints.items() if len(v)>1],
                "gold_or_decoded_conflicting_task_ids":[t for t,v in gold_decoded.items() if len(v)>1],
                "duplicate_error_types":{t:sorted(error_types[t]) for t,v in fingerprints.items() if len(v)>1},
                "eligible_failed_tasks":sum(bool(x["eligible_turns"]) for x in classified.values()),
                "constraint_violating_failed_tasks":sum(bool(x["violated_turns"]) for x in classified.values()),
                "violated_turns":sum(len(x["violated_turns"]) for x in classified.values()),
                "unobserved_eligible_turns":sum(len(x.get("unobserved_turns",[])) for x in classified.values()),
                "missing_gold_or_decoded":sum(x["unknown"] for x in classified.values()),
                "task_constraints":classified}
            assert sha(p)==source["sha256"],"official score changed"
        result[model]=categories
    output.mkdir(parents=True)
    payload={"scope":"failure-cohort-only; reference no-call constraint, not proven missing-information cause",
             "primary_labels_unchanged":True,"source_summary":str(summary_file),
             "models":result,"no_gpu":True}
    (output/"no_call_duplicate_audit.json").write_text(json.dumps(payload,indent=2))
    print(stable({m:{c:{k:v for k,v in data.items() if k!="task_constraints"}
                       for c,data in categories.items()} for m,categories in result.items()}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--summary",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();run(args.summary,args.output)
