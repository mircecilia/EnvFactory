"""CPU-only independent EnvFactory state-counterfactual feasibility prototype.
No BFCL task, answer, trajectory, prompt, model, or inference is an input.
This does not emit training-ready data or launch the LLM QueryGen.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path


def stable(v):
    return json.dumps(v,sort_keys=True,separators=(",",":"))


def load_api(root):
    path=Path(root)/"envs/tools/GorillaFileSystem.py"
    spec=importlib.util.spec_from_file_location("independent_envfactory_fs",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.GorillaFileSystemAPI,path


def scenario(filename,location,exists):
    dirs={}
    for folder in ("sector_lime","sector_coral"):
        contents={filename+".bak":{"type":"file","content":"preserve-decoy"}}
        if folder==location:
            contents[filename]={"type":"file","content":"transfer-marker"}
            if exists:contents["review_box"]={"type":"directory","contents":{}}
        dirs[folder]={"type":"directory","contents":contents}
    return {"root":{"bench":{"type":"directory","contents":dirs}},
            "current_dir":"/","current_time":"2026-01-01T00:00:00"}


def target_predicate(state,original,filename,location):
    final=state["root"]["bench"]["contents"]; initial=original["root"]["bench"]["contents"]
    for folder in final:
        current=final[folder]["contents"]
        previous=initial[folder]["contents"]
        if current.get(filename+".bak")!=previous[filename+".bak"]:return False
        if folder!=location and current!=previous:return False
    source=final[location]["contents"]
    if filename in source:return False
    expected={"type":"file","content":"transfer-marker"}
    # save_scenario includes only name/content/type; no wall-clock mutation.
    return source.get("review_box",{}).get("contents",{}).get(filename)==expected


def execute(api_cls,initial,filename,forced_folder=None):
    api=api_cls();api.load_scenario(copy.deepcopy(initial));trace=[]
    def call(tool_name,**arguments):
        response=getattr(api,tool_name)(**arguments)
        trace.append({"tool":tool_name,"arguments":arguments,"response":response})
        return response
    found=call("find",path=".",name=filename)
    exact=[x for x in found["matches"] if x.endswith("/"+filename)]
    assert len(exact)==1
    observed_folder=exact[0].split("/")[-2]
    folder=forced_folder or observed_folder
    call("cd",folder=folder)
    contents=call("ls")["current_directory_content"]
    if "review_box" not in contents:call("mkdir",dir_name="review_box")
    call("mv",source=filename,destination="review_box")
    return api.save_scenario(),trace,observed_folder


def build(root):
    api_cls,path=load_api(root);rows=[]
    for i in range(12):
        nonce=hashlib.sha256(("independent-state-prototype-20261008-"+str(i)).encode()).hexdigest()[:10]
        filename="artifact_"+nonce+".txt"
        for exists in (False,True):
            pair=[]
            for location in ("sector_lime","sector_coral"):
                initial=scenario(filename,location,exists)
                final,trace,observed=execute(api_cls,initial,filename)
                assert observed==location and target_predicate(final,initial,filename,location)
                wrong="sector_coral" if location=="sector_lime" else "sector_lime"
                bad_state,bad_trace,_=execute(api_cls,initial,filename,wrong)
                assert not target_predicate(bad_state,initial,filename,location)
                task="cpu-counterfactual-"+nonce+"-"+str(int(exists))+"-"+location
                pair.append({"task_id":task,"family_id":nonce+"-"+str(int(exists)),
                    "query":"Locate "+filename+" and move it into review_box within its owning directory. Create review_box only if absent. Preserve all decoy files.",
                    "initial_scenario":initial,"gold_actions":trace,
                    "counterfactual_wrong_folder":wrong,"negative_trace":bad_trace,
                    "gold_verified":True,"crossed_negative_verified":False,
                    "sidecar":{"producer_tool":"find","producer_field":"matches",
                       "consumer_tool":"cd","consumer_argument":"folder",
                       "transform":"unique exact basename match -> parent basename",
                       "producer_value":trace[0]["response"]["matches"],
                       "consumer_value":location,"branch_guard":"ls says review_box is absent",
                       "final_predicate":"target file moved; decoys and unrelated sector unchanged"},
                    "source":"EnvFactory executable API; fresh random opaque filenames; no BFCL input"})
            assert pair[0]["query"]==pair[1]["query"]
            assert pair[0]["sidecar"]["consumer_value"]!=pair[1]["sidecar"]["consumer_value"]
            rows.extend(pair)
    return rows,path


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True);args=parser.parse_args()
    output=Path(args.output);assert not output.exists();output.mkdir(parents=True)
    rows,path=build(args.root)
    (output/"prototype_cases.jsonl").write_text("\n".join(stable(x) for x in rows)+"\n")
    metrics={"status":"CPU_PROTOTYPE_PASS","examples":len(rows),"counterfactual_pairs":len(rows)//2,
      "gold_verified":sum(x["gold_verified"] for x in rows),
      "cross_swap_negatives_rejected":sum(not x["crossed_negative_verified"] for x in rows),
      "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
      "domain":"EnvFactory GorillaFileSystemAPI only",
      "real_api_methods":True,"MCP_transport_tested":False,
      "external_service":False,"BFCL_inputs_used":False,"model_used":False,
      "training_ready":False,
      "scope":"programmatic adapter prototype, not existing native graph generator capability; no migration claim"}
    (output/"prototype_summary.json").write_text(json.dumps(metrics,indent=2))
    print(stable(metrics))

if __name__=="__main__":main()
