"""Aggregate three frozen pilot runs and exact paired McNemar tests."""
from __future__ import annotations
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
LABELS=("base","original_sft","parameter_aware")
DISPLAY={"base":"Base","original_sft":"Original SFT","parameter_aware":"Parameter-Aware"}
def rate(s,n):return {"successes":s,"attempts":n,"raw_rate":s/n if n else None,"smoothed_rate":(s+1)/(n+2)}
def mcnemar(a,b):
 ids=sorted(set(a)&set(b));w=sum((not a[i]) and b[i] for i in ids);l=sum(a[i] and (not b[i]) for i in ids);n=w+l
 p=1.0 if not n else min(1.0,2*sum(math.comb(n,i) for i in range(min(w,l)+1))/(2**n))
 return {"paired_support":len(ids),"right_wins":w,"right_losses":l,"discordant":n,"exact_two_sided_p":p}
def edge_status(e,events,c):
 prod=[x for x in events if x["tool_name"]==e["producer_tool"]["tool_name"]];cons=[x for x in events if x["tool_name"]==e["consumer_tool"]["tool_name"]]
 if not prod:return "producer_not_executed"
 if not any(x.get("execution_success") is True for x in prod):return "producer_failed"
 if not cons:return "consumer_not_executed"
 if c and c.get("success") is True:return "success"
 if c and c.get("status")=="failed_wrong_value":return "wrong_propagated_value"
 if c and c.get("target_value_available") is False:return "missing_required_internal_value"
 if c and c.get("success") is False:return "dependency_failure"
 return "unresolved_gold_edge"
def load_results(root,label):return {p.name[:-12]:json.loads(p.read_text()) for p in (root/label/"tasks").glob("*.result.json")}
def aggregate(rows,manifest):
 valid=[r for r in rows.values() if r["valid_capability_probe"]];errors=Counter(r["system_status"] for r in rows.values() if not r["valid_capability_probe"]);ec=Counter();depth=defaultdict(lambda:[0,0]);internal=Counter();roots=Counter();first=Counter();gold_n=gold_ok=unexpected=redundant=0;maxa=[];maxs=[];per={}
 task=sum(r["task_success"] for r in valid);state=sum(r["final_state_success"] is True for r in valid);state_n=sum(isinstance(r["final_state_success"],bool) for r in valid);path=sum(r["path_adherence"] is True for r in valid);path_n=sum(isinstance(r["path_adherence"],bool) for r in valid)
 for tid,r in rows.items():
  if not r["valid_capability_probe"]:continue
  m=manifest[tid];p=r["profile"];events=p.get("actual_tool_nodes") or [];gold_n+=len(m["gold_tools"]);gold_ok+=sum(any(x["tool_name"]==t and x.get("execution_success") is True for x in events) for t in m["gold_tools"]);unexpected+=r["unexpected_calls"];redundant+=r["redundant_calls"];checks={x["edge_id"]:x for x in p.get("dependency_edge_checks") or []};bad=[];attempt=[]
  for e in m["dependency_edges"]:
   c=checks.get(e["edge_id"]);s=edge_status(e,events,c);ec[s]+=1;d=e["dependency_depth"];depth[str(d)][1]+=1;depth[str(d)][0]+=s=="success"
   if any(x["tool_name"]==e["consumer_tool"]["tool_name"] for x in events):attempt.append(d);internal["attempts"]+=1
   if s=="success":internal["correct"]+=1
   elif s=="wrong_propagated_value":internal["wrong"]+=1
   elif s in {"missing_required_internal_value","producer_not_executed","consumer_not_executed"}:internal["missing"]+=1
   if s!="success":bad.append(d)
  if attempt:maxa.append(max(attempt))
  solved=p.get("max_dependency_depth_reached")
  if isinstance(solved,int):maxs.append(solved)
  if bad:first[str(min(bad))]+=1
  rr=p.get("root_cause_failures") or []
  if rr:roots[rr[0].get("type","unknown")]+=1
  elif not r["task_success"]:
   statuses=[edge_status(e,events,checks.get(e["edge_id"])) for e in m["dependency_edges"]];roots[next((x for x in statuses if x!="success"),"final_state_mismatch")]+=1
  per[tid]={"task_success":r["task_success"],"edge_complete":r["edge_complete"],"internal_param_complete":r["internal_param_complete"]}
 return {"probe_count":len(rows),"valid_capability_probes":len(valid),"system_errors":dict(errors),"overall":{"completed":len(rows),"task_success":rate(task,len(valid)),"final_state_success":rate(state,state_n),"path_adherence":rate(path,path_n)},"tool_execution":{"gold_node_attempts":gold_n,"gold_node_executed":gold_ok,"node_coverage":gold_ok/gold_n if gold_n else None,"unexpected_calls":unexpected,"redundant_calls":redundant},"dependency_edges":{"distribution":dict(ec),"success":rate(ec["success"],sum(ec.values()))},"internal_parameter_propagation":{"correct":internal["correct"],"wrong":internal["wrong"],"missing":internal["attempts"]-internal["correct"]-internal["wrong"],"attempts":internal["attempts"],"raw_accuracy":internal["correct"]/internal["attempts"] if internal["attempts"] else None,"smoothed_accuracy":(internal["correct"]+1)/(internal["attempts"]+2)},"depth":{k:rate(v[0],v[1]) for k,v in sorted(depth.items(),key=lambda x:int(x[0]))},"max_attempted_dependency_depth":max(maxa) if maxa else None,"max_solved_dependency_depth":max(maxs) if maxs else None,"first_failed_dependency_depth":dict(first),"root_failure_type":dict(roots),"per_task":per}
def fmt(m):return "n/a" if m["raw_rate"] is None else f"{100*m['raw_rate']:.1f}% ({m['successes']}/{m['attempts']}; smooth {100*m['smoothed_rate']:.1f}%)"
def analyze(root,mp):
 manifest={x["task_id"]:x for x in [json.loads(z) for z in mp.read_text().splitlines() if z.strip()]};rows={x:load_results(root,x) for x in LABELS};agg={x:aggregate(rows[x],manifest) for x in LABELS};[agg[x].update(runtime_seconds=json.loads((root/x/"run_summary.json").read_text())["runtime_seconds"]) for x in LABELS];paired={}
 for l,r in (("base","original_sft"),("original_sft","parameter_aware"),("base","parameter_aware")):paired[f"{l}_vs_{r}"]={m:mcnemar({k:v[m] for k,v in agg[l]["per_task"].items()},{k:v[m] for k,v in agg[r]["per_task"].items()}) for m in ("task_success","edge_complete","internal_param_complete")}
 out={"schema_version":"graph_frontier_capability_comparison_v1","manifest":str(mp),"models":agg,"paired_comparison":paired,"limitations":["100-task pilot; not a final benchmark.","Edges within a task are clustered; no edge-level significance claim.","Exact McNemar uses task-level binary outcomes only."]};(root/"capability_comparison.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
 metrics=[("Task success",lambda a:a["overall"]["task_success"]),("Final-state success",lambda a:a["overall"]["final_state_success"]),("Edge success",lambda a:a["dependency_edges"]["success"]),("Internal param propagation",lambda a:{"successes":a["internal_parameter_propagation"]["correct"],"attempts":a["internal_parameter_propagation"]["attempts"],"raw_rate":a["internal_parameter_propagation"]["raw_accuracy"],"smoothed_rate":a["internal_parameter_propagation"]["smoothed_accuracy"]}),("Depth-1 success",lambda a:a["depth"].get("1",rate(0,0))),("Depth-2 success",lambda a:a["depth"].get("2",rate(0,0))),("Depth-3+ success",lambda a:a["depth"].get("3",rate(0,0)))]
 lines=["# Static Graph-Frontier capability comparison","","| Metric | Base | Original SFT | Parameter-Aware | SFT-Base | PA-SFT |","|---|---:|---:|---:|---:|---:|"]
 for name,fn in metrics:
  ms=[fn(agg[x]) for x in LABELS];vals=[m["raw_rate"] or 0 for m in ms];lines.append(f"| {name} | {fmt(ms[0])} | {fmt(ms[1])} | {fmt(ms[2])} | {vals[1]-vals[0]:+.3f} | {vals[2]-vals[1]:+.3f} |")
 for name,key in (("Wrong propagated value","wrong_propagated_value"),("Redundant calls",None)):
  vals=[agg[x]["dependency_edges"]["distribution"].get(key,0) if key else agg[x]["tool_execution"]["redundant_calls"] for x in LABELS];lines.append(f"| {name} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[1]-vals[0]:+d} | {vals[2]-vals[1]:+d} |")
 lines+=["","## Paired exact McNemar","",json.dumps(paired,indent=2,sort_keys=True),"","## Failure distributions",""]
 for x in LABELS:lines += [f"### {DISPLAY[x]}","",f"- Edge status: {agg[x]['dependency_edges']['distribution']}",f"- Root failure: {agg[x]['root_failure_type']}",f"- First failed depth: {agg[x]['first_failed_dependency_depth']}",f"- System errors: {agg[x]['system_errors']}",""]
 lines+=["## Limitations","","This is a 100-task signal-validation pilot. Edges within one task are clustered; no edge-level significance claim is made."];(root/"capability_comparison.md").write_text("\n".join(lines)+"\n");return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=ROOT/"repro_1p7b/results/graph_frontier/pilot_100");p.add_argument("--manifest",type=Path,default=ROOT/"repro_1p7b/results/graph_frontier/pilot_100/frozen/pilot_100_seed_20260914.jsonl");a=p.parse_args();o=analyze(a.root,a.manifest);print(json.dumps({k:{"valid":v["valid_capability_probes"],"task_success":v["overall"]["task_success"]} for k,v in o["models"].items()},indent=2))
if __name__=="__main__":main()
