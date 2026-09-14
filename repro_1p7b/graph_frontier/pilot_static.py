"""Frozen autonomous Graph-Frontier pilot. No training code."""
from __future__ import annotations
import argparse,asyncio,copy,hashlib,itertools,json,os,time
from collections import Counter
from pathlib import Path
from typing import Any
from agents import ModelSettings
ROOT=Path(__file__).resolve().parents[2]
for k,v in {"CHAT_API_KEY":"pilot","CHAT_URL":"http://127.0.0.1:1/v1","CHAT_MODEL":"unused","EMBEDDING_API_KEY":"pilot","EMBEDDING_URL":"http://127.0.0.1","EMBEDDING_MODEL":"unused"}.items():os.environ.setdefault(k,v)
from repro_1p7b.graph_frontier.export_pipeline import profile_sidecar_and_trace
from repro_1p7b.graph_frontier.fastmcp_adapter import fastmcp_execution_success,fastmcp_result_fields
from repro_1p7b.graph_frontier.gold_sidecar import build_gold_sidecar,write_gold_sidecar
from repro_1p7b.graph_frontier.rollout_trace import TypedRolloutRecorder
from repro_1p7b.graph_frontier.state_verifier import compare_final_states
from repro_1p7b.graph_frontier.traceable_sampler import TRACE_ATTRIBUTE
from src.gen.query_gen import QueryGenConfig,QueryGenContext
from src.gen.query_gen.query_gen_non_conv import QueryGenNonConv
from src.graph.tool_chain import ToolQueryChain,ToolQueryNode
from src.graph.tool_graph import EdgeType,NodeType,ToolGraph
from src.graph.tool_node import Tool
from src.manager.mcp_client_manager import MCPManager
SEED=20260914
SERVERS=("TradingBot","Weather","GoogleTasks","UUPaoTui","Calendar")
PATHS={"base":"repro_1p7b/models/Qwen3-1.7B","original_sft":"repro_1p7b/checkpoints/baseline_sft_8k_1p7b","parameter_aware":"repro_1p7b/checkpoints/parameter_aware_sft_8k_1p7b"}
PROTOCOL={"temperature":0.0,"top_p":1.0,"presence_penalty":0.0,"max_new_tokens":1024,"max_tool_calls":8,"max_tool_rounds":4,"max_turns":1,"probe_timeout_seconds":120,"retry_policy":"no outer retry","stop_condition":"first no-tool response or four rounds","environment_reset":"fresh client id and load_scenario","thinking":False}
def stable(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def digest(v):return hashlib.sha256(stable(v).encode()).hexdigest()
def scenarios(s):return [x["scenario_data"] for x in json.loads((ROOT/f"envs/intermediate/{s}_checkpoint.json").read_text())["scenarios"]]
def calendar(v):return {"calendars":{"primary":{"calendar_id":"primary","calendar_name":"Pilot Calendar","description":"Pilot","timezone":"Asia/Singapore","is_primary":True}},"events":{},"default_calendar_id":"primary","event_id_counter":100+v}
def specs():
 o=[]
 def add(s,n,ts,ds,k):o.append({"server":s,"template":n,"tools":ts,"deps":ds,"kind":k})
 add("TradingBot","symbol_info",["get_stock_symbol","get_stock_info"],[(0,"symbol",1,"symbol")],"lookup_lookup")
 add("TradingBot","quote_order",["get_stock_symbol","get_stock_info","place_order"],[(0,"symbol",1,"symbol"),(0,"symbol",2,"symbol"),(1,"price",2,"price")],"lookup_lookup_action")
 add("TradingBot","quote_order_details",["get_stock_symbol","get_stock_info","place_order","get_order_details"],[(0,"symbol",1,"symbol"),(0,"symbol",2,"symbol"),(1,"price",2,"price"),(2,"order_id",3,"order_id")],"lookup_lookup_action_lookup")
 for n,x in [("weather","get_current_weather"),("forecast","get_forecast"),("alerts","get_alerts"),("save","save_location")]:add("Weather","city_"+n,["search_location",x],[(0,"location.latitude",1,"latitude"),(0,"location.longitude",1,"longitude")],"lookup_action" if n=="save" else "lookup_lookup")
 add("GoogleTasks","list_get",["create_task_list","get_task_list"],[(0,"tasklist_id",1,"tasklist_id")],"create_lookup")
 add("GoogleTasks","list_task",["create_task_list","create_task"],[(0,"tasklist_id",1,"tasklist_id")],"create_create")
 for n,x,k in [("task_get","get_task","create_create_lookup"),("task_update","update_task","create_create_update"),("task_delete","delete_task","create_create_delete")]:add("GoogleTasks",n,["create_task_list","create_task",x],[(0,"tasklist_id",1,"tasklist_id"),(0,"tasklist_id",2,"tasklist_id"),(1,"task_id",2,"task_id")],k)
 add("UUPaoTui","estimate_create",["estimate_price","create_order"],[(0,"priceToken",1,"priceToken")],"lookup_action")
 add("UUPaoTui","estimate_query",["estimate_price","create_order","query_order"],[(0,"priceToken",1,"priceToken"),(1,"orderCode",2,"orderCode")],"lookup_action_lookup")
 add("UUPaoTui","estimate_cancel",["estimate_price","create_order","query_order","cancel_order"],[(0,"priceToken",1,"priceToken"),(1,"orderCode",2,"orderCode"),(2,"orderCode",3,"orderCode")],"lookup_action_lookup_action")
 add("Calendar","create_update",["create_event","update_event"],[(0,"event_id",1,"event_id")],"create_update")
 add("Calendar","create_update_delete",["create_event","update_event","delete_event"],[(0,"event_id",1,"event_id"),(1,"event_id",2,"event_id")],"create_update_delete")
 assert len(o)==17
 return o
def get(v,p):
 for x in p.split("."):v=v[int(x)] if isinstance(v,list) else v[x]
 return v
def refs(v,outs):
 if isinstance(v,str) and v.startswith("@"):i,p=v[1:].split(".",1);return get(outs[int(i)],p)
 if isinstance(v,dict):return {k:refs(x,outs) for k,x in v.items()}
 if isinstance(v,list):return [refs(x,outs) for x in v]
 return v
def trading(s):
 s=copy.deepcopy(s)
 s["users"]={"pilot":{"username":"pilot","password":"pilot"}};s["accounts"]={999:{"account_id":999,"balance":100000.0,"currency":"USD","binding_card":None}}
 s["current_user"]="pilot";s["session_token"]="pilot";s["market_open"]=True;s["current_time"]="2026-09-14 12:00:00";return s
def make(t,v):
 s=t["server"]
 if s=="Calendar":state=calendar(v)
 elif s=="GoogleTasks":state={"tasklists":{},"tasks":{},"next_tasklist_id":100+v*10,"next_task_id":1000+v*10,"default_tasklist_id":"@default","current_time":"2026-09-14T12:00:00"}
 elif s=="UUPaoTui":state={"orders":{},"priceTokens":{},"baseRatesMap":{"instant":8.0,"scheduled":6.0,"per_km":2.5,"per_min":0.5},"cityMultipliersMap":{"110000":1.2},"orderStates":["pending","delivering","completed","cancelled","exception"],"current_time":"2026-09-14T12:00:00"}
 else:state=copy.deepcopy(scenarios(s)[v%4])
 n=t["template"];tag=f"{v:02d}-{n}"
 if s=="TradingBot":
  state=trading(state);company=sorted(state["company_symbols"])[v%len(state["company_symbols"])];calls=[{"company_name":company},{"symbol":"@0.symbol"}];q=f"Look up {company}'s ticker and current quote"
  if n!="symbol_info":calls+=[{"order_type":"Buy","symbol":"@0.symbol","price":"@1.price","quantity":v+1}];q+=f", then buy {v+1} share(s) at that quoted price"
  if n=="quote_order_details":calls+=[{"order_id":"@2.order_id"}];q+=", and retrieve the created order details"
  q+="."
 elif s=="Weather":
  city=sorted(state["city_database"])[v%len(state["city_database"])];calls=[{"city":city}]
  if n=="city_weather":calls+=[{"latitude":"@0.location.latitude","longitude":"@0.location.longitude"}];q=f"Find {city}'s coordinates and report current weather there."
  elif n=="city_forecast":calls+=[{"latitude":"@0.location.latitude","longitude":"@0.location.longitude","days":2+v%3}];q=f"Find {city}'s coordinates and give its {2+v%3}-day forecast."
  elif n=="city_alerts":calls+=[{"latitude":"@0.location.latitude","longitude":"@0.location.longitude","active_only":True}];q=f"Find {city}'s coordinates and check active weather alerts there."
  else:alias=f"pilot-{tag}";calls+=[{"alias":alias,"latitude":"@0.location.latitude","longitude":"@0.location.longitude","name":f"{city} pilot"}];q=f"Find {city}'s coordinates and save them as alias {alias}, named '{city} pilot'."
 elif s=="GoogleTasks":
  lt=f"Pilot List {tag}";tt=f"Pilot Task {tag}";calls=[{"tasklist_title":lt}]
  if n=="list_get":calls+=[{"tasklist_id":"@0.tasklist_id"}];q=f"Create task list '{lt}', then retrieve that new list."
  else:
   calls+=[{"tasklist_id":"@0.tasklist_id","title":tt,"notes":"Graph Frontier pilot"}]
   if n=="list_task":q=f"Create task list '{lt}', then add task '{tt}'."
   elif n=="task_get":calls+=[{"task_id":"@1.task_id","tasklist_id":"@0.tasklist_id"}];q=f"Create list '{lt}', add task '{tt}', then retrieve the new task."
   elif n=="task_update":calls+=[{"tasklist_id":"@0.tasklist_id","task_id":"@1.task_id","title":tt+" updated"}];q=f"Create list '{lt}', add task '{tt}', then rename it to '{tt} updated'."
   else:calls+=[{"task_id":"@1.task_id","tasklist_id":"@0.tasklist_id"}];q=f"Create list '{lt}', add task '{tt}', then delete that new task."
 elif s=="UUPaoTui":
  a=f"{10+v} Pilot Road";b=f"{20+v} Frontier Avenue";sp=f"1380000{v:04d}";rp=f"1390000{v:04d}";calls=[{"fromAddress":a,"toAddress":b,"adCode":"110000","sendType":"instant"},{"priceToken":"@0.priceToken","receiverPhone":rp,"senderPhone":sp}]
  if n=="estimate_create":q=f"Estimate an instant delivery from {a} to {b} in ad code 110000, then create it with sender {sp} and receiver {rp}."
  elif n=="estimate_query":calls+=[{"orderCode":"@1.orderCode"}];q=f"Estimate and create an instant delivery from {a} to {b}, then query the new order. Sender {sp}, receiver {rp}, ad code 110000."
  else:calls+=[{"orderCode":"@1.orderCode"},{"orderCode":"@2.orderCode","reason":"pilot cancellation"}];q=f"Estimate and create an instant delivery from {a} to {b}, query it, then cancel it for 'pilot cancellation'. Sender {sp}, receiver {rp}, ad code 110000."
 else:
  title=f"Pilot Event {tag}";calls=[{"summary":title,"start_time":"2026-10-20T10:00:00Z","end_time":"2026-10-20T11:00:00Z","calendar_id":"primary"},{"event_id":"@0.event_id","summary":title+" Updated"}]
  if n=="create_update":q=f"Create '{title}' on primary from 2026-10-20 10:00 to 11:00 UTC, then rename it to '{title} Updated'."
  else:calls+=[{"event_id":"@1.event_id"}];q=f"Create '{title}' on primary from 2026-10-20 10:00 to 11:00 UTC, rename it to '{title} Updated', then delete it."
 return {**t,"variant":v,"candidate_id":f"{s.lower()}-{n}-{v:02d}","initial_state":state,"query":q,"call_args":calls}
def metadata(s):return json.loads((ROOT/f"envs/metadata/{s}_metadata.json").read_text())
def graph(c,expected):
 by={x["name"]:x for x in metadata(c["server"])["tools"]};g=ToolGraph();g.server_to_tools={c["server"]:[]};tools=[]
 for short in c["tools"]:
  raw=copy.deepcopy(by[short]);raw.update(server=c["server"],name=f"{c['server']}-{short}");t=Tool(raw);tools.append(t);g.server_to_tools[c["server"]].append(t);g.graph.add_node(t,node_type=NodeType.Tool)
  for p in t.input_schema["parameters"]:g.graph.add_node(p,node_type=NodeType.Parameter);g.graph.add_edge(p,t,edge_type=EdgeType.Tool_Input,required=(p.name in t.input_schema["required"] or p.name.split(".",1)[0] in t.input_schema["required"]))
  for p in t.output_schema["parameters"]:g.graph.add_node(p,node_type=NodeType.Parameter);g.graph.add_edge(t,p,edge_type=EdgeType.Tool_Output)
 targets={(d,p) for _,_,d,p in c["deps"]}
 for i,t in enumerate(tools):
  for p in t.input_schema["parameters"]:p.set_user_provided((i,p.name) not in targets)
 trace=[]
 for si,sp,di,dp in c["deps"]:
  o=next(p for p in tools[si].output_schema["parameters"] if p.name==sp);inp=next(p for p in tools[di].input_schema["parameters"] if p.name==dp);g.graph.add_edge(o,inp,edge_type=EdgeType.Parameter_Relate);g.graph.add_edge(tools[si],tools[di],edge_type=EdgeType.Tool_Depend)
  trace.append({"consumer_tool":tools[di].name,"consumer_input_parameter":dp,"consumer_input_data_type":inp.data_type,"selected_producer_tool":tools[si].name,"selected_producer_output_parameter":sp,"selected_producer_output_data_type":o.data_type,"required":True,"user_provided":False,"internal_parameter":True,"alternatives":[{"producer_tool":tools[si].name,"producer_output_parameters":[{"parameter_name":sp,"data_type":o.data_type}]}],"alternative_semantics":"or","provenance":"pilot_reference_execution_exact_typed_binding"})
 ch=ToolQueryChain(tools,seed=SEED);node=ToolQueryNode(raw_tool_call=tools,initial_scenario={c["server"]:copy.deepcopy(c["initial_state"])},query=c["query"],user_intent=c["query"],steps=[{"role":"user","content":c["query"]}]);ch.tool_chain=[node];ch.scenario=c["query"];setattr(ch,TRACE_ATTRIBUTE,trace)
 return g,ch,build_gold_sidecar(g,ch,0,task_id=c["candidate_id"],query_id=c["candidate_id"],expected_final_scenario={c["server"]:expected})
def parse(raw):
 try:return json.loads(raw)
 except Exception:return raw
def reference(c):
 s=c["server"];stem=hashlib.sha256(c["candidate_id"].encode()).hexdigest()[:15];cid=f"{s}-r"+stem;MCPManager.load_scenario(cid,c["initial_state"]);reset=parse(MCPManager.call_tool(cid,"save_scenario",{}));MCPManager.close_client(cid);c["initial_state"]=reset;cid=f"{s}-v"+stem;MCPManager.load_scenario(cid,reset);reset2=parse(MCPManager.call_tool(cid,"save_scenario",{}))
 if compare_final_states(reset2,reset) is not True:MCPManager.close_client(cid);raise RuntimeError("reset_roundtrip_mismatch")
 outs=[];records=[]
 try:
  for i,(tool,spec) in enumerate(zip(c["tools"],c["call_args"])):args=refs(spec,outs);result=parse(MCPManager.call_tool(cid,tool,args));outs.append(result);records.append({"step_index":i,"tool_name":f"{s}-{tool}","arguments":args,"result":result})
  final=parse(MCPManager.call_tool(cid,"save_scenario",{}))
 finally:MCPManager.close_client(cid)
 for si,sp,di,dp in c["deps"]:
  a,b=get(outs[si],sp),get(records[di]["arguments"],dp)
  if type(a) is not type(b) or a!=b:raise RuntimeError(f"typed_mismatch:{si}.{sp}->{di}.{dp}")
 return records,final
def register():
 for s in SERVERS:asyncio.run_coroutine_threadsafe(MCPManager.register_mcp_server_async(s,f"envs/tools/{s}.py",False),MCPManager._loop).result(timeout=60)
def freeze(root,compact):
 frozen=root/"frozen"
 for d in ("gold","states","reference"):(frozen/d).mkdir(parents=True,exist_ok=True)
 accepted=[];audit=[];rejects=Counter()
 for v in range(7):
  for t in specs():
   c=make(t,v);a={"candidate_id":c["candidate_id"],"eligible":False,"rejection_reason":"unknown"}
   try:
    ref,final=reference(c);g,ch,side=graph(c,final)
    if side["probe_eligibility"]["eligible"] is not True:raise RuntimeError("sidecar:"+",".join(side["probe_eligibility"]["rejection_reasons"]))
    ih,fh=digest(c["initial_state"]),digest(final);ip=frozen/"states"/f"{ih}.json";fp=frozen/"states"/f"{fh}.json"
    if not ip.exists():ip.write_text(json.dumps(c["initial_state"],ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    if not fp.exists():fp.write_text(json.dumps(final,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    gp=frozen/"gold"/f"{c['candidate_id']}.gold.json";rp=frozen/"reference"/f"{c['candidate_id']}.json";write_gold_sidecar(side,gp);rp.write_text(json.dumps(ref,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    rec={"schema_version":"graph_frontier_pilot_manifest_v1","task_id":c["candidate_id"],"seed":SEED,"eligible":True,"environment":c["server"],"environment_source":f"envs/tools/{c['server']}.py","environment_source_sha256":hashlib.sha256((ROOT/f"envs/tools/{c['server']}.py").read_bytes()).hexdigest(),"scenario_source":"deterministic executable EnvFactory state","initial_state_sha256":ih,"expected_final_state_sha256":fh,"initial_state_path":str(ip.relative_to(ROOT)),"expected_final_state_path":str(fp.relative_to(ROOT)),"gold_sidecar_path":str(gp.relative_to(ROOT)),"reference_trace_path":str(rp.relative_to(ROOT)),"query":c["query"],"gold_tools":[f"{c['server']}-{x}" for x in c["tools"]],"dependency_edges":side["dependency_edges"],"dependency_depth":side["dependency_depth"],"complexity_bucket":"3+" if side["dependency_depth"]>=3 else str(side["dependency_depth"]),"task_kind":c["kind"],"template":c["template"],"variant":v,"reference_metadata":{"execution":"real FastMCP typed reference","call_count":len(ref),"reset_roundtrip":True}}
    accepted.append(rec);a.update(eligible=True,rejection_reason=None)
   except Exception as e:reason=f"{type(e).__name__}:{e}";a["rejection_reason"]=reason;rejects[reason]+=1
   audit.append(a)
   if len(accepted)>=100:break
  if len(accepted)>=100:break
 if len(accepted)<100:raise RuntimeError(f"only {len(accepted)} eligible; {dict(rejects)}")
 mp=frozen/f"pilot_100_seed_{SEED}.jsonl";mp.write_text("".join(stable(x)+"\n" for x in accepted));compact.parent.mkdir(parents=True,exist_ok=True);compact.write_text("".join(stable({k:v for k,v in x.items() if k!="dependency_edges"}|{"dependency_edge_count":len(x["dependency_edges"])})+"\n" for x in accepted))
 (frozen/"candidate_audit.json").write_text(json.dumps({"seed":SEED,"candidates_generated":len(audit),"accepted":100,"rejected":len(audit)-100,"rejection_reason_counts":dict(rejects),"candidates":audit},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
 summary={"seed":SEED,"probe_count":100,"candidates_generated":len(audit),"accepted":100,"rejected":len(audit)-100,"rejection_reason_counts":dict(rejects),"complexity_distribution":dict(Counter(x["complexity_bucket"] for x in accepted)),"dependency_depth_distribution":dict(Counter(str(x["dependency_depth"]) for x in accepted)),"environment_distribution":dict(Counter(x["environment"] for x in accepted)),"task_kind_distribution":dict(Counter(x["task_kind"] for x in accepted)),"template_count":len(set(x["template"] for x in accepted)),"manifest_path":str(mp.relative_to(ROOT)),"manifest_sha256":hashlib.sha256(mp.read_bytes()).hexdigest()};(frozen/"manifest_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n");return summary
def load(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def hook(rec,max_calls):
 original=MCPManager._call_tool_async;counter=itertools.count();count={"n":0}
 async def traced(name,args,client,cid):
  short=name.split("-",1)[-1]
  if short in {"load_scenario","save_scenario"}:return await original(name,args,client,cid)
  parsed=json.loads(args) if isinstance(args,str) else args;server=cid.split("-",1)[0]
  async def execute(_n,_a):
   if count["n"]>=max_calls:raise RuntimeError("max_tool_calls_exceeded")
   count["n"]+=1;return await client.call_tool(short,parsed)
  async def record():return await rec.record_async_call(next(counter),name,parsed,execute,success_from_response=fastmcp_execution_success,fields_from_response=fastmcp_result_fields,environment_id=server,server_id=server)
  if server in MCPManager.stateless_clients:
   async with MCPManager._stateless_lock:r=await record()
  else:
   async with client:r=await record()
  return ",".join(x.text for x in r.content if hasattr(x,"text"))
 MCPManager._call_tool_async=traced;return original
async def one(r,label,out):
 td=out/"tasks";td.mkdir(parents=True,exist_ok=True);rp=td/f"{r['task_id']}.result.json"
 if rp.exists():return json.loads(rp.read_text())
 initial=json.loads((ROOT/r["initial_state_path"]).read_text());expected=json.loads((ROOT/r["expected_final_state_path"]).read_text());side=json.loads((ROOT/r["gold_sidecar_path"]).read_text());tools=[x.split("-",1)[1] for x in r["gold_tools"]];deps=[]
 for e in side["dependency_edges"]:deps.append((tools.index(e["producer_tool"]["tool_name"].split("-",1)[1]),e["producer_output_parameter"]["parameter_name"],tools.index(e["consumer_tool"]["tool_name"].split("-",1)[1]),e["consumer_input_parameter"]["parameter_name"]))
 c={"server":r["environment"],"tools":tools,"deps":deps,"initial_state":initial,"query":r["query"],"candidate_id":r["task_id"]};g,ch,_=graph(c,expected);node=ch.tool_chain[0];rec=TypedRolloutRecorder(r["task_id"],[r["environment"]],{r["environment"]:initial},auto_timestamp=True)
 cfg=QueryGenConfig(model_name="sglang",temperature=0.0,top_p=1.0,presence_penalty=0.0,pass_k=1,max_iterations=1,max_retry=1,max_solve_iterations=PROTOCOL["max_tool_rounds"],max_refine_iterations=0,enable_split_turns=False,enable_query_refinement=False,enable_user_interaction=False,enable_user_tool_use=False,enable_user_verification=False,enable_filteration=False,enable_log_thinking_content=False,save_folder=str(out/"querygen"),log_folder=str(out/"querygen_logs"));gen=QueryGenNonConv(g,cfg);gen.query_solver.model_settings=ModelSettings(temperature=0.0,top_p=1.0,presence_penalty=0.0,max_tokens=PROTOCOL["max_new_tokens"],extra_body={"chat_template_kwargs":{"enable_thinking":False}})
 ctx=QueryGenContext(config=cfg,tool_graph=g,tool_chain=ch,idx=0,conversation_id="gfp"+hashlib.sha256((label+r["task_id"]).encode()).hexdigest()[:20],k=0,user_tools={});orig=hook(rec,PROTOCOL["max_tool_calls"]);status=None;start=time.monotonic()
 try:await asyncio.wait_for(gen.solve(ctx),timeout=PROTOCOL["probe_timeout_seconds"])
 except asyncio.TimeoutError:status="inference_timeout"
 except Exception as e:status="model_server_error";rec.trace["trace_notes"].append(f"{type(e).__name__}:{e}")
 finally:MCPManager._call_tool_async=orig
 steps=node.pass_k_trace.get(0,[])
 if status is None and len(steps)==1:status="model_server_error"
 final=node.pass_k_scenario.get(0,"unknown");natural=bool(steps and steps[-1].get("role")=="assistant");state_ok=compare_final_states(final,{r["environment"]:expected});rec.finalize("success" if natural else "failure",final,state_ok,{"source":"canonical_reference_state","system_status":status or "none"});roll=td/f"{r['task_id']}.rollout.json";rec.write(roll)
 try:p=profile_sidecar_and_trace(side,rec.as_dict());perr=None
 except Exception as e:p={};perr=f"{type(e).__name__}:{e}";status="profiler_error"
 events=rec.trace["events"];successful={e["tool_name"] for e in events if e["execution_success"] is True};nodes=all(x in successful for x in r["gold_tools"]);checks=p.get("dependency_edge_checks",[]);known=[x for x in checks if isinstance(x.get("success"),bool)];edge=len(known)==len(side["dependency_edges"]) and all(x["success"] for x in known);internal=[x for x in p.get("internal_parameter_flow_checks",[]) if isinstance(x.get("success"),bool)];ic=len(internal)==len(side["dependency_edges"]) and all(x["success"] for x in internal);valid=status is None
 result={"schema_version":"graph_frontier_pilot_result_v1","task_id":r["task_id"],"model":label,"runtime_seconds":time.monotonic()-start,"valid_capability_probe":valid,"system_status":status or "none","profiler_error":perr,"task_success":bool(valid and nodes and edge and state_ok is True),"final_state_success":state_ok,"path_adherence":p.get("path_adherence","unknown"),"gold_node_complete":nodes,"edge_complete":edge,"internal_param_complete":ic,"tool_call_count":len(events),"unexpected_calls":len(p.get("extra_tool_calls") or []),"redundant_calls":len(p.get("redundant_tool_calls") or []),"profile":p,"rollout_path":str(roll.resolve().relative_to(ROOT))};rp.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n");return result
async def run(manifest,label,out):
 rows=load(manifest);out.mkdir(parents=True,exist_ok=True);(out/"run_config.json").write_text(json.dumps({"model":label,"model_path":PATHS[label],"manifest":str(manifest),"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"inference":PROTOCOL},indent=2,sort_keys=True)+"\n");ans=[];start=time.monotonic()
 for i,r in enumerate(rows,1):x=await one(r,label,out);ans.append(x);print(f"PILOT_PROGRESS model={label} completed={i}/100 valid={sum(z['valid_capability_probe'] for z in ans)} success={sum(z['task_success'] for z in ans)}",flush=True)
 summary={"model":label,"completed":len(ans),"valid":sum(x["valid_capability_probe"] for x in ans),"task_success":sum(x["task_success"] for x in ans),"runtime_seconds":time.monotonic()-start};(out/"run_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");return summary
def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest="cmd",required=True);f=sp.add_parser("freeze");f.add_argument("--root",type=Path,default=ROOT/"repro_1p7b/results/graph_frontier/pilot_100");f.add_argument("--compact",type=Path,default=ROOT/f"repro_1p7b/graph_frontier/manifests/pilot_100_seed_{SEED}.jsonl");rr=sp.add_parser("run");rr.add_argument("--manifest",type=Path,default=ROOT/f"repro_1p7b/results/graph_frontier/pilot_100/frozen/pilot_100_seed_{SEED}.jsonl");rr.add_argument("--model",choices=tuple(PATHS),required=True);rr.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();register()
 try:print(json.dumps(freeze(a.root,a.compact) if a.cmd=="freeze" else asyncio.run(run(a.manifest,a.model,a.output_dir)),ensure_ascii=False,indent=2))
 finally:MCPManager.shutdown()
if __name__=="__main__":main()
