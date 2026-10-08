"""Read-only BFCL historical audit. Never imports/runs a model or executor.

Output is a NEW analysis namespace; official result/score files are read only.
Primary labels are evidence-local observations in the first officially failing
turn, not claims of an independently replayed unique causal first divergence.
Final state mismatch alone is UNKNOWN. Category names never supply labels.
"""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import json
import random
import re
from pathlib import Path

CATEGORIES = ("base", "miss_func", "miss_param", "long_context")
UNKNOWN = "UNKNOWN"
LABELS = ("TOOL_SELECTION", "ARGUMENT_ERROR", "PARAMETER_PROPAGATION",
          "MULTI_STEP_EXECUTION", "INFORMATION_HANDLING", "TERMINATION",
          "OUTPUT_FORMAT", "ENVIRONMENT_EVALUATION", UNKNOWN)
RUNS = {
    **{x: "repro_1p7b/evaluation/bfcl/artifacts/full/" + x for x in
       ("base", "sft", "parameter_aware", "dynamic_v1", "dynamic_v2",
        "complete_c_pi3", "complete_c_pi4")},
    "standard_grpo": "repro_1p7b/graph_frontier/evals/standard_rl_v2_control_step64/bfcl",
    "graph_grpo": "repro_1p7b/graph_frontier/evals/graph_frontier_rl_v2_step64/bfcl",
}
SCHEMA = "bfcl_failure_driven_audit_v1"
SEED = 20261008


def stable(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                yield line_number, json.loads(line)


def call_ast(value):
    """No eval. Nonliteral/variadic expressions remain undecidable."""
    if not isinstance(value, str):
        return None
    try:
        x = ast.parse(value, mode="eval").body
        if not isinstance(x, ast.Call) or not isinstance(x.func, ast.Name):
            return None
        if any(k.arg is None for k in x.keywords):
            return None
        return {"name": x.func.id,
                "arguments": {k.arg: ast.literal_eval(k.value) for k in x.keywords},
                "positional": [ast.literal_eval(a) for a in x.args],
                "raw": value}
    except (SyntaxError, ValueError, TypeError):
        return None


def offered(prompt):
    """Use the actual per-request offered tools, not all environment functions."""
    # The system instruction contains a literal empty placeholder before
    # the actual advertisement. User messages cannot define offered tools.
    text = prompt or ""
    if text.startswith("<|im_start|>system"):
        text = text.split("<|im_end|>", 1)[0]
    blocks = re.findall(r"<tools>\s*(.*?)\s*</tools>", text, re.S)
    for block in blocks:
        if not block.strip():
            continue
        result = {}
        try:
            for line in block.splitlines():
                if line.strip():
                    tool = json.loads(line)
                    result[tool["name"]] = tool
            return result
        except (ValueError, KeyError, TypeError):
            return None
    return {} if blocks else None


def response_json(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        # Some official executors stringify a Python dict/list, not JSON.
        # literal_eval never executes code; arbitrary text remains text.
        if value.lstrip().startswith(("{", "[")):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except (ValueError, SyntaxError, TypeError):
                pass
        return value


def values_with_paths(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from values_with_paths(v, path + "." + str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from values_with_paths(v, path + "." + str(i))
    else:
        yield path.lstrip("."), value


def feedback_error(value):
    v = response_json(value)
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
        v = v[0]
    if isinstance(v, dict):
        for key in ("error", "error_message", "exception"):
            if v.get(key):
                return str(v[key])
        if v.get("success") is False and v.get("message"):
            return str(v["message"])
    if isinstance(v, str) and re.search(
            r"^(Error(?: during execution)?:|Failed to execute|Exception:)|TypeError:|Traceback", v):
        return v
    return None


def summarize_response(value):
    v = response_json(value)
    if feedback_error(value) or len(stable(v)) <= 1200:
        return v
    # Manual review sees full error feedback; large successful inventories are
    # summarized with a hash and their typed structure, not silently truncated.
    return {"large_response_omitted": True, "sha256": hashlib.sha256(
            stable(v).encode()).hexdigest(), "type": type(v).__name__,
            "keys": list(v)[:30] if isinstance(v, dict) else None,
            "items": len(v) if isinstance(v, (dict, list)) else None}


def trajectory(row, task=None):
    events, query_turns, ends = [], [], []
    turn = -1
    for entry in row.get("inference_log", []) or []:
        if not isinstance(entry, dict) or "begin_of_turn_query" not in entry:
            continue
        turn += 1
        query = entry["begin_of_turn_query"]
        query_turns.append(query)
        for step_key, records in entry.items():
            if not step_key.startswith("step_") or not isinstance(records, list):
                continue
            step = int(step_key.split("_")[-1])
            schemas = None
            for record in records:
                if record.get("role") == "inference_input":
                    schemas = offered(record.get("content", {}).get("formatted_prompt", ""))
            # Prompting-mode BFCL adds held-out functions in an authoritative
            # user turn, not by replacing the original system <tools> block.
            # Use the official task release map; never arbitrary user text.
            if schemas is not None:
                schemas = dict(schemas)
                for release, functions in (task or {}).get("missed_function", {}).items():
                    if int(release) <= turn:
                        for function in functions:
                            name = function if isinstance(function, str) else function.get("name")
                            if name:
                                schemas[name] = {"name": name}
            decoded, decode_errors, tool_responses = [], [], []
            for record in records:
                role = record.get("role")
                if role == "handler_log":
                    decoded.extend(record.get("model_response_decoded", []) or [])
                    content = str(record.get("content", ""))
                    if re.search(r"decode.*error|failed.*decode|error.*decod", content, re.I):
                        decode_errors.append(content)
                    if "Empty response from the model" in content:
                        ends.append({"turn": turn, "step": step, "reason": content})
                elif role == "tool":
                    tool_responses.append(record.get("content"))
            if decode_errors:
                events.append({"turn": turn, "step": step, "index": -1,
                               "format_error": decode_errors[0], "call": None,
                               "response": None, "schemas": schemas})
            # Decoder log with no tool execution is not a fabricated execution.
            for i, raw in enumerate(decoded):
                events.append({"turn": turn, "step": step, "index": i,
                               "call": call_ast(raw), "decoded_raw": raw,
                               "response": tool_responses[i] if i < len(tool_responses) else None,
                               "executed": i < len(tool_responses), "schemas": schemas})
    return events, query_turns, ends


def failed_turn(failure):
    if not failure:
        return None
    error = failure.get("error", {})
    message = stable(error.get("error_message", ""))
    match = re.search(r"turn (\d+)", message)
    if match:
        return int(match.group(1))
    results = error.get("execution_result") or error.get("details", {}).get("execution_result")
    if isinstance(results, list) and results:
        return len(results) - 1
    return None


def provenance(call, failure, turn, preceding):
    """Only unique reference argument values observed in actual student feedback.
    No inference from names, graph depth, reference-only responses, or prose.
    """
    gold = failure.get("possible_answer", []) or []
    reference = gold[turn] if turn is not None and turn < len(gold) else []
    matches = [a for s in reference if (a := call_ast(s)) and a["name"] == call["name"]]
    evidence = []
    for argument, actual in call["arguments"].items():
        expected = {stable(a["arguments"][argument]): a["arguments"][argument]
                    for a in matches if argument in a["arguments"]}
        if len(expected) != 1:
            continue
        correct = next(iter(expected.values()))
        if type(actual) is not type(correct):
            # String/int coercion is a type/value error, not source selection.
            continue
        if type(actual) is type(correct) and actual == correct:
            continue
        # Low-information scalar coincidences are not provenance.
        if not isinstance(correct, (str, int)) or isinstance(correct, bool) or len(str(correct)) < 3:
            continue
        for distance, producer in enumerate(reversed(preceding), 1):
            if not producer.get("executed") or not producer.get("call") or feedback_error(producer.get("response")):
                continue
            for path, value in values_with_paths(response_json(producer.get("response"))):
                field = path.split(".")[-1]
                same_field = field.lower().replace("_", "") == argument.lower().replace("_", "")
                if same_field and type(value) is type(correct) and value == correct:
                    evidence.append({"producer_tool": producer["call"]["name"],
                                     "producer_field": path, "consumer_tool": call["name"],
                                     "consumer_argument": argument,
                                     "expected": correct, "observed_argument": actual,
                                     "distance_in_tool_calls": distance,
                                     "source": "PREVIOUS_TOOL_OBSERVATION"})
                    break
            if evidence and evidence[-1]["consumer_argument"] == argument:
                break
    return evidence


def classify_event(event, failure, preceding):
    if event.get("format_error"):
        return "OUTPUT_FORMAT", "DECODE_ERROR", event["format_error"], []
    call = event.get("call")
    if not call:
        return None
    schemas = event.get("schemas")
    if schemas is not None and call["name"] not in schemas:
        return "TOOL_SELECTION", "UNAVAILABLE_NAME", "Decoded function is absent from actual offered schema", []
    if not event.get("executed"):
        return None
    error = feedback_error(event.get("response"))
    if not error:
        return None
    if re.search(r"NameResolutionError|Network is unreachable|ConnectTimeout|ReadTimeout|Connection timed out", error):
        return "ENVIRONMENT_EVALUATION", "NETWORK_UNAVAILABLE", error, []
    if re.search(r"missing .*required.*argument|required positional argument|unexpected keyword argument|too many arguments", error, re.I):
        return "ARGUMENT_ERROR", "MISSING_OR_EXTRA_ARGUMENT", error, []
    if re.search(r"must be (?:a |an )?(?:str|int|float|list|dict|boolean)|expected (?:type|str|int|float)|invalid (?:type|format)|TypeError|not supported between instances|could not convert|should be .*string", error, re.I):
        return "ARGUMENT_ERROR", "TYPE_OR_FORMAT", error, []
    prop = provenance(call, failure, event["turn"], preceding)
    if prop:
        return "PARAMETER_PROPAGATION", "OBSERVED_PRODUCER_VALUE_NOT_USED", error, prop
    if re.search(r"not (?:authenticated|logged in|connected)|must (?:authenticate|log in|login|connect)|authenticate first|login first|authentication required|connection not established|not initialized", error, re.I):
        return "MULTI_STEP_EXECUTION", "EXECUTION_PRECONDITION", error, []
    if re.search(r"(?:invalid|unknown|not found|does not exist).*(?:id|ticker|symbol|parameter|argument)|(?:id|ticker|symbol).*?(?:invalid|unknown|not found|does not exist)|invalid (?:price|quantity|amount|value)", error, re.I):
        return "ARGUMENT_ERROR", "EXECUTION_REJECTED_VALUE", error, []
    # Generic not-found, auth rejection, and filesystem errors are ambiguous
    # between wrong values, state grounding and missing prerequisites.
    return UNKNOWN, "EXECUTION_ERROR_UNRESOLVED", error, []


def classify(row, failure):
    events, queries, ends = trajectory(row, (failure or {}).get("prompt", {}))
    labels = []
    for i, event in enumerate(events):
        # A batch is authored before any of its tool feedback is visible.
        # Serial execution within that batch is not an observed provenance edge.
        visible = [p for p in events[:i] if (p["turn"], p["step"]) <
                   (event["turn"], event["step"])]
        result = classify_event(event, failure or {}, visible)
        if result:
            kind, subtype, text, prop = result
            labels.append({"label": kind, "subtype": subtype, "turn": event["turn"],
                           "step": event["step"], "call_index": event["index"],
                           "call": event.get("call"), "feedback": text,
                           "provenance": prop})
    if not failure:
        return {"primary": "SUCCESS", "first_observable_error": labels[0] if labels else None,
                "secondary": sorted({v["label"] for v in labels}),
                "failed_turn": None, "events": events, "queries": queries, "ends": ends}
    error = failure.get("error", {})
    et = error.get("error_type", "UNKNOWN")
    ft = failed_turn(failure)
    candidates = [x for x in labels if ft is not None and x["turn"] == ft]
    first = candidates[0] if candidates else None
    if first and first["label"] != UNKNOWN:
        primary = first["label"]
        why = first["subtype"]
    elif et == "multi_turn:irrelevance_error:decoder_success":
        primary, why = "INFORMATION_HANDLING", "OFFICIAL_IRRELEVANCE_VIOLATION"
    elif et == "multi_turn:empty_turn_model_response":
        primary, why = "TERMINATION", "NO_REQUIRED_CALL_IN_OFFICIALLY_FAILING_TURN"
    elif et == "multi_turn:force_terminated":
        # Forced truncation does not identify a unique loop/early-stop cause.
        primary, why = UNKNOWN, "BUDGET_OR_CONTEXT_CENSORED"
    else:
        primary, why = UNKNOWN, "FINAL_STATE_OR_RESPONSE_NOT_A_MECHANISM"
    return {"primary": primary, "subtype": why, "failed_turn": ft,
            "first_observable_error": labels[0] if labels else None,
            "failed_turn_observation": first,
            "secondary": sorted({v["label"] for v in labels}),
            "events": events, "queries": queries, "ends": ends,
            "official_error": et, "official_message": error.get("error_message"),
            "causal_first_divergence": "UNKNOWN_NOT_REPLAYED",
            "trajectory_complete_for_saved_result": bool(events or queries),
            "provenance": first.get("provenance", []) if first else []}


def row_summary(label, category, row, failure, path, line, source_hash):
    c = classify(row, failure)
    success = failure is None
    # Overflow result messages have no retained turn logs.
    no_trace = not row.get("inference_log")
    if not success and no_trace:
        primary, subtype = (("ENVIRONMENT_EVALUATION", "CONTEXT_WINDOW_REJECTION")
                            if re.search(r"context|length|token", stable(row.get("result")), re.I)
                            else (UNKNOWN, "MISSING_TRAJECTORY"))
        c.update(primary=primary, subtype=subtype)
    compact = {k:v for k,v in c.items() if k not in ("events", "queries", "ends")}
    compact.update(model=label, category=category, task_id=row["id"], success=success,
                   trajectory_available=not no_trace,
                   tool_event_count=sum(e.get("executed", False) for e in c["events"]),
                   turn_count=len(c["queries"]), evidence_source=str(path),
                   evidence_line=line, source_sha256=source_hash,
                   attribution_basis="observed failing-turn mechanism, not causally verified first divergence",
                   confidence="DIRECT_EVENT_EVIDENCE" if c["primary"] not in (UNKNOWN, "SUCCESS") else "NOT_ATTRIBUTED")
    return compact, c


def model_logs(root, label, run):
    if label in ("standard_grpo", "graph_grpo"):
        paths = [run/"logs/server.log"]
    else:
        d = root/"repro_1p7b/evaluation/bfcl/logs/full"
        paths = sorted(d.glob(label + "*server.log"))
    configs = []
    fields = ("model_path", "random_seed", "context_length", "tp_size", "dtype",
              "attention_backend", "sampling_backend", "mem_fraction_static",
              "max_total_tokens", "enable_deterministic_inference")
    for p in paths:
        if not p.exists():
            continue
        with p.open(errors="replace") as f:
            text = f.read(250000)
        config = {"source":str(p), "source_sha256":sha(p)}
        for key in fields:
            match = re.search(r"\b"+key+r"=(?:'([^']*)'|([^,\)]+))", text)
            config[key] = (match.group(1) if match.group(1) is not None else match.group(2)) if match else "UNKNOWN"
        configs.append(config)
    return configs


def compact_trace(row, failure, classification):
    events = classification["events"]
    return {"task_id":row["id"], "queries":classification["queries"],
            "possible_answer": failure.get("possible_answer", []) if failure else [],
            "official_error":failure.get("error", {}) if failure else None,
            "timeline":[{k:v for k,v in e.items() if k not in ("schemas","response")} |
                        {"response":summarize_response(e.get("response"))} for e in events],
            "stops":classification["ends"],
            "initial_config": failure.get("prompt", {}).get("initial_config", {}),
            "notice":"All saved decoded actions/executed feedback included. Large successful responses summarized/hash-referenced; no reasoning or full prompt copied. Error feedback unabridged. No new execution."}


def paired(left, right, records, n=10000):
    a={r["task_id"]:r for r in records if r["model"]==left}
    b={r["task_id"]:r for r in records if r["model"]==right}
    ids=sorted(a.keys() & b.keys())
    delta=[int(b[t]["success"])-int(a[t]["success"]) for t in ids]
    count=collections.Counter(delta)
    rng=random.Random(SEED)
    # Multinomial counts via bootstrap task sampling, preserving pairs.
    boot=sorted(sum(delta[rng.randrange(len(delta))] for _ in delta)/len(delta)
                for _ in range(n)) if delta else []
    fixed=collections.Counter(a[t]["primary"] for t in ids if not a[t]["success"] and b[t]["success"])
    lost=collections.Counter(b[t]["primary"] for t in ids if a[t]["success"] and not b[t]["success"])
    return {"left":left,"right":right,"paired_tasks":len(ids),
            "match":"COMPATIBLE_NOT_STRICT_SEEDS_UNCONTROLLED",
            "causal_attribution_allowed":False,
            "win":count[1],"loss":count[-1],"tie":count[0],
            "observed_accuracy_delta":sum(delta)/len(delta) if delta else None,
            "task_bootstrap_CI95":[boot[int(.025*n)],boot[int(.975*n)]] if boot else None,
            "CI_caveat":"task variability only; excludes stochastic between-run variance",
            "fixed_previous_observed_types":dict(fixed),"new_failed_observed_types":dict(lost)}


def run(root, output, hash_models=True):
    root=Path(root).resolve();output=Path(output).resolve()
    assert output.is_relative_to(root) and not output.exists(), "new output namespace required"
    output.mkdir(parents=True)
    records=[]; inventory={}; cache_hash={}; source_files={}
    for label, relative in RUNS.items():
        run_dir=root/relative; failures={}; scores={}; score_inputs=[]
        for category in CATEGORIES:
            matches=list((run_dir/"score").glob("*/BFCL_v3_multi_turn_"+category+"_score.json"))
            if len(matches)!=1:
                continue
            p=matches[0];h=sha(p);source_files[str(p)]=h;score_inputs.append({"path":str(p),"sha256":h})
            for line,x in jsonl(p):
                if line==1:
                    scores[category]=x
                else:
                    failures.setdefault(x["id"],[]).append(x)
            uniq={t for t in failures if t.startswith("multi_turn_"+category+"_")}
            assert len(uniq)==scores[category]["total_count"]-scores[category]["correct_count"], "score mismatch"
        result_paths=sorted((run_dir/"result").glob("*/BFCL_v3_multi_turn_*_result.json"))
        result_input=[];ids=[];rows_for_cases={}
        for p in result_paths:
            category=p.name.removeprefix("BFCL_v3_multi_turn_").removesuffix("_result.json")
            if category not in CATEGORIES:
                continue
            h=sha(p);source_files[str(p)]=h;result_input.append({"path":str(p),"sha256":h})
            for line,x in jsonl(p):
                ids.append(x["id"])
                failure=failures.get(x["id"],[None])[0]
                compact,c=row_summary(label,category,x,failure,p,line,h)
                if category not in scores:
                    compact.update(success=None,primary=UNKNOWN,subtype="MISSING_OFFICIAL_SCORE")
                records.append(compact)
                if failure:
                    rows_for_cases[x["id"]]=(x,failure,c)
        logs=model_logs(root,label,run_dir);weights={}
        for cfg in logs:
            model=Path(cfg["model_path"])
            if hash_models and model.is_dir():
                for p in sorted(model.glob("*.safetensors")):
                    if str(p) not in cache_hash:cache_hash[str(p)]=sha(p)
                    weights[str(p)]={"sha256":cache_hash[str(p)],"bytes":p.stat().st_size}
        manifest_path=run_dir/"runtime_manifest.json"
        inventory[label]={"run":str(run_dir),"scores":scores,"score_inputs":score_inputs,
            "result_inputs":result_input,"count":len(ids),"unique_ids":len(set(ids)),
            "id_set_sha256":hashlib.sha256(stable(sorted(set(ids))).encode()).hexdigest(),
            "failures_unique":len(failures),"duplicate_failure_rows":sum(len(v)-1 for v in failures.values()),
            "all800":len(ids)==800 and len(set(ids))==800 and len(scores)==4,
            "result_missing_failed_ids":sorted(set(failures)-set(ids)),
            "configs":logs,"weights":weights,
            "manifest":json.loads(manifest_path.read_text()) if manifest_path.exists() else None,
            "temperature":.7,"temperature_evidence":"shared historical driver/official request path; runtime seed not unified",
            "request_seed":"NOT_PASSED","comparability":"COMPATIBLE" if len(set(ids))==800 else "NOT-COMPARABLE"}
        # Retain selected compact traces per run for deterministic review sampling.
        case_dir=output/"candidate_traces"/label;case_dir.mkdir(parents=True)
        buckets=collections.defaultdict(list)
        for r in records:
            if r["model"]==label and r["success"] is False:
                buckets[r["primary"]].append(r["task_id"])
        for kind,task_ids in buckets.items():
            task_ids.sort(key=lambda t:hashlib.sha256((str(SEED)+label+t).encode()).hexdigest())
            for task_id in task_ids[:8]:
                x,f,c=rows_for_cases[task_id]
                (case_dir/(task_id+".json")).write_text(stable(compact_trace(x,f,c)),encoding="utf-8")
        print(stable({"finished_model":label,"tasks":len(ids),"scores":scores}),flush=True)
        del rows_for_cases
    assert all(v["all800"] for v in inventory.values()), "partial/missing run: do not silently infer successes"
    assert len({v["id_set_sha256"] for v in inventory.values()})==1, "task sets differ"
    counts={}
    for label in RUNS:
        rs=[r for r in records if r["model"]==label]
        counts[label]={"success":sum(r["success"] for r in rs),"failure":sum(not r["success"] for r in rs),
            "primary":dict(collections.Counter(r["primary"] for r in rs if not r["success"])),
            "by_category":{c:dict(collections.Counter(r["primary"] for r in rs if r["category"]==c and not r["success"])) for c in CATEGORIES},
            "full_trace_tasks":sum(r["trajectory_available"] for r in rs),
            "attributed_failures":sum(r["primary"] not in (UNKNOWN,"SUCCESS") for r in rs)}
    contrasts=[("base","sft"),("sft","dynamic_v1"),("parameter_aware","dynamic_v1"),
               ("dynamic_v1","standard_grpo"),("dynamic_v1","complete_c_pi3"),("dynamic_v1","complete_c_pi4")]
    summary={"schema":SCHEMA,"audit_source_sha256":sha(Path(__file__)),"seed":SEED,"bootstrap_resamples":10000,"inventory":inventory,
             "counts":counts,"comparisons":[paired(a,b,records) for a,b in contrasts],
             "no_gpu":True,"no_executor":True,"no_training":True,
             "source_immutable_hashes":source_files,"manual_review":"PENDING",
             "strict_matched_pairs":0}
    (output/"per_task_failure_analysis.jsonl").write_text(
        "\n".join(stable(r) for r in records)+"\n",encoding="utf-8")
    (output/"audit_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    for p,h in source_files.items():assert sha(p)==h,"historical artifact changed during analysis"
    (output/"COMPLETE.json").write_text(stable({"status":"CPU_AUDIT_COMPLETE_REVIEW_PENDING",
         "record_count":len(records),"source_hashes_verified":len(source_files)}),encoding="utf-8")
    print(stable({"status":"CPU_AUDIT_COMPLETE","records":len(records),"counts":counts}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True);parser.add_argument("--skip-model-hashes",action="store_true")
    args=parser.parse_args();run(args.root,args.output,not args.skip_model_hashes)
