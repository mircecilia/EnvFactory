# Repository map

Audited commit: `298e4744d329b2f88dc16347cc7734348d40464f`.

## Environment construction

| Stage | Entry | Function/class | Input -> output |
|---|---|---|---|
| Discover | `.agents/skills/mcp-sketch-discovery/SKILL.md`, `envs/schema_sketch/` | discovery workflow | API research -> schema sketch and note |
| Metadata | `src/gen/mcp_schema_gen.py` | module CLI | sketch -> metadata JSON |
| Generate | `src/gen/env_gen/__main__.py`, `env_gen.py` | `main()`, `EnvGen.generate_mcp_env()` | metadata -> MCP server/scenarios/validation |
| Validate | `src/gen/env_gen/validate_revise.py` | validation state machine | generated artifacts -> evidence/revision |
| Persist | `src/gen/env_gen/env_gen.py` | `save_checkpoint()`, `_save_intermediate()` | state -> `envs/intermediate/` checkpoint |

Executable servers are in `envs/tools/`; registration is `configs/mcp_server.json`. `MCPClientManager` registers/spawns servers, isolates sessions, loads/saves scenarios, and calls tools.

## ToolGraph

`src/graph/tool_node.py::Tool` parses input/output schemas into `Parameter` nodes. `batch_embedding()` embeds names/descriptions; `batch_get_user_provided()` classifies user-supplied parameters via LLM. `src/graph/tool_graph.py::ToolGraph.build_tool_graph()` builds a NetworkX DiGraph. `_build_edge_with_sim()` creates parameter/tool dependencies from embedding similarity; `_build_edge_with_llm()` refines same-server edges. `validate_parameter()` and `validate_tool_chain()` enforce producer-before-consumer. `save()`/`load()` use `graph.pkl`.

Node types: Tool, Parameter. Edge types: parameter-to-tool input, tool-to-parameter output, parameter-to-parameter semantic relation, and tool-to-tool dependency. Official construction example: `examples/load_tool_graph.ipynb`.

## Topology-aware sampling

`ToolGraph.sample()` seeds Python `random`, chooses a random target when none is supplied, and performs bounded BFS with `sample_prior()` before `sample()`. `src/graph/sampler.py::TopologySampler` recursively searches predecessor tools, uses random choice, skips optional parameters with probability 0.6, skips already-valid parameters with probability 0.9, defaults to at most 3 servers and recursion depth 5, and uniformly samples 1..all successors.

The sampler consumes no current-model capability, failure rate, explicit difficulty, environment redundancy, or trajectory novelty signal.

## QueryGen and trajectory

`examples/sythesize_query.py` loads `graph.pkl`, samples with `TopologySampler(max_nodes=15)`, routes through `QueryGen`, and defaults to N=200/concurrency 5. `src/gen/query_gen/__init__.py::QueryGen` selects conversational or non-conversational implementation.

`QueryGenNonConv.gen()` runs Preparing -> Starting -> Generating -> optional Refining -> Solving -> Terminated. Important calls: `prepare()`/`split_turns()`, `schema_generate()`, `generate()`, `refine()`, `solve()`, `solve_and_select()`, `terminate()`. `solve()` executes MCP calls; pass@k traces are selected before persistence.

`src/graph/tool_chain.py::ToolQueryNode` stores target tools, initial/final scenarios, query, steps, pass@k traces/decisions, and accuracy. `ToolQueryChain.save()` writes trajectory JSON.

## Data processing

`examples/process_data.sh` invokes `src/utils/data_process.py`. `load_tool_chains()` reads JSON. `convert_to_sft_data()` emits one Alpaca sample per input/output pair with instruction, empty input, output, system, and history. `format_step()` serializes `<tool_call>`/`<tool_response>`. `convert_to_rl_data()` emits one RL sample per valid node with prompt history, ground-truth tool calls, and initial/final scenarios.

## Training and evaluation

SFT is external LlamaFactory configured by `configs/llamafactory_sft.yaml`. RL is an external forked VeRL absent from this checkout. No executable BFCL, MCP-Atlas, tau2-Bench, or VitaBench evaluation pipeline exists here; only documentation/model-card claims reference them.
