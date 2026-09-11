import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import os

# Pure graph operations still instantiate LLMClient at import time.
os.environ.setdefault("CHAT_API_KEY", "placeholder")
os.environ.setdefault("CHAT_URL", "http://127.0.0.1:1/v1")
os.environ.setdefault("CHAT_MODEL", "unused")
os.environ.setdefault("EMBEDDING_API_KEY", "placeholder")
os.environ.setdefault("EMBEDDING_URL", "http://127.0.0.1:1")
os.environ.setdefault("EMBEDDING_MODEL", "unused")

from src.graph.sampler import TopologySampler  # noqa: E402
from src.graph.tool_graph import EdgeType, NodeType, ToolGraph  # noqa: E402
from src.graph.tool_node import Tool  # noqa: E402

balance = Tool({
    "server": "CampusCard", "name": "CampusCard-query_balance", "description": "Retrieve balance",
    "input_schema": {"type": "object", "properties": {"userId": {"type": "string", "description": "account id"}}, "required": ["userId"]},
    "output_schema": {"type": "object", "properties": {"userId": {"type": "string", "description": "account id"}, "balance": {"type": "number", "description": "current balance"}}},
})
recharge = Tool({
    "server": "CampusCard", "name": "CampusCard-recharge", "description": "Recharge account",
    "input_schema": {"type": "object", "properties": {"userId": {"type": "string", "description": "account id"}, "amount": {"type": "number", "description": "amount"}, "paymentMethod": {"type": "string", "description": "payment method"}}, "required": ["userId", "amount", "paymentMethod"]},
    "output_schema": {"type": "object", "properties": {}},
})
graph = ToolGraph()
graph.server_to_tools = {"CampusCard": [balance, recharge]}
for tool in (balance, recharge):
    graph.graph.add_node(tool, node_type=NodeType.Tool)
    for param in tool.input_schema["parameters"]:
        graph.graph.add_node(param, node_type=NodeType.Parameter)
        graph.graph.add_edge(param, tool, edge_type=EdgeType.Tool_Input, required=True)
    for param in tool.output_schema["parameters"]:
        graph.graph.add_node(param, node_type=NodeType.Parameter)
        graph.graph.add_edge(tool, param, edge_type=EdgeType.Tool_Output)
for param in balance.input_schema["parameters"]:
    param.set_user_provided(True)
for param in recharge.input_schema["parameters"]:
    param.set_user_provided(param.name in {"amount", "paymentMethod"})
out_user = next(p for p in balance.output_schema["parameters"] if p.name == "userId")
in_user = next(p for p in recharge.input_schema["parameters"] if p.name == "userId")
graph.graph.add_edge(out_user, in_user, edge_type=EdgeType.Parameter_Relate)
graph.graph.add_edge(balance, recharge, edge_type=EdgeType.Tool_Depend)
sampler = TopologySampler(max_servers=1, max_recursion_depth=5)
priors = sampler.sample_prior(graph, recharge, visited_nodes=[])
chain = graph.sample(sampler, max_nodes=2, start_node=balance, seed=42)
result = {
    "node_count": graph.graph.number_of_nodes(), "edge_count": graph.graph.number_of_edges(),
    "recharge_priors": [tool.name for tool in priors],
    "sampled_chain": [tool.name for tool in chain.init_tool_chain],
    "chain_valid": graph.validate_tool_chain(chain.init_tool_chain),
}
assert result["recharge_priors"] == ["CampusCard-query_balance"]
assert result["sampled_chain"] == ["CampusCard-query_balance", "CampusCard-recharge"]
assert result["chain_valid"] is True
print(json.dumps(result, indent=2))
