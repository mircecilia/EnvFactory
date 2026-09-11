import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.manager.mcp_client_manager import MCPManager
from src.utils.data_process import convert_to_sft_data

future = asyncio.run_coroutine_threadsafe(
    MCPManager.register_mcp_server_async("CampusCard", "envs/tools/CampusCard.py", False),
    MCPManager._loop,
)
future.result(timeout=30)
chain = {
    "seed": 42, "mcp_servers": {"CampusCard"}, "user_tools": [],
    "nodes": [{
        "decision": True, "mcp_servers": ["CampusCard"],
        "query": "Please recharge my campus card by 25 CNY using Alipay.",
        "steps": [
            {"role": "user", "content": "Please recharge my campus card by 25 CNY using Alipay."},
            {"role": "tool_call", "content": [{"name": "CampusCard-recharge", "arguments": {"userId": "u1", "amount": 25.0, "paymentMethod": "alipay"}}], "think": "I should use the recharge tool."},
            {"role": "tool_response", "content": [json.dumps({"userId": "u1", "success": True, "balanceAfter": 125.0})]},
            {"role": "assistant", "content": "Your campus card balance is now 125 CNY."},
        ],
    }],
}
output = Path("repro_1p7b/results/smoke/sft_sample.json")
output.parent.mkdir(parents=True, exist_ok=True)
convert_to_sft_data([chain], str(output), shuffle=False, seed=42, enable_think=True)
data = json.loads(output.read_text(encoding="utf-8"))
assert len(data) == 2
assert list(data[0]) == ["instruction", "input", "output", "system", "history"]
assert "<tool_call>" in data[0]["output"]
assert "<tool_response>" in data[1]["instruction"]
assert "CampusCard-recharge" in data[0]["system"]
print(json.dumps({
    "sample_count": len(data), "fields": list(data[0]),
    "first_history_length": len(data[0]["history"]),
    "second_history_length": len(data[1]["history"]),
    "system_has_tool": "CampusCard-recharge" in data[0]["system"],
    "output": str(output),
}, indent=2))
MCPManager.shutdown()
