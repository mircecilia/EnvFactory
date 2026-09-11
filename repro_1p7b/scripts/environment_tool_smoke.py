import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.manager.mcp_client_manager import MCPManager

future = asyncio.run_coroutine_threadsafe(
    MCPManager.register_mcp_server_async("CampusCard", "envs/tools/CampusCard.py", False),
    MCPManager._loop,
)
future.result(timeout=30)
client_id = "CampusCard-smoke"
scenario = {
    "accounts": {"u1": {"userId": "u1", "name": "Alice", "password": "pw", "balance": 100.0, "currency": "CNY", "status": 1}},
    "transactions": {"u1": []},
    "statusTextMap": {"1": "normal", "2": "lost", "3": "system frozen", "4": "closed", "5": "pre-closed", "6": "manually frozen"},
    "current_time": "2026-09-11 22:30:00",
}
loaded = MCPManager.load_scenario(client_id, scenario, check=True)
before = json.loads(MCPManager.call_tool(client_id, "save_scenario", {}))
result = json.loads(MCPManager.call_tool(client_id, "CampusCard-recharge", {"userId": "u1", "amount": 25.0, "paymentMethod": "alipay"}))
after = json.loads(MCPManager.call_tool(client_id, "save_scenario", {}))
assert loaded == "Successfully loaded scenario"
assert before["accounts"]["u1"]["balance"] == 100.0
assert result["success"] is True
assert after["accounts"]["u1"]["balance"] == 125.0
assert len(after["transactions"]["u1"]) == 1
print(json.dumps({
    "available_tools": [tool["function"]["name"] for tool in MCPManager.filter_tools(["CampusCard"])],
    "before_balance": before["accounts"]["u1"]["balance"],
    "tool_result": result,
    "after_balance": after["accounts"]["u1"]["balance"],
    "after_transactions": after["transactions"]["u1"],
}, indent=2))
MCPManager.close_client(client_id=client_id)
MCPManager.shutdown()
