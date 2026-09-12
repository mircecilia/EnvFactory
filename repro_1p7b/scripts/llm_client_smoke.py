import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.manager.llm_client_manager import LLMClient


def main() -> None:
    expected = "EnvFactory SGLang API passed."
    response = LLMClient.inference(
        "Reply with exactly: EnvFactory SGLang API passed.",
        disable_progress_bar=True,
        temperature=0.0,
        max_tokens=64,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )[0]
    assert response.strip() == expected, response
    print(json.dumps({"response": response}, ensure_ascii=False, indent=2))
    LLMClient.shutdown()


if __name__ == "__main__":
    main()
