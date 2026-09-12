import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "repro_1p7b" / "models" / "Qwen3-1.7B"


def main() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    messages = [{"role": "user", "content": "Reply with exactly: EnvFactory model smoke passed."}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.eval()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=24,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    result = {
        "model_dir": str(MODEL_DIR),
        "model_type": model.config.model_type,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "input_token_ids_prefix": inputs["input_ids"][0, :16].tolist(),
        "generated_token_count": int(generated.shape[0]),
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
